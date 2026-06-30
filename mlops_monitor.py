"""
mlops_monitor.py
----------------
MLOps (Machine Learning Operations) — Drift Detection & Retraining Pipeline.

WHAT IS MODEL DRIFT?
  Over time, real-world patterns change (new hospital wing opens, disease outbreak,
  policy change). When this happens, a model trained on old data becomes inaccurate.
  This is called "concept drift" and is a key challenge in production ML systems.

HOW DO WE DETECT DRIFT?
  Every monitoring run we compute a "rolling MAPE" — the model's actual error
  over the most recent 4 weeks — and compare it to the original baseline MAPE.
  If rolling MAPE exceeds baseline by more than 5 percentage points, we flag drift
  and can trigger automatic retraining.

HITL (Human-In-The-Loop) FEEDBACK:
  When doctors/managers approve or reject transfer recommendations in the dashboard,
  those decisions are logged. This module reads that log and adjusts the W2
  (transport score) weight for each route:
  - Routes rejected for logistical reasons repeatedly → W2 penalised (−0.05)
  - Routes consistently approved (>80%) → W2 boosted (+0.02)

Usage:
  python src/mlops_monitor.py --simulate-check   # check for drift, no actual retraining
  python src/mlops_monitor.py --retrain-all       # force retrain all 6 models
"""

import os, sys, json, warnings, argparse
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"; os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
from datetime import datetime
import numpy as np, pandas as pd

SRC_DIR  = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SRC_DIR)
if SRC_DIR not in sys.path: sys.path.insert(0, SRC_DIR)

# Output file paths
REPORTS_DIR = os.path.join(BASE_DIR, "outputs", "reports")
DATA_DIR    = os.path.join(BASE_DIR, "data")
os.makedirs(REPORTS_DIR, exist_ok=True)
HISTORY_CSV = os.path.join(REPORTS_DIR, "mlops_history.csv")   # log of every monitoring run
HEALTH_TXT  = os.path.join(REPORTS_DIR, "mlops_health_report.txt")  # human-readable summary
W2_STORE    = os.path.join(REPORTS_DIR, "route_weights.json")  # adjusted W2 weights from HITL

# All hospital-blood_group combinations to monitor
COMBOS = [("Hospital_A","O_positive"), ("Hospital_A","O_negative"),
          ("Hospital_B","O_positive"), ("Hospital_B","O_negative"),
          ("Hospital_C","O_positive"), ("Hospital_C","O_negative")]

# Tuning parameters
DRIFT_THRESHOLD_PCT = 5.0   # flag drift if rolling MAPE exceeds baseline by more than this
REJECTION_PENALTY   = 0.05  # reduce W2 by this much when a route is often rejected
APPROVAL_BOOST      = 0.02  # increase W2 by this much when a route is consistently approved
W2_MIN, W2_MAX      = 0.1, 1.5  # hard limits on W2 to prevent extremes


# ── Small utility functions ─────────────────────────────────────────────────
def _mape(actual, predicted):
    """Mean Absolute Percentage Error — our main accuracy metric."""
    mask = actual > 1
    return float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100) if mask.sum() else 0.0

def _load_baseline():
    """Load the original training MAPE for each combo from model_comparison.csv."""
    p = os.path.join(REPORTS_DIR, "model_comparison.csv")
    if not os.path.exists(p): return pd.DataFrame()
    df = pd.read_csv(p)
    return df[df["model"] == "Hybrid"][["hospital", "blood_group", "MAPE"]].copy()

def _load_route_weights():
    """Load previously saved W2 adjustments, or return an empty dict if none exist."""
    if os.path.exists(W2_STORE):
        with open(W2_STORE) as f: return json.load(f)
    return {}

def _save_route_weights(w):
    """Persist W2 weight adjustments to JSON for the next run."""
    with open(W2_STORE, "w") as f: json.dump(w, f, indent=2)

def _append_history(row):
    """Append one monitoring check result to the rolling history CSV."""
    cols = ["timestamp", "hospital", "blood_group", "baseline_mape", "rolling_mape", "drift_detected", "retrained"]
    hist = pd.read_csv(HISTORY_CSV) if os.path.exists(HISTORY_CSV) else pd.DataFrame(columns=cols)
    pd.concat([hist, pd.DataFrame([row])], ignore_index=True).to_csv(HISTORY_CSV, index=False)


# ── Core monitoring functions ────────────────────────────────────────────────
def compute_rolling_mape(df, forecasters, weeks_back=4):
    """
    For each hospital-blood_group combo:
    1. Re-run rolling forecasts over the last `weeks_back` weeks
    2. Compare the live MAPE to the baseline MAPE from training
    3. Flag as drifted if the gap exceeds DRIFT_THRESHOLD_PCT
    4. Log the result to mlops_history.csv

    Returns a DataFrame with one row per combo showing baseline vs rolling MAPE.
    """
    baseline = _load_baseline()
    results  = []
    end_date   = df["date"].max()
    start_date = end_date - pd.Timedelta(weeks=weeks_back)

    for h, bg in COMBOS:
        hf = forecasters.get((h, bg))
        if hf is None: continue

        sub = df[(df["hospital"] == h) & (df["blood_group"] == bg)].sort_values("date")
        actuals, preds = [], []

        # Step through the last N weeks, one week at a time
        for ref in pd.date_range(start=start_date, end=end_date - pd.Timedelta(days=7), freq="7D"):
            hist_slice = sub[sub["date"] <= ref]
            if len(hist_slice) < 30: continue   # need at least 30 days for LSTM window

            future = sub[(sub["date"] > ref) & (sub["date"] <= ref + pd.Timedelta(days=7))]
            if future.empty: continue

            try:
                hf._prophet._last_date = ref
                fc = hf.predict(hist_slice, target_date=ref)
                n  = min(len(fc), len(future))
                preds.extend(fc["hybrid_pred"].values[:n])
                actuals.extend(future["demand"].values[:n])
            except:
                continue   # skip if prediction fails for this window

        if not actuals: continue

        rolling_mape = _mape(np.array(actuals), np.array(preds))

        # Get baseline MAPE from the original training run
        br = baseline[(baseline["hospital"] == h) & (baseline["blood_group"] == bg)]
        baseline_mape = float(br["MAPE"].iloc[0]) if not br.empty else rolling_mape

        drift = (rolling_mape - baseline_mape) > DRIFT_THRESHOLD_PCT

        results.append({
            "hospital":      h,
            "blood_group":   bg,
            "baseline_mape": round(baseline_mape, 3),
            "rolling_mape":  round(rolling_mape, 3),
            "drift_detected":drift,
        })

        # Persist to history so the dashboard can plot trends over time
        _append_history({
            "timestamp":      datetime.now().isoformat(),
            "hospital":       h, "blood_group": bg,
            "baseline_mape":  round(baseline_mape, 3),
            "rolling_mape":   round(rolling_mape, 3),
            "drift_detected": drift, "retrained": False,
        })

    return pd.DataFrame(results)


def retrain_combo(df, hospital, blood_group, verbose=True):
    """
    Retrain both Prophet and LSTM models for a single hospital-blood_group combo.
    Called automatically when drift is detected, or manually with --retrain-all.
    """
    from prophet_model import ProphetForecaster
    from lstm_model    import LSTMForecaster
    try:
        if verbose: print(f"  Retraining {hospital} {blood_group}...", end="", flush=True)
        ProphetForecaster(hospital, blood_group).train(df)
        LSTMForecaster(hospital, blood_group).train(df)
        if verbose: print(" done")
        return True
    except Exception as e:
        if verbose: print(f" FAILED: {e}")
        return False

def retrain_all(df, verbose=True):
    """Force retrain all 6 models regardless of drift status."""
    for h, bg in COMBOS:
        retrain_combo(df, h, bg, verbose)


def analyse_hitl_feedback(verbose=True):
    """
    Read transfer decisions made by humans in the dashboard and adjust W2 weights.

    Logic:
    - If a route was rejected 2+ times with "logistical" reason out of at least 3 decisions
      → penalise W2 for that route (makes the route score lower in future recommendations)
    - If a route was approved ≥80% of the time (at least 3 decisions)
      → boost W2 for that route (makes it rank higher in future)

    The adjusted weights are saved to route_weights.json and read by the redistribution engine.
    """
    log_path = os.path.join(DATA_DIR, "processed", "transfer_log.csv")
    if not os.path.exists(log_path):
        if verbose: print("  transfer_log.csv not found — skipping HITL analysis.")
        return {}

    log = pd.read_csv(log_path)
    if log.empty: return {}

    weights = _load_route_weights()

    for (donor, recip), grp in log.groupby(["donor", "recipient"]):
        key      = f"{donor}->{recip}"
        total    = len(grp)
        approved = (grp["decision"] == "APPROVED").sum()
        rejected = (grp["decision"] == "REJECTED").sum()
        # Count only "logistical" rejections — clinical concerns shouldn't penalise the route
        log_rej  = ((grp["decision"] == "REJECTED") &
                    grp["rejection_reason"].str.lower().str.contains("logistical", na=False)).sum()

        cw = weights.get(key, 1.0)   # current W2 multiplier (default 1.0 = no adjustment)

        if total >= 3 and rejected >= 2 and log_rej / max(rejected, 1) >= 0.5:
            nw     = max(W2_MIN, cw - REJECTION_PENALTY)
            action = f"PENALISED (logistic rejections: {log_rej}/{total})"
        elif total >= 3 and approved / total >= 0.80:
            nw     = min(W2_MAX, cw + APPROVAL_BOOST)
            action = f"BOOSTED ({approved/total:.0%} approval rate)"
        else:
            nw     = cw
            action = "unchanged"

        weights[key] = round(nw, 4)
        if verbose and action != "unchanged":
            print(f"  {key}: W2 {cw:.3f} → {nw:.3f}  [{action}]")

    _save_route_weights(weights)
    return weights


def write_health_report(drift_df, weights, retrained_combos):
    """
    Write a human-readable health report summarising:
    - Overall model health status (HEALTHY / WARNING / CRITICAL)
    - Per-combo MAPE tracking table
    - Which combos were retrained
    - Current W2 route weight adjustments
    """
    lines = ["=" * 60, "  MLOps Health Report",
             f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", "=" * 60, ""]

    if not drift_df.empty:
        nd     = drift_df["drift_detected"].sum()
        health = "HEALTHY" if nd == 0 else ("WARNING" if nd <= 2 else "CRITICAL")
        lines += [
            f"OVERALL MODEL HEALTH: {health}  ({nd}/{len(drift_df)} combos drifted)", "",
            "PER-COMBO MAPE TRACKING", "-" * 55,
            f"  {'Combo':<28} {'Baseline':>9} {'Rolling':>9} {'Drift?':>8}",
            "  " + "-" * 52,
        ]
        for _, r in drift_df.iterrows():
            flag = "YES  <--" if r["drift_detected"] else "no"
            lines.append(f"  {r['hospital']} {r['blood_group']:<22} {r['baseline_mape']:>8.2f}% {r['rolling_mape']:>8.2f}%  {flag:>8}")
        lines.append("")

    if retrained_combos:
        lines += ["AUTO-RETRAINED COMBOS", "-" * 30] + [f"  {h} {bg}" for h, bg in retrained_combos] + [""]
    else:
        lines += ["AUTO-RETRAINING: None triggered (no drift detected)", ""]

    if weights:
        lines += ["HITL ROUTE WEIGHT ADJUSTMENTS (W2)", "-" * 40] + \
                 [f"  {r:<35}  W2 = {w:.4f}" for r, w in weights.items()] + [""]
    else:
        lines += ["HITL FEEDBACK: No route adjustments (insufficient log data)", ""]

    lines.append("=" * 60)
    txt = "\n".join(lines)
    with open(HEALTH_TXT, "w", encoding="utf-8") as f: f.write(txt)
    return txt


def run_monitor(simulate=False, retrain_all_flag=False, verbose=True):
    """
    Main entry point — runs the full monitoring pipeline:
    1. Load all forecasters
    2. Compute rolling MAPE and detect drift
    3. Retrain any drifted models (unless simulate=True)
    4. Analyse HITL feedback and adjust route weights
    5. Write the health report

    simulate=True  → detect drift and report, but don't actually retrain
    retrain_all_flag=True → retrain everything regardless of drift
    """
    from hybrid_ensemble import HybridForecaster
    print("=== MLOps Monitor ===\n")

    df = pd.read_csv(os.path.join(DATA_DIR, "raw", "synthetic_demand.csv"), parse_dates=["date"])

    print("Loading forecasters...")
    fcs = {}
    for h, bg in COMBOS:
        hf = HybridForecaster(h, bg); hf.load_models(); hf.optimise_weights(df)
        fcs[(h, bg)] = hf

    # Handle --retrain-all flag: skip drift detection, just retrain everything
    if retrain_all_flag:
        retrain_all(df, verbose); print("All models retrained."); return

    print("\n[1] Running drift detection (last 4 weeks)...")
    drift_df  = compute_rolling_mape(df, fcs, weeks_back=4)
    retrained = []

    if not drift_df.empty:
        for _, r in drift_df[drift_df["drift_detected"]].iterrows():
            tag = f"{r['hospital']} {r['blood_group']}"
            print(f"  Drift detected: {tag}  (rolling MAPE +{r['rolling_mape']-r['baseline_mape']:.1f}pp above baseline)")
            if simulate:
                # In simulate mode, mark as would-be-retrained but don't actually do it
                retrained.append((r["hospital"], r["blood_group"]))
            elif retrain_combo(df, r["hospital"], r["blood_group"], verbose):
                retrained.append((r["hospital"], r["blood_group"]))

    print("\n[2] Analysing HITL feedback from transfer_log.csv...")
    weights = analyse_hitl_feedback(verbose)

    print("\n[3] Writing health report...")
    report = write_health_report(drift_df, weights, retrained)
    print(f"\n{report}\nSaved to: {HEALTH_TXT}")


# ----------------------------------------------------------------------
# Command-line interface
# ----------------------------------------------------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="MLOps Drift Monitor")
    p.add_argument("--simulate-check", action="store_true",
                   help="Detect drift and report, but do NOT actually retrain models")
    p.add_argument("--retrain-all",    action="store_true",
                   help="Force retrain all 6 models regardless of drift status")
    args = p.parse_args()
    run_monitor(simulate=args.simulate_check, retrain_all_flag=args.retrain_all)
