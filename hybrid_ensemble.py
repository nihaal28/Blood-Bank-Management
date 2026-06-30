"""
hybrid_ensemble.py
------------------
Hybrid Ensemble Forecaster — combines Prophet and LSTM predictions.

WHY COMBINE TWO MODELS?
  No single model is best in every situation:
  - ProphetForecaster is strong at capturing seasonal patterns (holidays, weekends)
  - LSTMForecaster is strong at learning recent trends from raw data sequences
  Blending them: Final = alpha × Prophet + (1 - alpha) × LSTM
  usually outperforms either model alone.

HOW ARE THE WEIGHTS CHOSEN?
  Instead of using a fixed 60/40 split, we find the best alpha for each
  hospital-blood_group combination by testing many values on a validation
  window and picking the one that minimises the Mean Absolute Error.
  scipy.optimize.minimize_scalar does this search efficiently.
"""

import os, sys, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"; os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar  # efficient 1-D optimiser

# Make sure Python can find our other modules in the src/ folder
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path: sys.path.insert(0, SRC_DIR)

from prophet_model import ProphetForecaster
from lstm_model    import LSTMForecaster, FEATURE_COLS

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLOTS_DIR   = os.path.join(BASE_DIR, "outputs", "plots")
REPORTS_DIR = os.path.join(BASE_DIR, "outputs", "reports")
os.makedirs(PLOTS_DIR, exist_ok=True); os.makedirs(REPORTS_DIR, exist_ok=True)

# ── Metric helpers ──────────────────────────────────────────────────────────
def _mape(a, p):
    """Mean Absolute Percentage Error (lower = better)."""
    mask = a > 1
    return float(np.mean(np.abs((a[mask] - p[mask]) / a[mask])) * 100) if mask.sum() else 0.0

def _mae(a, p):
    """Mean Absolute Error — used as the optimisation objective."""
    return float(np.mean(np.abs(a - p)))

def _rmse(a, p):
    """Root Mean Squared Error — penalises large errors more than MAE."""
    return float(np.sqrt(np.mean((a - p) ** 2)))


def _rolling_preds(prophet, lstm, sub, start_idx, n_steps, horizon=7):
    """
    Simulate a realistic forecasting scenario by rolling forward in time.

    For each step we pretend we only have data up to that point, make a
    7-day prediction, then move forward 7 days and repeat.  This gives us
    predictions on unseen data — the honest way to measure accuracy.

    Returns three aligned arrays: actual demand, prophet predictions, lstm predictions.
    """
    aa, ap, al = [], [], []  # actual, prophet preds, lstm preds

    for step in range(0, n_steps * horizon, horizon):
        cut = start_idx + step
        if cut + horizon >= len(sub): break

        # Everything up to 'cut' is the history the model can see
        window_df = sub.iloc[:cut + 1]
        actual    = sub.iloc[cut + 1 : cut + 1 + horizon]["demand"].values
        if len(actual) == 0: break

        # Get Prophet prediction (tell it the last known date)
        prophet._last_date = window_df["date"].iloc[-1]
        pp = prophet.predict(periods=horizon)["yhat"].values[:len(actual)]

        # Get LSTM prediction (needs the last 30 rows as input)
        lp = lstm.predict(window_df.tail(30))[:len(actual)]

        aa.extend(actual); ap.extend(pp); al.extend(lp)

    return np.array(aa), np.array(ap).clip(0), np.array(al).clip(0)


class HybridForecaster:
    """
    Combines a ProphetForecaster and an LSTMForecaster into a weighted blend.

    Final prediction = alpha × prophet + (1 - alpha) × lstm
    where alpha is optimised per hospital-blood_group combination.
    """

    def __init__(self, hospital, blood_group, alpha=0.6, beta=0.4):
        self.hospital, self.blood_group = hospital, blood_group
        self.alpha = alpha  # weight for Prophet  (default: 60%)
        self.beta  = beta   # weight for LSTM     (default: 40%)
        self._prophet = self._lstm = None

    def load_models(self):
        """Load both underlying models from disk."""
        self._prophet = ProphetForecaster(self.hospital, self.blood_group); self._prophet.load()
        self._lstm    = LSTMForecaster(self.hospital, self.blood_group);    self._lstm.load()

    def optimise_weights(self, df, val_days=90, test_days=90):
        """
        Find the best alpha (Prophet weight) for this combination.

        We use a 90-day validation window that sits just before the 90-day test set.
        minimize_scalar tries different alpha values between 0 and 1 and returns
        the one that gives the lowest MAE on the validation window.
        """
        sub = df[(df["hospital"] == self.hospital) & (df["blood_group"] == self.blood_group)].sort_values("date").reset_index(drop=True)

        # Run rolling predictions on the validation window
        val_start = len(sub) - test_days - val_days - 1
        actual, pp, lp = _rolling_preds(self._prophet, self._lstm, sub, val_start, val_days // 7)

        if not len(actual): return self.alpha   # not enough data, keep default

        # The loss function: MAE of the hybrid for a given alpha
        def loss(alpha):
            hybrid = (alpha * pp + (1 - alpha) * lp).clip(0)
            return _mae(actual, hybrid)

        # Bounded 1-D optimisation: find alpha ∈ [0, 1] that minimises MAE
        result = minimize_scalar(loss, bounds=(0.0, 1.0), method="bounded")
        self.alpha = round(float(np.clip(result.x, 0, 1)), 4)
        self.beta  = round(1 - self.alpha, 4)
        return self.alpha

    def predict(self, df, target_date=None):
        """
        Generate a 7-day hybrid forecast.
        If target_date is given, only use data up to that date (for backtesting).
        Returns a DataFrame with columns: date, prophet_pred, lstm_pred, hybrid_pred.
        """
        sub = df[(df["hospital"] == self.hospital) & (df["blood_group"] == self.blood_group)].sort_values("date").reset_index(drop=True)
        if target_date is not None:
            sub = sub[sub["date"] <= pd.Timestamp(target_date)]

        # Get predictions from both models
        self._prophet._last_date = sub["date"].iloc[-1]
        pp = self._prophet.predict(7)["yhat"].values
        lp = self._lstm.predict(sub.tail(30))
        n  = min(len(pp), len(lp), 7)

        # Blend: weighted average of the two forecasts
        hp    = (self.alpha * pp[:n] + self.beta * lp[:n]).clip(0)
        dates = pd.date_range(start=pd.to_datetime(sub["date"].iloc[-1]) + pd.Timedelta(days=1), periods=n)

        return pd.DataFrame({"date": dates, "prophet_pred": pp[:n], "lstm_pred": lp[:n], "hybrid_pred": hp})

    def evaluate_all(self, df, hold_out=90):
        """
        Evaluate all four models (Prophet, LSTM, Hybrid, Moving Average baseline)
        on the last 90 days using rolling 7-day predictions.
        Returns a dict with MAE/RMSE/MAPE for each model, plus the raw prediction arrays.
        """
        sub = df[(df["hospital"] == self.hospital) & (df["blood_group"] == self.blood_group)].sort_values("date").reset_index(drop=True)
        si  = len(sub) - hold_out - 1

        actual, pp, lp = _rolling_preds(self._prophet, self._lstm, sub, si, hold_out // 7)
        hp = (self.alpha * pp + self.beta * lp).clip(0)

        # Moving average baseline: just repeat the last 7 days' average demand
        ma = np.full(len(actual), sub.iloc[:si + 1]["demand"].tail(7).mean())

        # Compute all three error metrics for each model
        def m(p): return {"MAE": round(_mae(actual, p), 3), "RMSE": round(_rmse(actual, p), 3), "MAPE": round(_mape(actual, p), 3)}

        return {
            "Prophet":       m(pp),
            "LSTM":          m(lp),
            "Hybrid":        m(hp),
            "MovingAverage": m(ma),
            "_arrays":       {"actual": actual, "prophet": pp, "lstm": lp, "hybrid": hp},
        }


# ----------------------------------------------------------------------
# Run this file directly to evaluate all models and save comparison charts
# ----------------------------------------------------------------------
if __name__ == "__main__":
    df = pd.read_csv(os.path.join(BASE_DIR, "data", "raw", "synthetic_demand.csv"), parse_dates=["date"])
    combos = [("Hospital_A","O_positive"), ("Hospital_A","O_negative"),
              ("Hospital_B","O_positive"), ("Hospital_B","O_negative"),
              ("Hospital_C","O_positive"), ("Hospital_C","O_negative")]

    rows, evs = [], {}
    for h, bg in combos:
        hf = HybridForecaster(h, bg)
        hf.load_models()
        hf.optimise_weights(df)   # learn the best alpha for this combo
        r = hf.evaluate_all(df)
        evs[(h, bg)] = {**r, "alpha": hf.alpha}
        print(f"  {h} {bg}: alpha={hf.alpha:.3f} | Hybrid MAPE={r['Hybrid']['MAPE']:.2f}%")
        for mn in ["Prophet", "LSTM", "Hybrid", "MovingAverage"]:
            rows.append({"hospital": h, "blood_group": bg, "model": mn, **r[mn]})

    pd.DataFrame(rows).to_csv(os.path.join(REPORTS_DIR, "model_comparison.csv"), index=False)

    # Generate a sample forecast chart for Hospital_A O_positive
    hf_a  = HybridForecaster("Hospital_A", "O_positive"); hf_a.load_models(); hf_a.optimise_weights(df)
    sub_a = df[(df["hospital"]=="Hospital_A") & (df["blood_group"]=="O_positive")].sort_values("date")
    fc_a  = hf_a.predict(df)

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(sub_a.tail(30)["date"], sub_a.tail(30)["demand"], color="steelblue", lw=1.8, label="Actual (last 30d)")
    ax.plot(fc_a["date"], fc_a["prophet_pred"], "--", color="purple", lw=1.5, label=f"Prophet (w={hf_a.alpha:.2f})")
    ax.plot(fc_a["date"], fc_a["lstm_pred"],    "--", color="orange", lw=1.5, label=f"LSTM (w={hf_a.beta:.2f})")
    ax.plot(fc_a["date"], fc_a["hybrid_pred"],       color="green",  lw=2.5, label="Hybrid (blended)")
    ax.axvline(pd.to_datetime(sub_a["date"].max()), color="gray", linestyle=":", lw=1.5, label="Today")
    ax.set(title="Hybrid Forecast — Hospital_A O_positive", xlabel="Date", ylabel="Demand (units)")
    ax.legend(fontsize=9); plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "hybrid_forecast_sample.png"), dpi=100); plt.close()

    mean_mape = np.mean([evs[(h, bg)]["Hybrid"]["MAPE"] for h, bg in combos])
    print(f"\nModel comparison saved. Mean Hybrid MAPE across all combos: {mean_mape:.2f}%")
