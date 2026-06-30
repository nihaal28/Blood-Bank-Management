"""
Predictive Blood Inventory Management System - Streamlit Dashboard
Human-in-the-Loop interface for forecast review, inventory monitoring,
and transfer recommendation approval / rejection.

IMPORTANT: All transfer decisions require human approval.
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

# ── path setup ────────────────────────────────────────────────────────────────
DASH_DIR  = os.path.dirname(os.path.abspath(__file__))
BASE_DIR  = os.path.dirname(DASH_DIR)
SRC_DIR   = os.path.join(BASE_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from hybrid_ensemble  import HybridForecaster
from redistribution   import RedistributionEngine
import mlops_monitor

DATA_DIR      = os.path.join(BASE_DIR, "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
REPORTS_DIR   = os.path.join(BASE_DIR, "outputs", "reports")
os.makedirs(PROCESSED_DIR, exist_ok=True)

LOG_PATH = os.path.join(PROCESSED_DIR, "transfer_log.csv")
LOG_COLS = ["timestamp", "date", "donor", "recipient", "blood_group",
            "qty", "priority_score", "decision", "rejection_reason", "manager_id"]

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Blood Inventory System",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── constants ──────────────────────────────────────────────────────────────────
HOSPITALS    = ["Hospital_A", "Hospital_B", "Hospital_C"]
BLOOD_GROUPS = ["O_positive", "O_negative"]
COMBOS       = [(h, bg) for h in HOSPITALS for bg in BLOOD_GROUPS]

# ── helpers ───────────────────────────────────────────────────────────────────
def _dos_color(dos: float) -> str:
    if dos < 2:   return "#e74c3c"   # red — critical
    if dos < 4:   return "#f39c12"   # amber — low
    return "#27ae60"                  # green — healthy

def _ensure_log():
    if not os.path.exists(LOG_PATH):
        pd.DataFrame(columns=LOG_COLS).to_csv(LOG_PATH, index=False)

def _append_log(row: dict):
    _ensure_log()
    log = pd.read_csv(LOG_PATH)
    log = pd.concat([log, pd.DataFrame([row])], ignore_index=True)
    log.to_csv(LOG_PATH, index=False)

# ── cached loaders ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading dataset...")
def load_data():
    df = pd.read_csv(
        os.path.join(DATA_DIR, "raw", "synthetic_demand.csv"),
        parse_dates=["date"]
    )
    graph = pd.read_csv(os.path.join(DATA_DIR, "raw", "hospital_graph.csv"))
    return df, graph

@st.cache_resource(show_spinner="Loading forecasting models...")
def load_forecasters():
    forecasters = {}
    for h, bg in COMBOS:
        hf = HybridForecaster(h, bg)
        hf.load_models()
        hf.optimise_weights(load_data()[0])
        forecasters[(h, bg)] = hf
    return forecasters

@st.cache_resource(show_spinner="Initialising redistribution engine...")
def load_engine():
    return RedistributionEngine()

# ── initialise resources ──────────────────────────────────────────────────────
_ensure_log()
df, graph_df = load_data()
forecasters  = load_forecasters()
engine       = load_engine()

MIN_DATE = df["date"].min().date()
MAX_DATE = df["date"].max().date()

# ── sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/blood-bag.png", width=60)
    st.title("Blood Inventory\nManagement System")
    st.divider()

    selected_date = st.date_input(
        "Reference Date",
        value=MAX_DATE,
        min_value=MIN_DATE,
        max_value=MAX_DATE,
    )
    selected_date = pd.Timestamp(selected_date)

    hosp_filter = st.selectbox("Hospital Filter",
                                ["All"] + HOSPITALS)
    bg_filter   = st.selectbox("Blood Group Filter",
                                ["All"] + BLOOD_GROUPS)
    st.divider()
    st.caption(f"Data: {MIN_DATE} to {MAX_DATE}")
    st.caption("Phase 1 — Synthetic data")

# ── header banner ──────────────────────────────────────────────────────────────
st.markdown(
    """
    <div style='background:#c0392b;padding:10px 20px;border-radius:6px;margin-bottom:12px;'>
      <span style='color:white;font-size:16px;font-weight:bold;'>
        &#9888;  All transfer decisions require human approval
      </span>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── tabs ───────────────────────────────────────────────────────────────────────
tabs = st.tabs([
    "Overview",
    "Inventory",
    "Forecast",
    "Transfers",
    "History",
    "Metrics",
    "MLOps & Drift",
])

# =============================================================================
# TAB 1 — OVERVIEW
# =============================================================================
with tabs[0]:
    st.subheader("Network Overview")

    # Compute current state
    snap = df[df["date"] <= selected_date].groupby(
        ["hospital", "blood_group"]
    ).last().reset_index()

    # Metric cards — total network stock
    opos_stock = snap[snap["blood_group"] == "O_positive"]["stock_on_hand"].sum()
    oneg_stock = snap[snap["blood_group"] == "O_negative"]["stock_on_hand"].sum()
    total_exp  = snap["units_expiring_soon"].sum()

    c1, c2, c3 = st.columns(3)
    c1.metric("Total O+ Stock (network)", f"{int(opos_stock):,} units")
    c2.metric("Total O- Stock (network)", f"{int(oneg_stock):,} units")
    c3.metric("Units Expiring Soon (<=3 days)", f"{int(total_exp):,} units",
              delta=None if total_exp == 0 else "Action needed",
              delta_color="inverse")

    st.divider()

    # Hospital node map using plotly scatter
    pos = {"Hospital_A": (0.5, 0.85),
           "Hospital_B": (0.15, 0.2),
           "Hospital_C": (0.85, 0.2)}

    fig = go.Figure()

    # Draw edges
    for _, row in graph_df.iterrows():
        a, b = row["from_hospital"], row["to_hospital"]
        fig.add_trace(go.Scatter(
            x=[pos[a][0], pos[b][0], None],
            y=[pos[a][1], pos[b][1], None],
            mode="lines",
            line=dict(color="lightgray", width=2),
            hoverinfo="none",
            showlegend=False,
        ))
        mx = (pos[a][0] + pos[b][0]) / 2
        my = (pos[a][1] + pos[b][1]) / 2
        fig.add_annotation(
            x=mx, y=my,
            text=f"{int(row['distance_km'])} km",
            showarrow=False,
            font=dict(size=10, color="gray"),
        )

    # Draw hospital nodes
    summary_rows = []
    for hosp in HOSPITALS:
        hosp_snap = snap[snap["hospital"] == hosp]
        total_stock = int(hosp_snap["stock_on_hand"].sum())
        mean_demand = df[df["hospital"] == hosp]["demand"].mean()
        dos = (total_stock / mean_demand) if mean_demand > 0 else 0
        color = _dos_color(dos)

        opos = hosp_snap[hosp_snap["blood_group"] == "O_positive"]["stock_on_hand"].values
        oneg = hosp_snap[hosp_snap["blood_group"] == "O_negative"]["stock_on_hand"].values

        fig.add_trace(go.Scatter(
            x=[pos[hosp][0]], y=[pos[hosp][1]],
            mode="markers+text",
            marker=dict(size=40, color=color, line=dict(width=2, color="black")),
            text=[hosp.replace("Hospital_", "H.")],
            textposition="top center",
            textfont=dict(size=11, color="black"),
            hovertemplate=(
                f"<b>{hosp}</b><br>"
                f"Total stock: {total_stock} units<br>"
                f"Days of supply: {dos:.1f}<br>"
                f"O+ stock: {int(opos[0]) if len(opos) else 0}<br>"
                f"O- stock: {int(oneg[0]) if len(oneg) else 0}"
                "<extra></extra>"
            ),
            showlegend=False,
        ))

        summary_rows.append({
            "Hospital": hosp,
            "O+ Stock": int(opos[0]) if len(opos) else 0,
            "O- Stock": int(oneg[0]) if len(oneg) else 0,
            "Total Stock": total_stock,
            "Days of Supply": round(dos, 1),
            "Status": "CRITICAL" if dos < 2 else ("LOW" if dos < 4 else "HEALTHY"),
        })

    fig.update_layout(
        height=350,
        xaxis=dict(visible=False, range=[-0.1, 1.1]),
        yaxis=dict(visible=False, range=[0, 1.1]),
        margin=dict(l=0, r=0, t=10, b=0),
        plot_bgcolor="white",
    )
    # Legend patches
    for label, color in [("HEALTHY", "#27ae60"), ("LOW", "#f39c12"), ("CRITICAL", "#e74c3c")]:
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(size=12, color=color),
            name=label, showlegend=True,
        ))
    st.plotly_chart(fig, width='stretch')

    st.subheader("Hospital Summary")
    summ_df = pd.DataFrame(summary_rows)
    st.dataframe(
        summ_df.style.apply(
            lambda col: [
                f"background-color: {'#fadbd8' if v == 'CRITICAL' else '#fef9e7' if v == 'LOW' else '#d5f5e3'}"
                for v in col
            ] if col.name == "Status" else [""] * len(col),
            axis=0,
        ),
        width='stretch',
        hide_index=True,
    )

# =============================================================================
# TAB 2 — INVENTORY
# =============================================================================
with tabs[1]:
    st.subheader("Inventory Status")

    snap2 = df[df["date"] <= selected_date].groupby(
        ["hospital", "blood_group"]
    ).last().reset_index()

    inv_rows = []
    for _, row in snap2.iterrows():
        hosp, bg = row["hospital"], row["blood_group"]
        sub = df[(df["hospital"] == hosp) & (df["blood_group"] == bg)]
        mean_daily = sub["demand"].mean()
        dos = row["stock_on_hand"] / mean_daily if mean_daily > 0 else 0

        # Apply filters
        if hosp_filter != "All" and hosp != hosp_filter:
            continue
        if bg_filter != "All" and bg != bg_filter:
            continue

        inv_rows.append({
            "Hospital":          hosp,
            "Blood Group":       bg,
            "Stock on Hand":     int(row["stock_on_hand"]),
            "Expiring Soon":     int(row["units_expiring_soon"]),
            "Days of Supply":    round(dos, 1),
            "Mean Daily Demand": round(mean_daily, 1),
            "Status":            "CRITICAL" if dos < 2 else ("LOW" if dos < 4 else "HEALTHY"),
        })

    inv_df = pd.DataFrame(inv_rows)
    if not inv_df.empty:
        def _color_status(val):
            if val == "CRITICAL": return "background-color:#fadbd8; color:#c0392b; font-weight:bold"
            if val == "LOW":      return "background-color:#fef9e7; color:#d35400; font-weight:bold"
            return "background-color:#d5f5e3; color:#1e8449"

        st.dataframe(
            inv_df.style.applymap(_color_status, subset=["Status"]),
            width='stretch', hide_index=True,
        )

        # Grouped bar chart
        fig_inv = px.bar(
            inv_df, x="Hospital", y="Stock on Hand",
            color="Blood Group", barmode="group",
            color_discrete_map={"O_positive": "#e74c3c", "O_negative": "#3498db"},
            title=f"Stock Levels by Hospital and Blood Group  (as of {selected_date.date()})",
        )
        fig_inv.update_layout(height=350)
        st.plotly_chart(fig_inv, width='stretch')
    else:
        st.info("No data for selected filters.")

# =============================================================================
# TAB 3 — FORECAST
# =============================================================================
with tabs[2]:
    st.subheader("7-Day Demand Forecast")

    fc_hosp = st.selectbox("Hospital", HOSPITALS, key="fc_hosp")
    fc_bg   = st.selectbox("Blood Group", BLOOD_GROUPS, key="fc_bg")

    sub_fc = df[(df["hospital"] == fc_hosp) &
                (df["blood_group"] == fc_bg)].sort_values("date")

    # Last 14 days actual
    last14 = sub_fc[sub_fc["date"] <= selected_date].tail(14)

    # Get forecasts
    hf = forecasters[(fc_hosp, fc_bg)]
    hf._prophet._last_date = selected_date
    prophet_fc = hf._prophet.predict(periods=7)
    lstm_fc    = hf._lstm.predict(sub_fc[sub_fc["date"] <= selected_date].tail(30))
    hybrid_fc  = hf.predict(df, target_date=selected_date)

    fig_fc = go.Figure()
    fig_fc.add_trace(go.Scatter(
        x=last14["date"], y=last14["demand"],
        mode="lines+markers", name="Actual (14 days)",
        line=dict(color="#2980b9", width=2),
    ))
    fig_fc.add_trace(go.Scatter(
        x=prophet_fc["ds"], y=prophet_fc["yhat"],
        mode="lines+markers", name="Prophet (Fourier)",
        line=dict(color="#8e44ad", width=1.5, dash="dash"),
    ))
    fig_fc.add_trace(go.Scatter(
        x=hybrid_fc["date"], y=hybrid_fc["lstm_pred"],
        mode="lines+markers", name="LSTM",
        line=dict(color="#e67e22", width=1.5, dash="dash"),
    ))
    fig_fc.add_trace(go.Scatter(
        x=hybrid_fc["date"], y=hybrid_fc["hybrid_pred"],
        mode="lines+markers", name=f"Hybrid (a={hf.alpha:.2f})",
        line=dict(color="#27ae60", width=3),
    ))
    fig_fc.add_vline(
        x=selected_date.timestamp() * 1000,
        line_width=1.5, line_dash="dot", line_color="gray",
        annotation_text="Today", annotation_position="top left",
    )
    fig_fc.add_vrect(
        x0=hybrid_fc["date"].min(), x1=hybrid_fc["date"].max(),
        fillcolor="#27ae60", opacity=0.05,
        annotation_text="7-day window", annotation_position="top left",
    )
    fig_fc.update_layout(
        title=f"Demand Forecast — {fc_hosp}  |  {fc_bg.replace('_',' ')}",
        xaxis_title="Date", yaxis_title="Demand (units)",
        height=420, hovermode="x unified",
    )
    st.plotly_chart(fig_fc, width='stretch')

    # Day-by-day table
    st.subheader("Day-by-Day Forecast Values")
    fc_table = hybrid_fc.copy()
    fc_table["Prophet"] = prophet_fc["yhat"].values
    fc_table["LSTM"]    = lstm_fc
    fc_table["Day"]     = [d.strftime("%a %d %b") for d in fc_table["date"]]
    fc_table = fc_table[["Day", "Prophet", "LSTM", "hybrid_pred"]].rename(
        columns={"hybrid_pred": "Hybrid"}
    )
    fc_table[["Prophet", "LSTM", "Hybrid"]] = fc_table[["Prophet", "LSTM", "Hybrid"]].round(1)
    st.dataframe(fc_table, width='stretch', hide_index=True)

# =============================================================================
# TAB 4 — TRANSFERS
# =============================================================================
with tabs[3]:
    st.subheader("Transfer Recommendations")
    st.info("Human approval required for all transfers. Approved/rejected decisions are logged for model learning.")

    # Build forecast_dict for selected date
    fc_dict: dict = {}
    for (h, bg), hf_t in forecasters.items():
        try:
            fc = hf_t.predict(df, target_date=selected_date)
            fc_dict.setdefault(h, {})[bg] = float(fc["hybrid_pred"].sum())
        except Exception:
            fc_dict.setdefault(h, {})[bg] = 0.0

    recs = engine.generate_recommendations(df, fc_dict, selected_date)

    if recs.empty:
        st.success(f"No transfer recommendations for {selected_date.date()} — all hospitals are balanced.")
        st.caption("Recommendations are generated when any hospital has < 9 days of supply "
                   "while another has > 8 days of supply. Try dates in early Jan 2021.")
    else:
        st.markdown(f"**{len(recs)} recommendation(s) found for {selected_date.date()}**")

        # Session state for button tracking
        if "approved" not in st.session_state:
            st.session_state.approved = set()
        if "rejected" not in st.session_state:
            st.session_state.rejected = set()

        for i, row in recs.iterrows():
            uid = f"{row['donor_hospital']}_{row['recipient_hospital']}_{row['blood_group']}_{i}"

            with st.container():
                col_info, col_approve, col_reject = st.columns([5, 1, 1])
                with col_info:
                    st.markdown(
                        f"**#{i+1}** &nbsp; "
                        f"`{row['donor_hospital']}` &rarr; `{row['recipient_hospital']}` &nbsp;|&nbsp; "
                        f"**{row['blood_group'].replace('_',' ')}** &nbsp;|&nbsp; "
                        f"**{int(row['transfer_qty'])} units** &nbsp;|&nbsp; "
                        f"Priority: **{row['priority_score']:.3f}** &nbsp;|&nbsp; "
                        f"{row['distance_km']} km / {row['transit_hr']} hr &nbsp;|&nbsp; "
                        f"Donor DOS: {row.get('donor_dos','?')} d &nbsp;|&nbsp; "
                        f"Recipient DOS: {row.get('recipient_dos','?')} d"
                    )

                with col_approve:
                    if uid not in st.session_state.approved and uid not in st.session_state.rejected:
                        if st.button("APPROVE", key=f"app_{uid}", type="primary"):
                            st.session_state.approved.add(uid)
                            _append_log({
                                "timestamp":        datetime.now().isoformat(),
                                "date":             str(selected_date.date()),
                                "donor":            row["donor_hospital"],
                                "recipient":        row["recipient_hospital"],
                                "blood_group":      row["blood_group"],
                                "qty":              int(row["transfer_qty"]),
                                "priority_score":   row["priority_score"],
                                "decision":         "APPROVED",
                                "rejection_reason": "",
                                "manager_id":       "manager_01",
                            })
                            st.toast("Transfer approved and logged!", icon="✅")
                            st.rerun()

                with col_reject:
                    if uid not in st.session_state.approved and uid not in st.session_state.rejected:
                        if st.button("REJECT", key=f"rej_{uid}"):
                            st.session_state.rejected.add(uid)

                # Show rejection reason picker
                if uid in st.session_state.rejected and uid not in st.session_state.approved:
                    reason = st.selectbox(
                        "Rejection reason",
                        ["Clinical concern", "Logistical issue",
                         "Insufficient quantity", "Better alternative available", "Other"],
                        key=f"reason_{uid}",
                    )
                    if st.button("Confirm Rejection", key=f"conf_{uid}"):
                        _append_log({
                            "timestamp":        datetime.now().isoformat(),
                            "date":             str(selected_date.date()),
                            "donor":            row["donor_hospital"],
                            "recipient":        row["recipient_hospital"],
                            "blood_group":      row["blood_group"],
                            "qty":              int(row["transfer_qty"]),
                            "priority_score":   row["priority_score"],
                            "decision":         "REJECTED",
                            "rejection_reason": reason,
                            "manager_id":       "manager_01",
                        })
                        st.toast("Rejection logged for model learning.", icon="ℹ️")
                        st.rerun()

                # Status badges
                if uid in st.session_state.approved:
                    st.success("APPROVED — logged to transfer_log.csv")
                elif uid in st.session_state.rejected and uid not in st.session_state.approved:
                    pass  # handled above

                st.divider()

# =============================================================================
# TAB 5 — HISTORY
# =============================================================================
with tabs[4]:
    st.subheader("Transfer Decision History")
    _ensure_log()
    log_df = pd.read_csv(LOG_PATH)

    if log_df.empty:
        st.info("No transfer decisions logged yet. Use the Transfers tab to approve or reject recommendations.")
    else:
        total_app = (log_df["decision"] == "APPROVED").sum()
        total_rej = (log_df["decision"] == "REJECTED").sum()
        approval_rate = 100 * total_app / len(log_df) if len(log_df) > 0 else 0

        h1, h2, h3 = st.columns(3)
        h1.metric("Total Approved",  int(total_app))
        h2.metric("Total Rejected",  int(total_rej))
        h3.metric("Approval Rate",   f"{approval_rate:.1f}%")

        st.dataframe(log_df, width='stretch', hide_index=True)

        # Weekly bar chart
        if len(log_df) > 1:
            log_df["date"] = pd.to_datetime(log_df["date"])
            log_df["week"] = log_df["date"].dt.to_period("W").astype(str)
            weekly = log_df.groupby(["week", "decision"]).size().reset_index(name="count")
            fig_hist = px.bar(
                weekly, x="week", y="count", color="decision", barmode="group",
                color_discrete_map={"APPROVED": "#27ae60", "REJECTED": "#e74c3c"},
                title="Approvals vs Rejections by Week",
            )
            fig_hist.update_layout(height=320)
            st.plotly_chart(fig_hist, width='stretch')

# =============================================================================
# TAB 6 — METRICS
# =============================================================================
with tabs[5]:
    st.subheader("Model Accuracy Metrics")

    def _load_metrics(name):
        p = os.path.join(REPORTS_DIR, name)
        return pd.read_csv(p) if os.path.exists(p) else pd.DataFrame()

    prophet_m  = _load_metrics("prophet_metrics.csv")
    lstm_m     = _load_metrics("lstm_metrics.csv")
    compare_m  = _load_metrics("model_comparison.csv")

    if not compare_m.empty:
        hybrid_rows = compare_m[compare_m["model"] == "Hybrid"]
        best_mae    = hybrid_rows["MAE"].min()
        best_rmse   = hybrid_rows["RMSE"].min()
        best_mape   = hybrid_rows["MAPE"].min()

        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("Best Hybrid MAE",  f"{best_mae:.2f} units")
        mc2.metric("Best Hybrid RMSE", f"{best_rmse:.2f} units")
        mc3.metric("Best Hybrid MAPE", f"{best_mape:.2f}%")

        st.divider()

        # MAE comparison grouped bar chart
        fig_m = px.bar(
            compare_m,
            x=compare_m["hospital"] + " " + compare_m["blood_group"],
            y="MAE", color="model", barmode="group",
            color_discrete_map={
                "Prophet": "#8e44ad", "LSTM": "#e67e22",
                "Hybrid": "#27ae60", "MovingAverage": "#95a5a6",
            },
            title="MAE Comparison — All Models and Hospital/Blood-Group Combinations",
        )
        fig_m.update_layout(height=380, xaxis_title="Hospital + Blood Group")
        st.plotly_chart(fig_m, width='stretch')

        st.subheader("Full Comparison Table")
        st.dataframe(compare_m, width='stretch', hide_index=True)
    else:
        st.warning("Run src/hybrid_ensemble.py first to generate model_comparison.csv")

# =============================================================================
# TAB 7 — MLOps & Drift
# =============================================================================
with tabs[6]:
    st.subheader("MLOps — Model Drift Detection & Continuous Learning")

    HISTORY_CSV = os.path.join(REPORTS_DIR, "mlops_history.csv")
    HEALTH_TXT  = os.path.join(REPORTS_DIR, "mlops_health_report.txt")
    W2_STORE    = os.path.join(REPORTS_DIR, "route_weights.json")

    # ── Trigger retraining button ─────────────────────────────────────────────
    col_btn, col_last = st.columns([2, 3])
    with col_btn:
        if st.button("Trigger Retraining Now", type="primary"):
            with st.spinner("Running drift check and retraining if needed..."):
                try:
                    mlops_monitor.run_monitor(simulate=False, verbose=False)
                    st.toast("MLOps check complete. See report below.", icon="✅")
                    st.rerun()
                except Exception as e:
                    st.error(f"Monitor error: {e}")

    with col_last:
        if os.path.exists(HISTORY_CSV):
            hist_preview = pd.read_csv(HISTORY_CSV)
            if not hist_preview.empty:
                last_run = hist_preview["timestamp"].max()
                st.caption(f"Last monitoring run: **{last_run}**")
        else:
            st.caption("No monitoring runs yet. Click 'Trigger Retraining Now' or run `python src/mlops_monitor.py --simulate-check`")

    st.divider()

    # ── MAPE history chart ────────────────────────────────────────────────────
    if os.path.exists(HISTORY_CSV):
        hist_df = pd.read_csv(HISTORY_CSV, parse_dates=["timestamp"])
        if not hist_df.empty:
            st.subheader("Rolling MAPE Evolution")

            # Health indicator per combo
            latest = hist_df.sort_values("timestamp").groupby(
                ["hospital", "blood_group"]
            ).last().reset_index()

            health_cols = st.columns(len(latest))
            for col, (_, row) in zip(health_cols, latest.iterrows()):
                tag   = f"{row['hospital'].replace('Hospital_','H.')} {row['blood_group'].replace('_',' ')}"
                mape  = row["rolling_mape"]
                base  = row["baseline_mape"]
                delta = mape - base
                color = "normal" if delta <= 2 else ("off" if delta <= 5 else "inverse")
                col.metric(tag, f"{mape:.1f}%", f"{delta:+.1f}pp vs baseline",
                           delta_color=color)

            # Line chart
            hist_df["combo"] = hist_df["hospital"] + " " + hist_df["blood_group"]
            fig_drift = px.line(
                hist_df, x="timestamp", y="rolling_mape",
                color="combo", markers=True,
                title="Rolling MAPE Over Time (weekly checks)",
                labels={"rolling_mape": "Rolling MAPE (%)", "timestamp": "Check Time"},
            )
            # Add baseline reference lines
            for _, row in latest.iterrows():
                fig_drift.add_hline(
                    y=row["baseline_mape"], line_dash="dot",
                    line_color="gray", opacity=0.4,
                )
            fig_drift.update_layout(height=380, hovermode="x unified")
            st.plotly_chart(fig_drift, width='stretch')

            # Drift events table
            drift_events = hist_df[hist_df["drift_detected"] == True]
            if not drift_events.empty:
                st.warning(f"**{len(drift_events)} drift event(s) detected** in history")
                st.dataframe(
                    drift_events[["timestamp","hospital","blood_group",
                                  "baseline_mape","rolling_mape"]].rename(
                        columns={"baseline_mape":"Baseline MAPE","rolling_mape":"Rolling MAPE"}
                    ),
                    width='stretch', hide_index=True,
                )
            else:
                st.success("No drift events detected. All models healthy.")
        else:
            st.info("No monitoring history yet.")
    else:
        st.info("No monitoring history yet. Run the monitor to populate this view.")

    st.divider()

    # ── HITL route weight adjustments ────────────────────────────────────────
    st.subheader("HITL Feedback — Transport Score (W2) Adjustments")
    import json as _json
    if os.path.exists(W2_STORE):
        with open(W2_STORE) as f:
            route_weights = _json.load(f)
        if route_weights:
            wdf = pd.DataFrame([
                {"Route": k, "W2 Multiplier": v,
                 "Status": "Penalised" if v < 1.0 else ("Boosted" if v > 1.0 else "Default")}
                for k, v in route_weights.items()
            ])
            def _w2_color(val):
                if val == "Penalised": return "background-color:#fadbd8; color:#c0392b"
                if val == "Boosted":   return "background-color:#d5f5e3; color:#1e8449"
                return ""
            st.dataframe(
                wdf.style.applymap(_w2_color, subset=["Status"]),
                width='stretch', hide_index=True,
            )
        else:
            st.info("No route adjustments yet (insufficient transfer log data).")
    else:
        st.info("No route weights file found. Run the MLOps monitor to generate it.")

    st.divider()

    # ── Health report text ────────────────────────────────────────────────────
    st.subheader("Latest Health Report")
    if os.path.exists(HEALTH_TXT):
        with open(HEALTH_TXT, encoding="utf-8") as f:
            st.code(f.read(), language=None)
    else:
        st.info("No health report generated yet. Click 'Trigger Retraining Now'.")
