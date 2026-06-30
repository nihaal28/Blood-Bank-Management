"""
app4.py — Live-Everything Edition
-----------------------------------
Same as app2.py but with 4 additions:
  1. Sidebar LIVE badge — pulses every second showing current network stock
  2. Tab 1 Overview — hospital network graph auto-refreshes every 2s with live DOS colours
  3. Tab 2 Inventory — stock table & bar chart auto-refresh every 2s from sim_inventory
  4. Tab 4 Transfers — auto-transfer log from the live simulation prepended at top

All other tabs (Forecast, History, Metrics, MLOps, Live Sim, Baseline) unchanged.
Delete this file if you don't like the live mode — app2.py is the stable fallback.
"""
import os, sys, json, warnings, time
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"; os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import pandas as pd, numpy as np, streamlit as st
import plotly.graph_objects as go, plotly.express as px
from datetime import datetime

DASH_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(DASH_DIR)
SRC_DIR  = os.path.join(BASE_DIR, "src")
if SRC_DIR not in sys.path: sys.path.insert(0, SRC_DIR)

from hybrid_ensemble import HybridForecaster
from redistribution  import RedistributionEngine
import mlops_monitor

DATA_DIR      = os.path.join(BASE_DIR, "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
REPORTS_DIR   = os.path.join(BASE_DIR, "outputs", "reports")
os.makedirs(PROCESSED_DIR, exist_ok=True)

LOG_PATH = os.path.join(PROCESSED_DIR, "transfer_log.csv")
LOG_COLS = ["timestamp","date","donor","recipient","blood_group",
            "qty","priority_score","decision","rejection_reason","manager_id"]

st.set_page_config(page_title="Blood Inventory System — Live", layout="wide", initial_sidebar_state="expanded")

# ── White theme: global CSS overrides ─────────────────────────────────────────
st.markdown("""
<style>
/* ── Page & sidebar ── */
.stApp                          { background-color: #ffffff; }
[data-testid="stSidebar"]       { background-color: #f0f2f6; border-right: 1px solid #dde1e7; }
[data-testid="stSidebar"] *     { color: #2c3e50 !important; }

/* ── Main text ── */
body, .stMarkdown, p, li, span  { color: #2c3e50; }
h1, h2, h3, h4                  { color: #1a252f; }

/* ── Metric cards ── */
[data-testid="metric-container"] {
    background-color: #f8f9fa;
    border: 1px solid #dde1e7;
    border-radius: 8px;
    padding: 12px 16px;
}

/* ── Tabs ── */
[data-baseweb="tab-list"]       { background-color: #f0f2f6; border-radius: 8px; }
[data-baseweb="tab"]            { color: #2c3e50 !important; }
[aria-selected="true"]          { background-color: #ffffff !important; border-radius: 6px; }

/* ── Dataframe ── */
.stDataFrame, .dataframe        { background-color: #ffffff; }
thead tr th                     { background-color: #f0f2f6 !important; color: #2c3e50 !important; }

/* ── Buttons ── */
.stButton > button              { border: 1px solid #dde1e7; color: #2c3e50; background: #ffffff; }
.stButton > button:hover        { background-color: #f0f2f6; }

/* ── Divider ── */
hr                              { border-color: #dde1e7; }

/* ── Selectbox / date input ── */
[data-baseweb="select"]         { background-color: #ffffff; }
</style>
""", unsafe_allow_html=True)

# ── White theme helper for all Plotly figures ─────────────────────────────────
def _apply_white_theme(fig):
    """Apply white background and dark text/grid to any Plotly figure."""
    fig.update_layout(
        paper_bgcolor="white",
        plot_bgcolor="#fafafa",
        font=dict(color="#2c3e50", family="sans-serif"),
    )
    fig.update_xaxes(gridcolor="#e8ecf0", zerolinecolor="#e8ecf0", linecolor="#dde1e7")
    fig.update_yaxes(gridcolor="#e8ecf0", zerolinecolor="#e8ecf0", linecolor="#dde1e7")
    return fig

HOSPITALS    = ["Hospital_A","Hospital_B","Hospital_C"]
BLOOD_GROUPS = ["O_positive","O_negative"]
COMBOS       = [(h,bg) for h in HOSPITALS for bg in BLOOD_GROUPS]

# Daily demand reference (matches simulation constants)
_LIVE_DMD = {
    ("Hospital_A","O_positive"):45, ("Hospital_A","O_negative"):18,
    ("Hospital_B","O_positive"):28, ("Hospital_B","O_negative"):11,
    ("Hospital_C","O_positive"):14, ("Hospital_C","O_negative"): 6,
}

def _dos_color(dos):
    return "#e74c3c" if dos < 2 else ("#f39c12" if dos < 4 else "#27ae60")

def _ensure_log():
    if not os.path.exists(LOG_PATH):
        pd.DataFrame(columns=LOG_COLS).to_csv(LOG_PATH, index=False)

def _append_log(row):
    _ensure_log()
    log = pd.read_csv(LOG_PATH)
    pd.concat([log, pd.DataFrame([row])], ignore_index=True).to_csv(LOG_PATH, index=False)

def _log_entry(row, decision, reason=""):
    return {"timestamp":datetime.now().isoformat(),"date":str(selected_date.date()),
            "donor":row["donor_hospital"],"recipient":row["recipient_hospital"],
            "blood_group":row["blood_group"],"qty":int(row["transfer_qty"]),
            "priority_score":row["priority_score"],"decision":decision,
            "rejection_reason":reason,"manager_id":"manager_01"}

# ── Live-mode helpers ─────────────────────────────────────────────────────────
def _sim_active():
    """True when Tab 8 simulation has been initialised and is running."""
    return st.session_state.get("sim_running", False) and \
           "sim_inventory" in st.session_state

def _live_stock(h, bg):
    """Current stock from sim_inventory if sim is active, else None."""
    if _sim_active():
        return st.session_state.sim_inventory.get((h, bg))
    return None

def _live_dos(h, bg):
    """Current DOS from sim if active, else None."""
    s = _live_stock(h, bg)
    if s is not None:
        return s / _LIVE_DMD[(h, bg)]
    return None

@st.cache_data(show_spinner="Loading dataset...")
def load_data():
    df    = pd.read_csv(os.path.join(DATA_DIR,"raw","synthetic_demand.csv"), parse_dates=["date"])
    graph = pd.read_csv(os.path.join(DATA_DIR,"raw","hospital_graph.csv"))
    return df, graph

@st.cache_resource(show_spinner="Loading forecasting models...")
def load_forecasters():
    fcs = {}
    for h, bg in COMBOS:
        hf = HybridForecaster(h, bg); hf.load_models(); hf.optimise_weights(load_data()[0])
        fcs[(h, bg)] = hf
    return fcs

@st.cache_resource(show_spinner="Initialising redistribution engine...")
def load_engine(): return RedistributionEngine()

_ensure_log()
df, graph_df = load_data()
forecasters  = load_forecasters()
engine       = load_engine()
MIN_DATE, MAX_DATE = df["date"].min().date(), df["date"].max().date()

# =============================================================================
# SIDEBAR — with live status badge (auto-refreshes every 1 s when sim running)
# =============================================================================
with st.sidebar:
    st.image("https://img.icons8.com/color/96/blood-bag.png", width=60)
    st.title("Blood Inventory\nManagement System"); st.divider()
    selected_date = pd.Timestamp(st.date_input("Reference Date", value=MAX_DATE, min_value=MIN_DATE, max_value=MAX_DATE))
    hosp_filter = st.selectbox("Hospital Filter",    ["All"]+HOSPITALS)
    bg_filter   = st.selectbox("Blood Group Filter", ["All"]+BLOOD_GROUPS)
    st.divider(); st.caption(f"Data: {MIN_DATE} to {MAX_DATE}"); st.caption("Phase 1 — Synthetic data")
    st.divider()

    @st.fragment(run_every=1)
    def _sidebar_live():
        if _sim_active():
            sim_t = st.session_state.sim_time.strftime("%Y-%m-%d %H:%M")
            st.markdown(
                f"<div style='background:#27ae60;color:white;border-radius:8px;"
                f"padding:6px 12px;margin-bottom:8px;text-align:center;'>"
                f"<b>● LIVE</b> &nbsp; {sim_t}</div>",
                unsafe_allow_html=True)
            for h in HOSPITALS:
                opos = _live_stock(h, "O_positive") or 0
                oneg = _live_stock(h, "O_negative") or 0
                dos_p = _live_dos(h, "O_positive") or 0
                dos_n = _live_dos(h, "O_negative") or 0
                clr_p = _dos_color(dos_p); clr_n = _dos_color(dos_n)
                st.markdown(
                    f"<div style='font-size:11px;margin-bottom:4px;'>"
                    f"<b>{h.replace('Hospital_','H.')}</b> &nbsp;"
                    f"<span style='color:{clr_p};'>O+ {opos:.0f}u ({dos_p:.1f}d)</span> &nbsp;"
                    f"<span style='color:{clr_n};'>O- {oneg:.0f}u ({dos_n:.1f}d)</span>"
                    f"</div>",
                    unsafe_allow_html=True)
        else:
            st.caption("Start the Live Simulation (Tab 8) to see live stock here.")

    _sidebar_live()

    st.divider()
    st.markdown("**Simulation Controls**")
    st.caption("Start/pause/reset both Tab 8 & Tab 9 in sync.")

    _BOTH_START = pd.Timestamp("2026-01-01 06:00:00")
    _BOTH_STOCK = {
        ("Hospital_A","O_positive"):495, ("Hospital_A","O_negative"):198,
        ("Hospital_B","O_positive"):168, ("Hospital_B","O_negative"): 66,
        ("Hospital_C","O_positive"): 56, ("Hospital_C","O_negative"): 15,
    }

    def _reset_both():
        for prefix, time_key, inv_key, log_key, hist_key, tick_key, sh_key, ex_key, co_key in [
            ("sim_", "sim_time", "sim_inventory", "sim_log", "sim_history", "sim_last_tick",
             "sim_total_shortage", "sim_total_expired", "sim_total_collected"),
            ("sim2_", "sim2_time", "sim2_inventory", "sim2_log", "sim2_history", "sim2_last_tick",
             "sim2_total_shortage", "sim2_total_expired", "sim2_total_collected"),
        ]:
            st.session_state[prefix + "running"] = False
            st.session_state[time_key]  = _BOTH_START
            st.session_state[inv_key]   = {k: float(v) for k, v in _BOTH_STOCK.items()}
            st.session_state[log_key]   = []
            st.session_state[hist_key]  = []
            st.session_state[tick_key]  = None
            st.session_state[sh_key]    = {k: 0.0 for k in _BOTH_STOCK}
            st.session_state[ex_key]    = {k: 0.0 for k in _BOTH_STOCK}
            st.session_state[co_key]    = {k: 0.0 for k in _BOTH_STOCK}

    both_running = st.session_state.get("sim_running", False) or st.session_state.get("sim2_running", False)
    sb1, sb2 = st.columns(2)
    with sb1:
        if both_running:
            if st.button("⏸ Pause Both", key="sb_pause", use_container_width=True):
                st.session_state.sim_running  = False
                st.session_state.sim2_running = False
                st.rerun()
        else:
            if st.button("▶ Start Both", type="primary", key="sb_start", use_container_width=True):
                now = time.time()
                st.session_state.sim_running   = True;  st.session_state.sim_last_tick  = now
                st.session_state.sim2_running  = True;  st.session_state.sim2_last_tick = now
                st.rerun()
    with sb2:
        if st.button("↺ Reset Both", key="sb_reset", use_container_width=True):
            _reset_both(); st.rerun()

st.markdown("<div style='background:#c0392b;padding:10px 20px;border-radius:6px;margin-bottom:12px;'>"
            "<span style='color:white;font-size:16px;font-weight:bold;'>&#9888; All transfer decisions require human approval</span></div>",
            unsafe_allow_html=True)

tabs = st.tabs(["Overview & Inventory","Forecast & Model Report","Transfers & History","MLOps & Drift","Live Simulation","Baseline (No Transfers)"])

# =============================================================================
# TAB 1 — OVERVIEW  (live-aware, auto-refreshes every 2 s)
# =============================================================================
@st.fragment(run_every=2)
def _tab1_overview():
    # Source: live sim if running, else CSV snapshot
    if _sim_active():
        st.caption("🔴 **LIVE** — data from running simulation")
        opos = sum(_live_stock(h,"O_positive") or 0 for h in HOSPITALS)
        oneg = sum(_live_stock(h,"O_negative") or 0 for h in HOSPITALS)
        exp  = 0  # expiry tracking not in sim_inventory — show 0 when live
        summary_src = "live"
    else:
        snap = df[df["date"]<=selected_date].groupby(["hospital","blood_group"]).last().reset_index()
        opos = snap[snap["blood_group"]=="O_positive"]["stock_on_hand"].sum()
        oneg = snap[snap["blood_group"]=="O_negative"]["stock_on_hand"].sum()
        exp  = snap["units_expiring_soon"].sum()
        summary_src = "csv"

    c1,c2,c3 = st.columns(3)
    c1.metric("Total O+ Stock (network)", f"{int(opos):,} units")
    c2.metric("Total O- Stock (network)", f"{int(oneg):,} units")
    c3.metric("Units Expiring Soon (<=3d)",
              f"{int(exp):,} units" if summary_src == "csv" else "N/A (live mode)",
              delta=None if exp==0 or summary_src=="live" else "Action needed",
              delta_color="inverse")
    st.divider()

    pos = {"Hospital_A":(0.5,0.85),"Hospital_B":(0.15,0.2),"Hospital_C":(0.85,0.2)}
    fig = go.Figure()
    for _,r in graph_df.iterrows():
        a,b = r["from_hospital"],r["to_hospital"]
        fig.add_trace(go.Scatter(x=[pos[a][0],pos[b][0],None],y=[pos[a][1],pos[b][1],None],
            mode="lines",line=dict(color="lightgray",width=2),hoverinfo="none",showlegend=False))
        fig.add_annotation(x=(pos[a][0]+pos[b][0])/2,y=(pos[a][1]+pos[b][1])/2,
            text=f"{int(r['distance_km'])} km",showarrow=False,font=dict(size=10,color="gray"))

    summary_rows = []
    for h in HOSPITALS:
        if summary_src == "live":
            opos_v = int(_live_stock(h,"O_positive") or 0)
            oneg_v = int(_live_stock(h,"O_negative") or 0)
            ts     = opos_v + oneg_v
            md     = df[df["hospital"]==h]["demand"].mean()
            dos    = ts / md if md > 0 else 0
        else:
            hs     = snap[snap["hospital"]==h]
            ts     = int(hs["stock_on_hand"].sum())
            md     = df[df["hospital"]==h]["demand"].mean()
            dos    = ts / md if md > 0 else 0
            opos_v = int(hs[hs["blood_group"]=="O_positive"]["stock_on_hand"].values[0]) if len(hs[hs["blood_group"]=="O_positive"]) else 0
            oneg_v = int(hs[hs["blood_group"]=="O_negative"]["stock_on_hand"].values[0]) if len(hs[hs["blood_group"]=="O_negative"]) else 0

        fig.add_trace(go.Scatter(x=[pos[h][0]],y=[pos[h][1]],mode="markers+text",
            marker=dict(size=40,color=_dos_color(dos),line=dict(width=2,color="black")),
            text=[h.replace("Hospital_","H.")],textposition="top center",textfont=dict(size=11,color="black"),
            hovertemplate=f"<b>{h}</b><br>Total stock: {ts}<br>DOS: {dos:.1f}<br>O+: {opos_v}<br>O-: {oneg_v}<extra></extra>",
            showlegend=False))
        summary_rows.append({"Hospital":h,"O+ Stock":opos_v,"O- Stock":oneg_v,"Total Stock":ts,
            "Days of Supply":round(dos,1),"Status":"CRITICAL" if dos<2 else ("LOW" if dos<4 else "HEALTHY")})

    for label,color in [("HEALTHY","#27ae60"),("LOW","#f39c12"),("CRITICAL","#e74c3c")]:
        fig.add_trace(go.Scatter(x=[None],y=[None],mode="markers",marker=dict(size=12,color=color),name=label))
    fig.update_layout(height=350,xaxis=dict(visible=False,range=[-0.1,1.1]),
                      yaxis=dict(visible=False,range=[0,1.1]),margin=dict(l=0,r=0,t=10,b=0),plot_bgcolor="white")
    fig = _apply_white_theme(fig)
    st.plotly_chart(fig, width='stretch')
    st.subheader("Hospital Summary")
    summ_df = pd.DataFrame(summary_rows)
    st.dataframe(summ_df.style.apply(lambda col:[
        f"background-color:{'#fadbd8' if v=='CRITICAL' else '#fef9e7' if v=='LOW' else '#d5f5e3'}"
        for v in col] if col.name=="Status" else [""]*len(col),axis=0),width='stretch',hide_index=True)

# =============================================================================
@st.fragment(run_every=2)
def _tab2_inventory():
    if _sim_active():
        st.caption("🔴 **LIVE** — data from running simulation")
        inv_rows = []
        for h in HOSPITALS:
            if hosp_filter != "All" and h != hosp_filter: continue
            for bg in BLOOD_GROUPS:
                if bg_filter != "All" and bg != bg_filter: continue
                stock = _live_stock(h, bg) or 0
                daily = _LIVE_DMD[(h, bg)]
                dos   = stock / daily
                inv_rows.append({"Hospital":h,"Blood Group":bg,
                    "Stock on Hand":int(stock),"Expiring Soon":"N/A",
                    "Days of Supply":round(dos,1),"Mean Daily Demand":daily,
                    "Status":"CRITICAL" if dos<2 else ("LOW" if dos<4 else "HEALTHY")})
        title_suffix = f"LIVE  {st.session_state.sim_time.strftime('%Y-%m-%d %H:%M')}"
    else:
        snap2 = df[df["date"]<=selected_date].groupby(["hospital","blood_group"]).last().reset_index()
        inv_rows = []
        for _,r in snap2.iterrows():
            h,bg = r["hospital"],r["blood_group"]
            if hosp_filter!="All" and h!=hosp_filter: continue
            if bg_filter!="All" and bg!=bg_filter: continue
            md  = df[(df["hospital"]==h)&(df["blood_group"]==bg)]["demand"].mean()
            dos = r["stock_on_hand"]/md if md>0 else 0
            inv_rows.append({"Hospital":h,"Blood Group":bg,"Stock on Hand":int(r["stock_on_hand"]),
                "Expiring Soon":int(r["units_expiring_soon"]),"Days of Supply":round(dos,1),
                "Mean Daily Demand":round(md,1),"Status":"CRITICAL" if dos<2 else ("LOW" if dos<4 else "HEALTHY")})
        title_suffix = str(selected_date.date())

    inv_df = pd.DataFrame(inv_rows)
    if not inv_df.empty:
        def _cs(v):
            if v=="CRITICAL": return "background-color:#fadbd8;color:#c0392b;font-weight:bold"
            if v=="LOW":      return "background-color:#fef9e7;color:#d35400;font-weight:bold"
            return "background-color:#d5f5e3;color:#1e8449"
        st.dataframe(inv_df.style.applymap(_cs, subset=["Status"]), width='stretch', hide_index=True)
        chart_df = inv_df.copy()
        chart_df["Stock on Hand"] = chart_df["Stock on Hand"].astype(float)
        fig_inv = px.bar(chart_df, x="Hospital", y="Stock on Hand", color="Blood Group", barmode="group",
            color_discrete_map={"O_positive":"#e74c3c","O_negative":"#3498db"},
            title=f"Stock Levels by Hospital and Blood Group  ({title_suffix})")
        fig_inv.update_layout(height=350)
        fig_inv = _apply_white_theme(fig_inv)
        st.plotly_chart(fig_inv, width='stretch')
    else:
        st.info("No data for selected filters.")

with tabs[0]:
    st.subheader("Network Overview")
    _tab1_overview()
    st.divider()
    st.subheader("Inventory Status")
    _tab2_inventory()

# =============================================================================
# TAB 3 — FORECAST  (live when simulation is running, static otherwise)
# =============================================================================
with tabs[1]:
    fc_sel_left, fc_sel_right = st.columns(2)
    with fc_sel_left:
        fc_hosp = st.selectbox("Hospital",    HOSPITALS,    key="fc_hosp")
    with fc_sel_right:
        fc_bg   = st.selectbox("Blood Group", BLOOD_GROUPS, key="fc_bg")

    @st.fragment(run_every=2)
    def _tab3_forecast():
        hf     = forecasters[(fc_hosp, fc_bg)]
        sub_fc = df[(df["hospital"]==fc_hosp)&(df["blood_group"]==fc_bg)].sort_values("date")

        # ── Choose "today" ────────────────────────────────────────────────────
        sim_on = _sim_active()
        if sim_on:
            sim_now = st.session_state.sim_time
            today   = sim_now.normalize()          # midnight — forecast anchor
            st.caption(
                f"📡 Live — sim clock **{sim_now.strftime('%Y-%m-%d  %H:%M')}**  "
                f"| forecast advances each new sim-day  (auto-refresh every 2 s)"
            )
        else:
            today = pd.Timestamp(selected_date)

        # ── Build forecast for next 7 days from `today` ──────────────────────
        # Bypass hf.predict() — it resets _last_date to the CSV's last row.
        hf._prophet._last_date = today
        pfc       = hf._prophet.predict(periods=7)
        lfc       = hf._lstm.predict(sub_fc.tail(30))
        _n        = min(len(pfc), len(lfc), 7)
        _hp       = (hf.alpha * pfc["yhat"].values[:_n] + hf.beta * lfc[:_n]).clip(0)
        fc_dates  = pd.date_range(start=today + pd.Timedelta(days=1), periods=_n)
        hfc = pd.DataFrame({"date": fc_dates,
                            "prophet_pred": pfc["yhat"].values[:_n],
                            "lstm_pred":    lfc[:_n],
                            "hybrid_pred":  _hp})

        # ── Last-14-days actual ───────────────────────────────────────────────
        if sim_on:
            _acc  = st.session_state.sim_day_acc.get((fc_hosp, fc_bg), {})
            _past = sorted(_acc.items())[-14:]
            last14_dates  = [pd.Timestamp(d) for d, _ in _past]
            last14_demand = [v for _, v in _past]
        else:
            _tmp = sub_fc[sub_fc["date"] <= today].tail(14)
            last14_dates  = _tmp["date"].tolist()
            last14_demand = _tmp["demand"].tolist()

        # ── Forecast chart ────────────────────────────────────────────────────
        st.subheader("7-Day Demand Forecast")
        fig_fc = go.Figure()
        if last14_dates:
            fig_fc.add_trace(go.Scatter(
                x=last14_dates, y=last14_demand, mode="lines+markers",
                name="Actual (sim)", line=dict(color="#2980b9", width=2)))
        fig_fc.add_trace(go.Scatter(
            x=hfc["date"], y=hfc["prophet_pred"], mode="lines+markers",
            name="Prophet", line=dict(color="#8e44ad", width=1.5, dash="dash")))
        fig_fc.add_trace(go.Scatter(
            x=hfc["date"], y=hfc["lstm_pred"], mode="lines+markers",
            name="LSTM", line=dict(color="#e67e22", width=1.5, dash="dash")))
        fig_fc.add_trace(go.Scatter(
            x=hfc["date"], y=hfc["hybrid_pred"], mode="lines+markers",
            name=f"Hybrid (α={hf.alpha:.2f})", line=dict(color="#27ae60", width=3)))
        fig_fc.add_vline(x=today.timestamp()*1000, line_width=1.5, line_dash="dot",
            line_color="gray", annotation_text="Today", annotation_position="top left")
        fig_fc.add_vrect(x0=hfc["date"].min(), x1=hfc["date"].max(),
            fillcolor="#27ae60", opacity=0.05,
            annotation_text="7-day window", annotation_position="top left")
        fig_fc.update_layout(
            title=f"Demand Forecast — {fc_hosp}  |  {fc_bg.replace('_',' ')}",
            xaxis_title="Date", yaxis_title="Demand (units)",
            height=420, hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.08),
            margin=dict(l=0, r=0, t=85, b=0),
        )
        fig_fc = _apply_white_theme(fig_fc)
        st.plotly_chart(fig_fc, use_container_width=True)

        # ── Day-by-day table ──────────────────────────────────────────────────
        st.subheader("Day-by-Day Forecast Values")
        fc_tbl = hfc.copy()
        fc_tbl["Day"] = [d.strftime("%a %d %b") for d in fc_tbl["date"]]
        fc_tbl = (fc_tbl[["Day","prophet_pred","lstm_pred","hybrid_pred"]]
                  .rename(columns={"prophet_pred":"Prophet","lstm_pred":"LSTM","hybrid_pred":"Hybrid"}))
        fc_tbl[["Prophet","LSTM","Hybrid"]] = fc_tbl[["Prophet","LSTM","Hybrid"]].round(1)
        st.dataframe(fc_tbl, use_container_width=True, hide_index=True)

        if not sim_on:
            return

        # ── Live today accuracy ───────────────────────────────────────────────
        st.divider()
        st.subheader("Live Accuracy")
        _acc_today  = st.session_state.sim_day_acc.get((fc_hosp, fc_bg), {})
        _fc_today   = st.session_state.sim_day_fc.get((fc_hosp, fc_bg), {})
        _today_str  = str(today.date())
        _actual_so_far = _acc_today.get(_today_str, 0.0)
        _fc_so_far     = _fc_today.get(_today_str, float(_LIVE_DMD[(fc_hosp, fc_bg)]))

        la1, la2, la3 = st.columns(3)
        la1.metric("Today's demand so far",   f"{_actual_so_far:.1f} units")
        la2.metric("Model expects today",      f"{_fc_so_far:.1f} units")
        _today_err = abs(_actual_so_far - _fc_so_far) / max(_fc_so_far, 1) * 100
        la3.metric("Running APE today",        f"{_today_err:.1f}%",
                   delta=f"{'over' if _actual_so_far > _fc_so_far else 'under'} by "
                         f"{abs(_actual_so_far-_fc_so_far):.1f} u",
                   delta_color="off")

        # ── Historical accuracy: all completed days ───────────────────────────
        acc_log   = st.session_state.sim_accuracy_log
        combo_log = [r for r in acc_log if r["hospital"]==fc_hosp and r["blood_group"]==fc_bg]
        if not combo_log:
            st.caption("⏳ Historical accuracy appears after the first completed sim-day.")
            return

        # Use sim_accuracy_log directly — forecast column = sim_day_fc (seasonal daily mean,
        # already encodes weekday + dengue multipliers). No ML re-call, no broken trend.
        st.subheader("Forecast vs Actual — Last 7 Completed Sim-Days")
        st.caption(
            "Forecast = seasonally-adjusted daily mean (weekday × dengue multiplier applied per tick). "
            "Actual = stochastic demand recorded at day-boundary. Both from the live simulation — "
            "no ML extrapolation to 2026."
        )
        rdf = (pd.DataFrame(combo_log)
               .sort_values("date")
               .tail(7)
               .reset_index(drop=True))

        rdf["APE (%)"] = rdf["ape"].round(1)
        mape  = rdf["APE (%)"].mean()
        mae   = (rdf["actual"] - rdf["forecast"]).abs().mean()
        n_days = len(pd.DataFrame(combo_log))

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total sim-days recorded", n_days)
        m2.metric("7-day window MAPE", f"{mape:.1f}%")
        m3.metric("7-day window MAE", f"{mae:.1f} u")
        m4.metric("Sim date", str(today.date()))

        fig_acc = go.Figure()
        fig_acc.add_trace(go.Scatter(
            x=rdf["date"], y=rdf["actual"],
            name="Actual Demand", mode="lines+markers",
            line=dict(color="#2980b9", width=2)))
        fig_acc.add_trace(go.Scatter(
            x=rdf["date"], y=rdf["forecast"],
            name="Model Forecast (seasonal mean)", mode="lines+markers",
            line=dict(color="#27ae60", width=2, dash="dash")))
        fig_acc.add_trace(go.Scatter(
            x=rdf["date"], y=rdf["APE (%)"],
            name="APE (%)", yaxis="y2", mode="lines+markers",
            line=dict(color="#e74c3c", width=1.5, dash="dot")))
        fig_acc.update_layout(
            title=f"Last 7 Sim-Days: Forecast vs Actual  |  MAPE {mape:.1f}%  |  MAE {mae:.1f} u",
            height=360, hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.08),
            margin=dict(l=0, r=0, t=85, b=0),
            xaxis_title="Simulation Date", yaxis_title="Demand (units)",
            yaxis2=dict(title="APE (%)", overlaying="y", side="right",
                        showgrid=False, range=[0, 100]),
        )
        fig_acc = _apply_white_theme(fig_acc)
        st.plotly_chart(fig_acc, use_container_width=True)
        st.dataframe(
            rdf[["date", "forecast", "actual", "APE (%)"]].rename(
                columns={"date": "Date", "forecast": "Forecast (units)", "actual": "Actual (units)"}),
            use_container_width=True, hide_index=True)

    _tab3_forecast()

    # =========================================================================
    # MAPE COMPARISON — Historic (static) vs Live simulation (live)
    # =========================================================================
    st.divider()
    st.subheader("MAPE Comparison — Historic Test Set vs Live Simulation")
    st.caption(
        "**Historic** = model evaluated on the held-out 2021–2023 training data. "
        "Frozen — never changes. &nbsp;|&nbsp; "
        "**Live** = Hybrid model tracking stochastic demand in the 2026 simulation. "
        "Updates as each sim-day completes."
    )

    _mc_path = os.path.join(REPORTS_DIR, "model_comparison.csv")
    if os.path.exists(_mc_path):
        _mc = pd.read_csv(_mc_path)
        _mc["combo"] = (_mc["hospital"].str.replace("Hospital_", "H.")
                        + "  " + _mc["blood_group"].str.replace("_", " "))
        _MODEL_ORDER  = ["Hybrid", "Prophet", "LSTM", "MovingAverage"]
        _MODEL_COLORS = {"Hybrid":"#27ae60","Prophet":"#8e44ad","LSTM":"#e67e22","MovingAverage":"#95a5a6"}

        # ── Dual scorecard: historic avg MAPE per model ──────────────────────
        st.markdown("#### Historic MAPE per Model (2021–2023 test set, static)")
        sc_cols = st.columns(4)
        for col, m in zip(sc_cols, _MODEL_ORDER):
            sub = _mc[_mc["model"] == m]
            col.metric(m, f"{sub['MAPE'].mean():.1f}%",
                       f"best {sub['MAPE'].min():.1f}%", delta_color="off")

        @st.fragment(run_every=3)
        def _mape_comparison_panel():
            live_log = st.session_state.get("sim_accuracy_log", [])
            if live_log:
                _df_live = pd.DataFrame(live_log)
                _df_live["combo"] = (_df_live["hospital"].str.replace("Hospital_","H.")
                                     + "  " + _df_live["blood_group"].str.replace("_"," "))
                live_avg = _df_live.groupby("combo")["ape"].mean()
                n_live_days = _df_live["date"].nunique()
                st.markdown(f"#### Live MAPE — Hybrid Model ({n_live_days} sim-days recorded)")
                lm1, lm2 = st.columns(2)
                lm1.metric("Overall Live MAPE", f"{_df_live['ape'].mean():.1f}%")
                lm2.metric("Sim-Days Recorded", n_live_days)
            else:
                st.info("Start the simulation (Tab 8) — live MAPE appears after the first completed sim-day.")
                live_avg = pd.Series(dtype=float)

            # ── Side-by-side heatmaps ────────────────────────────────────────
            h_col, l_col = st.columns(2)

            with h_col:
                st.markdown("**Historic MAPE — all models (frozen)**")
                _hist_hybrid = _mc[_mc["model"]=="Hybrid"].set_index("combo")["MAPE"]
                _pivot_h = (_mc[_mc["model"].isin(_MODEL_ORDER)]
                            .pivot(index="combo", columns="model", values="MAPE")[_MODEL_ORDER])
                fig_hh = go.Figure(go.Heatmap(
                    z=_pivot_h.values,
                    x=_pivot_h.columns.tolist(),
                    y=_pivot_h.index.tolist(),
                    colorscale=[[0,"#27ae60"],[0.45,"#f1c40f"],[1,"#e74c3c"]],
                    text=[[f"{v:.1f}%" for v in row] for row in _pivot_h.values],
                    texttemplate="%{text}", textfont={"size":12},
                    showscale=False, zmin=0, zmax=55,
                ))
                fig_hh.update_layout(height=260, margin=dict(l=0,r=0,t=10,b=0),
                                     xaxis=dict(side="top"))
                fig_hh = _apply_white_theme(fig_hh)
                st.plotly_chart(fig_hh, use_container_width=True)

            with l_col:
                st.markdown("**Live MAPE — Hybrid Model (simulation)**")
                if not live_avg.empty:
                    _combos = live_avg.index.tolist()
                    fig_lh = go.Figure(go.Heatmap(
                        z=[[live_avg.get(c, float("nan"))] for c in _combos],
                        x=["Hybrid (Live)"],
                        y=_combos,
                        colorscale=[[0,"#27ae60"],[0.45,"#f1c40f"],[1,"#e74c3c"]],
                        text=[[f"{live_avg.get(c,0):.1f}%"] for c in _combos],
                        texttemplate="%{text}", textfont={"size":12},
                        showscale=False, zmin=0, zmax=55,
                    ))
                    fig_lh.update_layout(height=260, margin=dict(l=0,r=0,t=10,b=0),
                                         xaxis=dict(side="top"))
                    fig_lh = _apply_white_theme(fig_lh)
                    st.plotly_chart(fig_lh, use_container_width=True)
                else:
                    st.markdown("<div style='height:260px;display:flex;align-items:center;"
                                "justify-content:center;color:gray;'>Waiting for simulation…</div>",
                                unsafe_allow_html=True)

        _mape_comparison_panel()

        # ── Full historic breakdown (static) ──────────────────────────────────
        st.divider()
        st.subheader("Historic Model Performance — Full Breakdown (Static)")
        st.caption("All numbers below are from the 2021–2023 test set. They do not change.")

        st.markdown("**Side-by-Side Comparison**")
        metric_sel = st.radio("Metric", ["MAPE","MAE","RMSE"], horizontal=True, key="acc_metric")
        y_label = f"{metric_sel} ({'%' if metric_sel=='MAPE' else 'units'})"
        fig_bar = px.bar(_mc, x="combo", y=metric_sel, color="model", barmode="group",
                         color_discrete_map=_MODEL_COLORS,
                         labels={"combo":"Hospital + Blood Group", metric_sel: y_label})
        fig_bar.update_layout(height=340, legend=dict(orientation="h",yanchor="bottom",y=1.08),
                              margin=dict(l=0,r=0,t=75,b=0), xaxis_tickangle=-20)
        fig_bar = _apply_white_theme(fig_bar)
        st.plotly_chart(fig_bar, use_container_width=True)

        st.markdown("**Winner per Combination**")
        winner_rows = []
        for combo, grp in _mc.groupby("combo"):
            best   = grp.loc[grp["MAPE"].idxmin()]
            others = grp[grp["model"] != best["model"]].nsmallest(1,"MAPE")
            margin = f"+{others.iloc[0]['MAPE']-best['MAPE']:.1f} pp" if len(others) else "—"
            winner_rows.append({"Combination":combo, "Best Model":best["model"],
                                 "MAPE (%)":f"{best['MAPE']:.1f}", "MAE (units)":f"{best['MAE']:.2f}",
                                 "RMSE (units)":f"{best['RMSE']:.2f}", "Edge vs 2nd":margin})
        st.dataframe(pd.DataFrame(winner_rows), use_container_width=True, hide_index=True)

        # ── Naïve Baseline Comparison ─────────────────────────────────────────
        st.divider()
        st.markdown("#### Naïve Baseline Comparison (same-day-last-week)")
        st.caption(
            "Naïve baseline predicts tomorrow's demand = same weekday last week. "
            "Hybrid MAPE is shown alongside to quantify model improvement."
        )

        @st.cache_data(show_spinner=False)
        def _naive_baseline_mape():
            demand_csv = os.path.join(BASE_DIR, "data", "raw", "synthetic_demand.csv")
            df_all = pd.read_csv(demand_csv, parse_dates=["date"])
            rows = []
            for (h, bg), grp in df_all.groupby(["hospital", "blood_group"]):
                grp = grp.sort_values("date").reset_index(drop=True)
                grp["naive"] = grp["demand"].shift(7)
                grp = grp.dropna(subset=["naive"]).tail(260)
                naive_mape = (np.abs(grp["demand"] - grp["naive"]) /
                              grp["demand"].replace(0, np.nan)).mean() * 100
                rows.append({"combo": f"{h}/{bg}", "hospital": h,
                             "blood_group": bg, "naive_mape": round(naive_mape, 2)})
            return pd.DataFrame(rows)

        _naive_df = _naive_baseline_mape()
        _hybrid_mc = _mc[_mc["model"] == "Hybrid"][["hospital", "blood_group", "MAPE"]].rename(columns={"MAPE": "hybrid_mape"})
        _cmp = _naive_df.merge(_hybrid_mc, on=["hospital", "blood_group"])
        _cmp["improvement"] = (_cmp["naive_mape"] - _cmp["hybrid_mape"]).round(2)
        _cmp["improvement_pct"] = ((_cmp["improvement"] / _cmp["naive_mape"]) * 100).round(1)

        # Summary metrics
        nb1, nb2, nb3 = st.columns(3)
        nb1.metric("Avg Naïve MAPE", f"{_cmp['naive_mape'].mean():.1f}%")
        nb2.metric("Avg Hybrid MAPE", f"{_cmp['hybrid_mape'].mean():.1f}%")
        nb3.metric("Avg Improvement", f"{_cmp['improvement'].mean():.1f} pp",
                   f"{_cmp['improvement_pct'].mean():.0f}% better", delta_color="normal")

        # Per-combination table
        st.dataframe(
            _cmp[["combo", "naive_mape", "hybrid_mape", "improvement", "improvement_pct"]]
            .rename(columns={
                "combo": "Combination",
                "naive_mape": "Naïve MAPE (%)",
                "hybrid_mape": "Hybrid MAPE (%)",
                "improvement": "Improvement (pp)",
                "improvement_pct": "Reduction (%)"
            }),
            use_container_width=True, hide_index=True
        )

        st.markdown("**Actual vs Predicted — Prophet on Test Set (2023 H2)**")
        st.caption("Rolling 7-day forecasts — honest out-of-sample evaluation on held-out data.")

        @st.cache_data(show_spinner="Generating test-set predictions…")
        def _prophet_test_preds(hospital, blood_group):
            from prophet_model import ProphetForecaster
            demand_csv = os.path.join(BASE_DIR, "data", "raw", "synthetic_demand.csv")
            df_all = pd.read_csv(demand_csv, parse_dates=["date"])
            sub = (df_all[(df_all["hospital"]==hospital)&(df_all["blood_group"]==blood_group)]
                   .sort_values("date").reset_index(drop=True))
            test_start = len(sub) - 180
            pf = ProphetForecaster(hospital, blood_group); pf.load()
            aa, ap, dates_out = [], [], []
            for step in range(0, 180, 7):
                cut = test_start + step
                if cut + 7 > len(sub): break
                actual = sub.iloc[cut+1:cut+8]["demand"].values
                pf._last_date = sub.iloc[:cut+1]["date"].iloc[-1]
                pp = pf.predict(periods=7)["yhat"].values[:len(actual)]
                aa.extend(actual); ap.extend(pp.clip(0))
                dates_out.extend(sub.iloc[cut+1:cut+1+len(actual)]["date"].tolist())
            return dates_out, np.array(aa), np.array(ap)

        av_l, av_r = st.columns([1,3])
        with av_l:
            av_h  = st.selectbox("Hospital",    HOSPITALS,    key="av_hosp")
            av_bg = st.selectbox("Blood Group", BLOOD_GROUPS, key="av_bg")
        with av_r:
            try:
                dates_a, actual_a, pred_a = _prophet_test_preds(av_h, av_bg)
                mask = actual_a > 1
                mape_v = float(np.mean(np.abs((actual_a[mask]-pred_a[mask])/actual_a[mask])*100))
                mae_v  = float(np.mean(np.abs(actual_a-pred_a)))
                ma7    = pd.Series(actual_a).rolling(7,min_periods=1).mean().values
                fig_av = go.Figure()
                fig_av.add_trace(go.Scatter(x=dates_a,y=actual_a,name="Actual",
                    line=dict(color="#2c3e50",width=2)))
                fig_av.add_trace(go.Scatter(x=dates_a,y=pred_a,name="Prophet Forecast",
                    line=dict(color="#8e44ad",width=2,dash="dash")))
                fig_av.add_trace(go.Scatter(x=dates_a,y=ma7,name="7-day MA",
                    line=dict(color="#95a5a6",width=1.5,dash="dot")))
                fig_av.update_layout(height=360, hovermode="x unified",
                    title=f"{av_h.replace('Hospital_','H.')} {av_bg.replace('_',' ')} — MAPE: {mape_v:.1f}%  MAE: {mae_v:.2f}u",
                    legend=dict(orientation="h",yanchor="bottom",y=1.08),
                    margin=dict(l=0,r=0,t=85,b=0),
                    xaxis_title="Date", yaxis_title="Demand (units)")
                fig_av = _apply_white_theme(fig_av)
                st.plotly_chart(fig_av, use_container_width=True)
            except Exception as e:
                st.error(f"Could not generate predictions: {e}")

        st.markdown("**Full Metrics Table**")
        st.dataframe(_mc.rename(columns={"hospital":"Hospital","blood_group":"Blood Group",
            "model":"Model","MAE":"MAE (units)","RMSE":"RMSE (units)","MAPE":"MAPE (%)"})
            .drop(columns=["combo"]), use_container_width=True, hide_index=True)
    else:
        st.warning("Run `src/hybrid_ensemble.py` first to generate model_comparison.csv.")

# =============================================================================
# TAB 4 — TRANSFERS  (live auto-transfers at top when sim is running)
# =============================================================================
with tabs[2]:
    st.subheader("Transfer Recommendations")

    # ── Live auto-transfer feed ────────────────────────────────────────────────
    @st.fragment(run_every=2)
    def _live_transfer_feed():
        if _sim_active() and st.session_state.get("sim_log"):
            recent = [e for e in st.session_state.sim_log[-10:]
                      if e.get("donor") != "Central Bank"]
            if recent:
                st.markdown("#### Live Auto-Transfers (no approval needed)")
                for ev in reversed(recent):
                    bg_clr = "#fde8e8" if ev["blood_group"]=="O_negative" else "#eafaf1"
                    border = "#c0392b" if ev["blood_group"]=="O_negative" else "#27ae60"
                    st.markdown(
                        f"<div style='background:{bg_clr};padding:6px 14px;border-radius:6px;"
                        f"margin-bottom:4px;font-size:13px;border-left:4px solid {border};'>"
                        f"🕐 <b>{ev['sim_time']}</b> &nbsp;·&nbsp; "
                        f"<b>{ev['donor'].replace('Hospital_','H.')}</b> ➜ "
                        f"<b>{ev['recipient'].replace('Hospital_','H.')}</b> &nbsp;·&nbsp; "
                        f"{ev['distance_km']} km &nbsp;·&nbsp; "
                        f"<b>{ev['blood_group'].replace('_',' ')}</b> &nbsp;·&nbsp; "
                        f"<b>{ev['qty']} units</b>"
                        f"</div>",
                        unsafe_allow_html=True)
                st.divider()

    _live_transfer_feed()

    # ── Static ML transfer recommendations (unchanged) ─────────────────────────
    st.info("Human approval required for all transfers. Decisions are logged for model learning.")
    fc_dict = {}
    for (h,bg),hf_t in forecasters.items():
        try:    fc_dict.setdefault(h,{})[bg] = float(hf_t.predict(df,target_date=selected_date)["hybrid_pred"].sum())
        except: fc_dict.setdefault(h,{})[bg] = 0.0
    recs = engine.generate_recommendations(df, fc_dict, selected_date)

    if recs.empty:
        st.success(f"No transfer recommendations for {selected_date.date()} — all hospitals balanced.")
        st.caption("Recommendations appear when any hospital has <9 days supply while another has >8. Try early Jan 2021.")
    else:
        st.markdown(f"**{len(recs)} recommendation(s) found for {selected_date.date()}**")
        if "approved" not in st.session_state: st.session_state.approved = set()
        if "rejected" not in st.session_state: st.session_state.rejected = set()

        for i,row in recs.iterrows():
            uid = f"{row['donor_hospital']}_{row['recipient_hospital']}_{row['blood_group']}_{i}"
            ci, ca, cr = st.columns([5,1,1])
            with ci:
                st.markdown(f"**#{i+1}** &nbsp; `{row['donor_hospital']}` &rarr; `{row['recipient_hospital']}` "
                    f"&nbsp;|&nbsp; **{row['blood_group'].replace('_',' ')}** &nbsp;|&nbsp; "
                    f"**{int(row['transfer_qty'])} units** &nbsp;|&nbsp; PS: **{row['priority_score']:.3f}** "
                    f"&nbsp;|&nbsp; {row['distance_km']}km/{row['transit_hr']}hr "
                    f"&nbsp;|&nbsp; Donor DOS:{row.get('donor_dos','?')}d &nbsp;|&nbsp; Recip DOS:{row.get('recipient_dos','?')}d")
            if uid not in st.session_state.approved and uid not in st.session_state.rejected:
                with ca:
                    if st.button("APPROVE", key=f"app_{uid}", type="primary"):
                        st.session_state.approved.add(uid)
                        _append_log(_log_entry(row,"APPROVED"))
                        st.toast("Transfer approved and logged!", icon="✅"); st.rerun()
                with cr:
                    if st.button("REJECT", key=f"rej_{uid}"):
                        st.session_state.rejected.add(uid)
            if uid in st.session_state.rejected and uid not in st.session_state.approved:
                reason = st.selectbox("Rejection reason",
                    ["Clinical concern","Logistical issue","Insufficient quantity",
                     "Better alternative available","Other"], key=f"reason_{uid}")
                if st.button("Confirm Rejection", key=f"conf_{uid}"):
                    _append_log(_log_entry(row,"REJECTED",reason))
                    st.toast("Rejection logged for model learning.", icon="ℹ️"); st.rerun()
            if uid in st.session_state.approved:
                st.success("APPROVED — logged to transfer_log.csv")
            st.divider()

    st.divider()
    st.subheader("Transfer Decision History")
    _ensure_log(); log_df = pd.read_csv(LOG_PATH)
    if log_df.empty:
        st.info("No transfer decisions logged yet. Use the recommendations above to approve or reject.")
    else:
        ta = (log_df["decision"]=="APPROVED").sum(); tr = (log_df["decision"]=="REJECTED").sum()
        h1, h2, h3 = st.columns(3)
        h1.metric("Total Approved", int(ta)); h2.metric("Total Rejected", int(tr))
        h3.metric("Approval Rate", f"{100*ta/len(log_df):.1f}%")
        st.dataframe(log_df, use_container_width=True, hide_index=True)
        if len(log_df) > 1:
            log_df["date"] = pd.to_datetime(log_df["date"])
            log_df["week"] = log_df["date"].dt.to_period("W").astype(str)
            weekly = log_df.groupby(["week","decision"]).size().reset_index(name="count")
            fig_h = px.bar(weekly, x="week", y="count", color="decision", barmode="group",
                color_discrete_map={"APPROVED":"#27ae60","REJECTED":"#e74c3c"},
                title="Approvals vs Rejections by Week")
            fig_h.update_layout(height=300); fig_h = _apply_white_theme(fig_h); st.plotly_chart(fig_h, use_container_width=True)

# =============================================================================
# TAB 5 — MLOps & Drift
# =============================================================================
with tabs[3]:
    st.subheader("MLOps — Model Drift Detection & Continuous Learning")
    HISTORY_CSV = os.path.join(REPORTS_DIR,"mlops_history.csv")
    HEALTH_TXT  = os.path.join(REPORTS_DIR,"mlops_health_report.txt")
    W2_STORE    = os.path.join(REPORTS_DIR,"route_weights.json")

    col_btn, col_last = st.columns([2,3])
    with col_btn:
        if st.button("Trigger Retraining Now", type="primary"):
            with st.spinner("Running drift check..."):
                try:
                    mlops_monitor.run_monitor(simulate=False, verbose=False)
                    st.toast("MLOps check complete.", icon="✅"); st.rerun()
                except Exception as e:
                    st.error(f"Monitor error: {e}")
    with col_last:
        if os.path.exists(HISTORY_CSV):
            hp = pd.read_csv(HISTORY_CSV)
            if not hp.empty: st.caption(f"Last run: **{hp['timestamp'].max()}**")
        else:
            st.caption("No runs yet.")
    st.divider()

    if os.path.exists(HISTORY_CSV):
        hist_df = pd.read_csv(HISTORY_CSV, parse_dates=["timestamp"])
        if not hist_df.empty:
            st.subheader("Rolling MAPE Evolution")
            latest = hist_df.sort_values("timestamp").groupby(["hospital","blood_group"]).last().reset_index()
            for col,(_, r) in zip(st.columns(len(latest)), latest.iterrows()):
                tag = f"{r['hospital'].replace('Hospital_','H.')} {r['blood_group'].replace('_',' ')}"
                delta = r["rolling_mape"]-r["baseline_mape"]
                col.metric(tag, f"{r['rolling_mape']:.1f}%", f"{delta:+.1f}pp vs baseline",
                           delta_color="normal" if delta<=2 else ("off" if delta<=5 else "inverse"))
            hist_df["combo"] = hist_df["hospital"]+" "+hist_df["blood_group"]
            fig_d = px.line(hist_df,x="timestamp",y="rolling_mape",color="combo",markers=True,
                title="Rolling MAPE Over Time (weekly checks)",
                labels={"rolling_mape":"Rolling MAPE (%)","timestamp":"Check Time"})
            for _,r in latest.iterrows():
                fig_d.add_hline(y=r["baseline_mape"],line_dash="dot",line_color="gray",opacity=0.4)
            fig_d.update_layout(height=380,hovermode="x unified"); fig_d = _apply_white_theme(fig_d); st.plotly_chart(fig_d, width='stretch')
            drift_ev = hist_df[hist_df["drift_detected"]==True]
            if not drift_ev.empty:
                st.warning(f"**{len(drift_ev)} drift event(s) detected**")
                st.dataframe(drift_ev[["timestamp","hospital","blood_group","baseline_mape","rolling_mape"]]
                    .rename(columns={"baseline_mape":"Baseline MAPE","rolling_mape":"Rolling MAPE"}),
                    width='stretch', hide_index=True)
            else:
                st.success("No drift events detected. All models healthy.")
        else:
            st.info("No monitoring history yet.")
    else:
        st.info("No monitoring history yet.")
    st.divider()

    st.subheader("HITL Feedback — Transport Score (W2) Adjustments")
    if os.path.exists(W2_STORE):
        with open(W2_STORE) as f: rw = json.load(f)
        if rw:
            wdf = pd.DataFrame([{"Route":k,"W2 Multiplier":v,
                "Status":"Penalised" if v<1.0 else ("Boosted" if v>1.0 else "Default")}
                for k,v in rw.items()])
            def _w2c(v):
                if v=="Penalised": return "background-color:#fadbd8;color:#c0392b"
                if v=="Boosted":   return "background-color:#d5f5e3;color:#1e8449"
                return ""
            st.dataframe(wdf.style.applymap(_w2c, subset=["Status"]), width='stretch', hide_index=True)
        else:
            st.info("No route adjustments yet.")
    else:
        st.info("No route weights file found.")
    st.divider()

    st.subheader("Latest Health Report")
    if os.path.exists(HEALTH_TXT):
        with open(HEALTH_TXT, encoding="utf-8") as f: st.code(f.read(), language=None)
    else:
        st.info("No health report yet.")

# =============================================================================
# TAB 8 — LIVE SIMULATION  (unchanged from app2.py)
# =============================================================================
@st.fragment(run_every=1)
def _tab8_simulation():
    st.subheader("Live Blood Inventory Simulation")
    st.caption("Real hospital simulation — blood is consumed, collected in batches, expires, and auto-transfers fire without approval. Simulation period: 2026 onwards (out-of-sample).")

    _SIM_H   = ["Hospital_A", "Hospital_B", "Hospital_C"]
    _SIM_BG  = ["O_positive", "O_negative"]
    _DIST = {
        ("Hospital_A","Hospital_B"):18, ("Hospital_B","Hospital_A"):18,
        ("Hospital_A","Hospital_C"):44, ("Hospital_C","Hospital_A"):44,
        ("Hospital_B","Hospital_C"):31, ("Hospital_C","Hospital_B"):31,
    }
    _TRANSIT = {
        ("Hospital_A","Hospital_B"):0.5, ("Hospital_B","Hospital_A"):0.5,
        ("Hospital_A","Hospital_C"):1.2, ("Hospital_C","Hospital_A"):1.2,
        ("Hospital_B","Hospital_C"):0.9, ("Hospital_C","Hospital_B"):0.9,
    }
    _DAILY_DMD = {
        ("Hospital_A","O_positive"):45, ("Hospital_A","O_negative"):18,
        ("Hospital_B","O_positive"):28, ("Hospital_B","O_negative"):11,
        ("Hospital_C","O_positive"):14, ("Hospital_C","O_negative"): 6,
    }
    _HOUR_WT = {
        0:0.3, 1:0.2, 2:0.2, 3:0.2, 4:0.2, 5:0.4,
        6:0.8, 7:1.2, 8:1.8, 9:2.1, 10:1.9, 11:1.7,
        12:1.5, 13:1.7, 14:1.9, 15:2.0, 16:1.8, 17:1.4,
        18:1.2, 19:1.0, 20:0.8, 21:0.7, 22:0.5, 23:0.4,
    }
    _HOUR_WT_SUM = sum(_HOUR_WT.values())
    _DONOR_THR  = 7.0
    _RECIP_THR  = 4.0
    _MIN_XFER   = 5
    _MAX_DIST   = 50
    _CENTRAL_TRIGGER = 2.0
    _CENTRAL_REFILL  = 5.0
    _EXPIRY_RATE = 1.0 / (35.0 * 24.0)
    _INIT_STOCK = {
        ("Hospital_A","O_positive"):495, ("Hospital_A","O_negative"):198,
        ("Hospital_B","O_positive"):168, ("Hospital_B","O_negative"): 66,
        ("Hospital_C","O_positive"): 56, ("Hospital_C","O_negative"): 15,
    }
    _SIM_START = pd.Timestamp("2026-01-01 06:00:00")

    def _sim_reset():
        st.session_state.sim_running      = False
        st.session_state.sim_time         = _SIM_START
        st.session_state.sim_inventory    = {k: float(v) for k, v in _INIT_STOCK.items()}
        st.session_state.sim_log          = []
        st.session_state.sim_history      = []
        st.session_state.sim_last_tick    = None
        st.session_state.sim_total_shortage  = {k: 0.0 for k in _INIT_STOCK}
        st.session_state.sim_total_expired   = {k: 0.0 for k in _INIT_STOCK}
        st.session_state.sim_total_collected = {k: 0.0 for k in _INIT_STOCK}
        st.session_state.sim_accuracy_log = []   # [{date,hospital,blood_group,actual,forecast,ape}]
        st.session_state.sim_day_acc      = {}   # (h,bg) -> {date_str -> accumulated actual demand}
        st.session_state.sim_day_fc       = {}   # (h,bg) -> {date_str -> accumulated expected demand}

    if "sim_running" not in st.session_state:
        _sim_reset()
    if "sim_accuracy_log" not in st.session_state:
        st.session_state.sim_accuracy_log = []
    if "sim_day_acc" not in st.session_state:
        st.session_state.sim_day_acc = {}
    if "sim_day_fc" not in st.session_state:
        st.session_state.sim_day_fc = {}

    cc1, cc2, cc3 = st.columns([1, 1, 4])
    with cc1:
        if st.session_state.sim_running:
            if st.button("⏸ Pause", type="secondary", key="sim_pause"):
                st.session_state.sim_running = False; st.rerun()
        else:
            if st.button("▶ Start", type="primary", key="sim_start"):
                st.session_state.sim_running   = True
                st.session_state.sim_last_tick = time.time(); st.rerun()
    with cc2:
        if st.button("↺ Reset", key="sim_reset"):
            _sim_reset(); st.rerun()
    with cc3:
        _speed_opts  = [1/60, 1/30, 1/12, 1/6, 1.0, 6.0, 24.0]
        _speed_labels = {
            1/60: "Real-time  (1 min = 1 hr)",  1/30: "2× faster  (30 sec = 1 hr)",
            1/12: "5× faster  (12 sec = 1 hr)", 1/6:  "10× faster (6 sec = 1 hr)",
            1.0:  "60× — 1 day per 24 min",     6.0:  "Fast — 6 sim-hrs/sec",
            24.0: "Max — 1 sim-day/sec",
        }
        sim_speed = st.select_slider("Simulation speed", options=_speed_opts, value=1/60,
            format_func=lambda v: _speed_labels.get(v, f"{v} hr/sec"), key="sim_speed",
            help="Default is real-time: 1 real minute = 1 simulation hour")
    st.divider()

    if st.session_state.sim_running or st.session_state.get("sim2_running", False):
        now_real     = time.time()
        last_tick    = st.session_state.sim_last_tick or st.session_state.get("sim2_last_tick") or now_real
        elapsed_real = min(now_real - last_tick, 5.0)
        sim_hrs      = elapsed_real * sim_speed
        if st.session_state.sim_running:
            st.session_state.sim_last_tick = now_real
        if st.session_state.get("sim2_running", False):
            st.session_state.sim2_last_tick = now_real

        tick_seed = int(now_real * 1000) % 2**32
        rng8 = np.random.default_rng(tick_seed)
        rng9 = np.random.default_rng(tick_seed)

        if st.session_state.sim_running:
            sim_now  = st.session_state.sim_time
            sim_next = sim_now + pd.Timedelta(hours=sim_hrs)
            is_weekend = sim_now.weekday() in (5, 6)
            is_monday  = sim_now.weekday() == 0
            is_dengue  = sim_now.month in (7, 8, 9, 10)

            for h in _SIM_H:
                for bg in _SIM_BG:
                    stock = st.session_state.sim_inventory[(h, bg)]
                    daily = _DAILY_DMD[(h, bg)]
                    day_adj = 0.70 if is_weekend else (1.20 if is_monday else 1.0)
                    if is_dengue and bg == "O_positive": day_adj *= 1.18
                    hr_w   = _HOUR_WT[int(sim_now.hour) % 24] / _HOUR_WT_SUM * 24 if sim_hrs < 1.0 else 1.0
                    mean_d = daily * day_adj * hr_w * sim_hrs / 24.0
                    demand = max(0.0, float(rng8.normal(mean_d, mean_d * 0.25)))
                    _d_key = (h, bg); _d_str = str(sim_now.date())
                    _day_acc = st.session_state.sim_day_acc
                    if _d_key not in _day_acc: _day_acc[_d_key] = {}
                    _day_acc[_d_key][_d_str] = _day_acc[_d_key].get(_d_str, 0.0) + demand
                    _day_fc = st.session_state.sim_day_fc
                    if _d_key not in _day_fc: _day_fc[_d_key] = {}
                    _day_fc[_d_key][_d_str] = _day_fc[_d_key].get(_d_str, 0.0) + mean_d
                    collect_rate = 1.15 if h == "Hospital_A" else 0.0
                    mean_c_base  = daily * day_adj * sim_hrs / 24.0
                    raw_c  = max(0.0, float(rng8.normal(mean_c_base, mean_c_base * 0.30)))
                    drive  = rng8.random()
                    bonus  = float(rng8.uniform(daily * 0.05, daily * 0.15))
                    collect = raw_c * collect_rate + (bonus if drive < 0.01 * sim_hrs and collect_rate > 0 else 0.0)
                    expiry = min(stock * _EXPIRY_RATE * sim_hrs, stock * 0.02)
                    used     = min(demand, stock)
                    shortage = max(0.0, demand - stock)
                    stock    = max(0.0, stock - used - expiry + collect)
                    st.session_state.sim_inventory[(h, bg)]       = round(min(stock, daily * 14), 1)
                    st.session_state.sim_total_shortage[(h, bg)]  += shortage
                    st.session_state.sim_total_expired[(h, bg)]   += expiry
                    st.session_state.sim_total_collected[(h, bg)] += collect

            for h in _SIM_H:
                for bg in _SIM_BG:
                    s = st.session_state.sim_inventory[(h, bg)]; d = _DAILY_DMD[(h, bg)]; dos = s / d
                    if dos < _CENTRAL_TRIGGER:
                        refill = d * _CENTRAL_REFILL
                        st.session_state.sim_inventory[(h, bg)] = min(s + refill, d * 14)
                        st.session_state.sim_log.append({
                            "sim_time": sim_next.strftime("%Y-%m-%d %H:%M"), "donor": "Central Bank",
                            "recipient": h, "blood_group": bg, "qty": int(refill),
                            "distance_km": 0, "transit_hr": 0, "donor_dos": 999, "recipient_dos": round(dos, 1), "auto": True,
                        })

            for h in _SIM_H:
                for bg in _SIM_BG:
                    st.session_state.sim_history.append({
                        "sim_time": sim_now, "hospital": h, "blood_group": bg,
                        "stock": st.session_state.sim_inventory[(h, bg)],
                    })
            if len(st.session_state.sim_history) > 10_000:
                st.session_state.sim_history = st.session_state.sim_history[::2]

            st.session_state.sim_time = sim_next

            if sim_now.date() != sim_next.date():
                _done_date = str(sim_now.date())
                for _h in _SIM_H:
                    for _bg in _SIM_BG:
                        _actual = st.session_state.sim_day_acc.get((_h, _bg), {}).get(_done_date, 0.0)
                        _fc     = st.session_state.sim_day_fc.get((_h, _bg), {}).get(_done_date,
                                  float(_DAILY_DMD[(_h, _bg)]))
                        _ape    = abs(_actual - _fc) / _fc * 100 if _fc > 0 else 0.0
                        st.session_state.sim_accuracy_log.append({
                            "date": _done_date, "hospital": _h, "blood_group": _bg,
                            "actual": round(_actual, 1), "forecast": round(_fc, 1), "ape": round(_ape, 1),
                        })

            recent_xfers = {(e["donor"], e["recipient"], e["blood_group"]) for e in st.session_state.sim_log[-6:]}
            for bg in _SIM_BG:
                inv   = {h: st.session_state.sim_inventory[(h, bg)] for h in _SIM_H}
                daily = {h: _DAILY_DMD[(h, bg)]                     for h in _SIM_H}
                dos   = {h: inv[h] / daily[h]                       for h in _SIM_H}
                recips = sorted([h for h in _SIM_H if dos[h] < _RECIP_THR], key=lambda x: dos[x])
                donors = sorted([h for h in _SIM_H if dos[h] > _DONOR_THR], key=lambda x: -dos[x])
                for recip in recips:
                    for donor in donors:
                        if donor == recip: continue
                        dist = _DIST.get((donor, recip))
                        if dist is None or dist > _MAX_DIST: continue
                        if (donor, recip, bg) in recent_xfers: continue
                        surplus = inv[donor] - (_DONOR_THR * daily[donor])
                        deficit = (_RECIP_THR * daily[recip]) - inv[recip]
                        qty     = int(min(surplus * 0.4, deficit))
                        if qty < _MIN_XFER: continue
                        new_donor_stock = max(0.0, inv[donor] - qty)
                        new_recip_stock = min(inv[recip] + qty, daily[recip] * 14)
                        st.session_state.sim_inventory[(donor, bg)] = round(new_donor_stock, 1)
                        st.session_state.sim_inventory[(recip, bg)] = round(new_recip_stock, 1)
                        inv[donor] = new_donor_stock; inv[recip] = new_recip_stock
                        dos[donor] = new_donor_stock / daily[donor]; dos[recip] = new_recip_stock / daily[recip]
                        st.session_state.sim_log.append({
                            "sim_time": sim_next.strftime("%Y-%m-%d %H:%M"), "donor": donor,
                            "recipient": recip, "blood_group": bg, "qty": qty,
                            "distance_km": dist, "transit_hr": _TRANSIT[(donor, recip)],
                            "donor_dos": round(dos[donor], 1), "recipient_dos": round(dos[recip], 1), "auto": True,
                        })
                        break

        if st.session_state.get("sim2_running", False):
            sim2_now  = st.session_state.sim2_time
            sim2_next = sim2_now + pd.Timedelta(hours=sim_hrs)
            is_weekend2 = sim2_now.weekday() in (5, 6)
            is_monday2  = sim2_now.weekday() == 0
            is_dengue2  = sim2_now.month in (7, 8, 9, 10)

            for h in _SIM_H:
                for bg in _SIM_BG:
                    stock = st.session_state.sim2_inventory[(h, bg)]
                    daily = _DAILY_DMD[(h, bg)]
                    day_adj = 0.70 if is_weekend2 else (1.20 if is_monday2 else 1.0)
                    if is_dengue2 and bg == "O_positive": day_adj *= 1.18
                    hr_w   = _HOUR_WT[int(sim2_now.hour) % 24] / _HOUR_WT_SUM * 24 if sim_hrs < 1.0 else 1.0
                    mean_d = daily * day_adj * hr_w * sim_hrs / 24.0
                    demand = max(0.0, float(rng9.normal(mean_d, mean_d * 0.25)))
                    collect_rate = 1.15 if h == "Hospital_A" else 0.0
                    mean_c_base  = daily * day_adj * sim_hrs / 24.0
                    raw_c  = max(0.0, float(rng9.normal(mean_c_base, mean_c_base * 0.30)))
                    _drive = rng9.random()
                    _bonus = float(rng9.uniform(daily * 0.05, daily * 0.15))
                    collect = raw_c * collect_rate + (_bonus if _drive < 0.01 * sim_hrs and collect_rate > 0 else 0.0)
                    expiry = min(stock * _EXPIRY_RATE * sim_hrs, stock * 0.02)
                    used     = min(demand, stock)
                    shortage = max(0.0, demand - stock)
                    stock    = max(0.0, stock - used - expiry + collect)
                    st.session_state.sim2_inventory[(h, bg)]       = round(min(stock, daily * 14), 1)
                    st.session_state.sim2_total_shortage[(h, bg)]  += shortage
                    st.session_state.sim2_total_expired[(h, bg)]   += expiry
                    st.session_state.sim2_total_collected[(h, bg)] += collect

            for h in _SIM_H:
                for bg in _SIM_BG:
                    s = st.session_state.sim2_inventory[(h, bg)]; d = _DAILY_DMD[(h, bg)]; dos = s / d
                    if dos < _CENTRAL_TRIGGER:
                        refill = d * _CENTRAL_REFILL
                        st.session_state.sim2_inventory[(h, bg)] = min(s + refill, d * 14)
                        st.session_state.sim2_log.append({
                            "sim_time": sim2_next.strftime("%Y-%m-%d %H:%M"), "donor": "Central Bank",
                            "recipient": h, "blood_group": bg, "qty": int(refill), "recipient_dos": round(dos, 1),
                        })

            for h in _SIM_H:
                for bg in _SIM_BG:
                    st.session_state.sim2_history.append({
                        "sim_time": sim2_now, "hospital": h, "blood_group": bg,
                        "stock": st.session_state.sim2_inventory[(h, bg)],
                    })
            if len(st.session_state.sim2_history) > 10_000:
                st.session_state.sim2_history = st.session_state.sim2_history[::2]
            st.session_state.sim2_time = sim2_next

    sim_label = "● RUNNING" if st.session_state.sim_running else "❚❚ PAUSED"
    sim_clr   = "#27ae60"   if st.session_state.sim_running else "#f39c12"
    elapsed_sim_days = (st.session_state.sim_time - _SIM_START).total_seconds() / 86400
    st.markdown(
        f"<div style='display:flex;align-items:center;gap:24px;padding:8px 0;'>"
        f"<span style='background:{sim_clr};color:white;padding:4px 16px;border-radius:12px;font-weight:bold;'>{sim_label}</span>"
        f"<span style='font-size:20px;font-weight:bold;'>📅 {st.session_state.sim_time.strftime('%Y-%m-%d  %H:%M')}</span>"
        f"<span style='color:gray;font-size:12px;'>Sim elapsed: {elapsed_sim_days:.1f} days &nbsp;|&nbsp; "
        f"Auto-transfers: {len(st.session_state.sim_log)}</span></div>", unsafe_allow_html=True)

    st.subheader("Live Inventory")
    gcols = st.columns(3)
    for col, h in zip(gcols, _SIM_H):
        with col:
            st.markdown(f"**{h.replace('Hospital_', 'Hospital ')}**")
            for bg in _SIM_BG:
                stock = st.session_state.sim_inventory[(h, bg)]; ddmnd = _DAILY_DMD[(h, bg)]
                dos = stock / ddmnd; max_cap = ddmnd * 14; pct = min(100.0, stock / max_cap * 100)
                clr = "#e74c3c" if dos < 2 else ("#f39c12" if dos < 4 else "#27ae60")
                status = "CRITICAL" if dos < 2 else ("LOW" if dos < 4 else "OK")
                st.markdown(
                    f"<div style='margin-bottom:12px;'>"
                    f"<div style='display:flex;justify-content:space-between;font-size:12px;margin-bottom:3px;'>"
                    f"<b>{bg.replace('_',' ')}</b>"
                    f"<span style='color:{clr};font-weight:bold;'>{stock:.0f} units &nbsp;|&nbsp; {dos:.1f} days &nbsp;"
                    f"<span style='background:{clr};color:white;border-radius:4px;padding:1px 5px;font-size:10px;'>{status}</span></span></div>"
                    f"<div style='background:#e0e0e0;border-radius:6px;height:18px;'>"
                    f"<div style='background:{clr};width:{pct:.1f}%;height:18px;border-radius:6px;'></div>"
                    f"</div></div>", unsafe_allow_html=True)
            tot_short   = sum(st.session_state.sim_total_shortage.get((h, bg), 0)   for bg in _SIM_BG)
            tot_expired = sum(st.session_state.sim_total_expired.get((h, bg), 0)    for bg in _SIM_BG)
            tot_collect = sum(st.session_state.sim_total_collected.get((h, bg), 0)  for bg in _SIM_BG)
            st.caption(f"Collected: {tot_collect:.0f} u &nbsp;·&nbsp; Expired: {tot_expired:.0f} u &nbsp;·&nbsp; Shortage: {tot_short:.0f} u")

    if len(st.session_state.sim_history) > 10:
        st.subheader("Inventory Over Simulation Time")
        hdf = pd.DataFrame(st.session_state.sim_history)
        hdf["label"] = hdf["hospital"].str.replace("Hospital_","H.") + " " + hdf["blood_group"].str.replace("_"," ")
        fig_s = go.Figure()
        colors_map = {"H.A O positive":"#e74c3c","H.A O negative":"#c0392b","H.B O positive":"#3498db",
                      "H.B O negative":"#2471a3","H.C O positive":"#27ae60","H.C O negative":"#1e8449"}
        for lbl in sorted(hdf["label"].unique()):
            sub = hdf[hdf["label"]==lbl].tail(2000)
            fig_s.add_trace(go.Scatter(x=sub["sim_time"],y=sub["stock"],mode="lines",name=lbl,
                line=dict(width=1.8,color=colors_map.get(lbl))))
        for (h_ref, bg_ref), dash, col_, pos in [
            (("Hospital_A","O_positive"),"dash","#e74c3c","top right"),
            (("Hospital_A","O_negative"),"dot","#c0392b","top right"),
        ]:
            trig_lvl = _DAILY_DMD[(h_ref, bg_ref)] * _RECIP_THR
            fig_s.add_hline(y=trig_lvl,line_dash=dash,line_color=col_,opacity=0.35,
                annotation_text=f"{bg_ref.replace('_',' ')} trigger ({trig_lvl:.0f} u)",annotation_position=pos)
        fig_s.update_layout(height=340,hovermode="x unified",xaxis_title="Simulation Time",yaxis_title="Stock (units)",
            legend=dict(orientation="h",yanchor="bottom",y=1.08),margin=dict(l=0,r=0,t=75,b=0))
        fig_s = _apply_white_theme(fig_s)
        st.plotly_chart(fig_s, width="stretch")

    st.subheader(f"Auto-Transfer Events  ({len(st.session_state.sim_log)} total — no approval needed)")
    if st.session_state.sim_log:
        for ev in reversed(st.session_state.sim_log[-15:]):
            is_central = ev["donor"] == "Central Bank"
            bg_clr = "#1a1a2e" if is_central else ("#fde8e8" if ev["blood_group"]=="O_negative" else "#eafaf1")
            border = "#f39c12" if is_central else ("#c0392b" if ev["blood_group"]=="O_negative" else "#27ae60")
            icon   = "🏦" if is_central else "🕐"
            route  = (f"Central Bank ➜ <b>{ev['recipient'].replace('Hospital_','H.')}</b> &nbsp;·&nbsp; <span style='color:#f39c12;'>EMERGENCY REFILL</span>"
                      if is_central else
                      f"<b>{ev['donor'].replace('Hospital_','H.')}</b> ➜ <b>{ev['recipient'].replace('Hospital_','H.')}</b>"
                      f" &nbsp;·&nbsp; {ev['distance_km']} km / {ev['transit_hr']} hr"
                      f" &nbsp;·&nbsp; Donor {ev['donor_dos']}d ➜ Recip {ev['recipient_dos']}d")
            st.markdown(
                f"<div style='background:{bg_clr};padding:6px 14px;border-radius:6px;margin-bottom:4px;"
                f"font-size:13px;border-left:4px solid {border};'>"
                f"{icon} <b>{ev['sim_time']}</b> &nbsp;·&nbsp; {route} &nbsp;·&nbsp; "
                f"<b>{ev['blood_group'].replace('_',' ')}</b> &nbsp;·&nbsp; <b>{ev['qty']} units</b>"
                f"</div>", unsafe_allow_html=True)
    else:
        st.info("No transfers yet. Hospital_C O- starts critically low — a transfer should fire within the first sim-day.")

with tabs[4]:
    _tab8_simulation()

# =============================================================================
# TAB 9 — BASELINE SIMULATION  (unchanged from app2.py)
# =============================================================================
@st.fragment(run_every=1)
def _tab9_simulation():
    st.subheader("Baseline Simulation — No Inter-Hospital Transfers")
    st.caption(
        "Same demand / collection / expiry as Tab 8, but hospitals work "
        "**in isolation** — no redistribution between them. Central Bank "
        "emergency refills still apply (fair baseline). "
        "Use the comparison panel below to see extra expiry & shortage vs Tab 8."
    )
    _B_SIM_H   = ["Hospital_A", "Hospital_B", "Hospital_C"]
    _B_SIM_BG  = ["O_positive", "O_negative"]
    _B_DAILY_DMD = {
        ("Hospital_A","O_positive"):45, ("Hospital_A","O_negative"):18,
        ("Hospital_B","O_positive"):28, ("Hospital_B","O_negative"):11,
        ("Hospital_C","O_positive"):14, ("Hospital_C","O_negative"): 6,
    }
    _B_COLLECT_RATE = {"Hospital_A": 1.15, "Hospital_B": 0.0, "Hospital_C": 0.0}
    _B_HOUR_WT = {
        0:0.3, 1:0.2, 2:0.2, 3:0.2, 4:0.2, 5:0.4,
        6:0.8, 7:1.2, 8:1.8, 9:2.1, 10:1.9, 11:1.7,
        12:1.5, 13:1.7, 14:1.9, 15:2.0, 16:1.8, 17:1.4,
        18:1.2, 19:1.0, 20:0.8, 21:0.7, 22:0.5, 23:0.4,
    }
    _B_HOUR_WT_SUM  = sum(_B_HOUR_WT.values())
    _B_CENTRAL_TRIGGER = 2.0; _B_CENTRAL_REFILL = 5.0
    _B_EXPIRY_RATE  = 1.0 / (35.0 * 24.0)
    _B_INIT_STOCK   = {
        ("Hospital_A","O_positive"):495, ("Hospital_A","O_negative"):198,
        ("Hospital_B","O_positive"):168, ("Hospital_B","O_negative"): 66,
        ("Hospital_C","O_positive"): 56, ("Hospital_C","O_negative"): 15,
    }
    _B_SIM_START = pd.Timestamp("2026-01-01 06:00:00")

    def _sim2_reset():
        st.session_state.sim2_running      = False
        st.session_state.sim2_time         = _B_SIM_START
        st.session_state.sim2_inventory    = {k: float(v) for k, v in _B_INIT_STOCK.items()}
        st.session_state.sim2_log          = []
        st.session_state.sim2_history      = []
        st.session_state.sim2_last_tick    = None
        st.session_state.sim2_total_shortage  = {k: 0.0 for k in _B_INIT_STOCK}
        st.session_state.sim2_total_expired   = {k: 0.0 for k in _B_INIT_STOCK}
        st.session_state.sim2_total_collected = {k: 0.0 for k in _B_INIT_STOCK}

    if "sim2_running" not in st.session_state:
        _sim2_reset()

    bc1, bc2, bc3 = st.columns([1, 1, 4])
    with bc1:
        if st.session_state.sim2_running:
            if st.button("⏸ Pause", type="secondary", key="sim2_pause"):
                st.session_state.sim2_running = False; st.rerun()
        else:
            if st.button("▶ Start", type="primary", key="sim2_start"):
                st.session_state.sim2_running   = True
                st.session_state.sim2_last_tick = time.time(); st.rerun()
    with bc2:
        if st.button("↺ Reset", key="sim2_reset"):
            _sim2_reset(); st.rerun()
    with bc3:
        st.info("⚡ Speed is shared with Tab 8 — adjust the slider there. Both sims advance in lockstep for a controlled comparison.")
    st.divider()

    sim2_label = "● RUNNING" if st.session_state.sim2_running else "❚❚ PAUSED"
    sim2_clr   = "#27ae60"    if st.session_state.sim2_running else "#f39c12"
    elapsed_sim_days = (st.session_state.sim2_time - _B_SIM_START).total_seconds() / 86400
    n_central = sum(1 for e in st.session_state.sim2_log if e["donor"] == "Central Bank")
    st.markdown(
        f"<div style='display:flex;align-items:center;gap:24px;padding:8px 0;'>"
        f"<span style='background:{sim2_clr};color:white;padding:4px 16px;border-radius:12px;font-weight:bold;'>{sim2_label}</span>"
        f"<span style='font-size:20px;font-weight:bold;'>📅 {st.session_state.sim2_time.strftime('%Y-%m-%d  %H:%M')}</span>"
        f"<span style='color:gray;font-size:12px;'>Sim elapsed: {elapsed_sim_days:.1f} days &nbsp;|&nbsp; "
        f"Inter-hospital transfers: <b style='color:#c0392b;'>0</b> &nbsp;|&nbsp; Central Bank refills: {n_central}"
        f"</span></div>", unsafe_allow_html=True)

    st.subheader("Live Inventory")
    gcols = st.columns(3)
    for col, h in zip(gcols, _B_SIM_H):
        with col:
            role = "HUB" if _B_COLLECT_RATE[h] > 0 else "CONSUMER"
            role_clr = "#2471a3" if _B_COLLECT_RATE[h] > 0 else "#7d3c98"
            st.markdown(
                f"<div style='display:flex;align-items:center;gap:6px;margin-bottom:6px;'>"
                f"<b>{h.replace('Hospital_','H.')}</b>"
                f"<span style='background:{role_clr};color:white;font-size:9px;padding:1px 6px;"
                f"border-radius:8px;'>{role}</span></div>", unsafe_allow_html=True)
            for bg in _B_SIM_BG:
                stock = st.session_state.sim2_inventory.get((h, bg), 0.0)
                ddmnd = _B_DAILY_DMD[(h, bg)]
                dos = stock / ddmnd; pct = min(100.0, stock / (ddmnd * 14) * 100)
                clr = "#e74c3c" if dos < 2 else ("#f39c12" if dos < 4 else "#27ae60")
                status = "CRITICAL" if dos < 2 else ("LOW" if dos < 4 else "OK")
                st.markdown(
                    f"<div style='margin-bottom:10px;'>"
                    f"<div style='display:flex;justify-content:space-between;font-size:11px;margin-bottom:2px;'>"
                    f"<b>{bg.replace('_',' ')}</b>"
                    f"<span style='color:{clr};font-weight:bold;'>{stock:.0f}u &nbsp;{dos:.1f}d &nbsp;"
                    f"<span style='background:{clr};color:white;border-radius:3px;padding:1px 4px;font-size:9px;'>{status}</span>"
                    f"</span></div>"
                    f"<div style='background:#e0e0e0;border-radius:4px;height:14px;'>"
                    f"<div style='background:{clr};width:{pct:.1f}%;height:14px;border-radius:4px;'></div>"
                    f"</div></div>", unsafe_allow_html=True)
            tot_short   = sum(st.session_state.sim2_total_shortage.get((h, bg), 0)   for bg in _B_SIM_BG)
            tot_expired = sum(st.session_state.sim2_total_expired.get((h, bg), 0)    for bg in _B_SIM_BG)
            tot_collect = sum(st.session_state.sim2_total_collected.get((h, bg), 0)  for bg in _B_SIM_BG)
            st.caption(f"Col: {tot_collect:.0f} &nbsp;Exp: {tot_expired:.0f} &nbsp;Short: {tot_short:.0f}")

    if len(st.session_state.sim2_history) > 10:
        st.subheader("Inventory Over Simulation Time (Baseline)")
        hdf = pd.DataFrame(st.session_state.sim2_history)
        hdf["label"] = hdf["hospital"].str.replace("Hospital_","H.") + " " + hdf["blood_group"].str.replace("_"," ")
        fig_b = go.Figure()
        colors_map = {
            "H.A O positive":"#e74c3c","H.A O negative":"#c0392b",
            "H.B O positive":"#3498db","H.B O negative":"#2471a3",
            "H.C O positive":"#27ae60","H.C O negative":"#1e8449",
        }
        for lbl in sorted(hdf["label"].unique()):
            sub = hdf[hdf["label"]==lbl].tail(2000)
            fig_b.add_trace(go.Scatter(x=sub["sim_time"],y=sub["stock"],mode="lines",name=lbl,
                line=dict(width=1.8,color=colors_map.get(lbl,"#888"))))
        fig_b.update_layout(height=340,hovermode="x unified",xaxis_title="Simulation Time",yaxis_title="Stock (units)",
            legend=dict(orientation="h",yanchor="bottom",y=1.08),margin=dict(l=0,r=0,t=75,b=0))
        fig_b = _apply_white_theme(fig_b)
        st.plotly_chart(fig_b, width="stretch")

    st.divider()
    st.subheader("📊 Comparison — ML Redistribution vs Isolated Baseline")
    ml_exp   = sum(st.session_state.get("sim_total_expired",   {}).values())
    ml_sh    = sum(st.session_state.get("sim_total_shortage",  {}).values())
    ml_col   = sum(st.session_state.get("sim_total_collected", {}).values())
    ml_xfers = len([e for e in st.session_state.get("sim_log", []) if e.get("donor") != "Central Bank"])
    ml_cb    = len([e for e in st.session_state.get("sim_log", []) if e.get("donor") == "Central Bank"])
    b_exp    = sum(st.session_state.sim2_total_expired.values())
    b_sh     = sum(st.session_state.sim2_total_shortage.values())
    b_col    = sum(st.session_state.sim2_total_collected.values())
    b_cb     = len(st.session_state.sim2_log)

    cmp_cols = st.columns(2)
    with cmp_cols[0]:
        st.markdown("**🔄 Tab 8 — ML Redistribution**")
        st.metric("Expired", f"{ml_exp:.0f} u"); st.metric("Shortage", f"{ml_sh:.0f} u")
        st.metric("Collected", f"{ml_col:.0f} u"); st.metric("Inter-hosp transfers", ml_xfers)
        st.metric("Central Bank refills", ml_cb)
    with cmp_cols[1]:
        st.markdown("**🚫 Tab 9 — Baseline (no transfers)**")
        st.metric("Expired",  f"{b_exp:.0f} u", f"{b_exp-ml_exp:+.0f} u vs ML", delta_color="inverse")
        st.metric("Shortage", f"{b_sh:.0f} u",  f"{b_sh-ml_sh:+.0f} u vs ML",  delta_color="inverse")
        st.metric("Collected", f"{b_col:.0f} u"); st.metric("Inter-hosp transfers", 0)
        st.metric("Central Bank refills", b_cb, f"{b_cb-ml_cb:+d} vs ML", delta_color="inverse")

    delta_exp = b_exp - ml_exp; delta_sh = b_sh - ml_sh
    if delta_exp > 0 or delta_sh > 0:
        st.success(f"✅ ML redistribution saved **{max(0,delta_exp):.0f} expired units** "
                   f"and prevented **{max(0,delta_sh):.0f} shortage units** compared to the isolated baseline.")
    else:
        st.info("Run both Tab 8 and Tab 9 side-by-side to see the comparison delta.")

with tabs[5]:
    _tab9_simulation()

# end of app
