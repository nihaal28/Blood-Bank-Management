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

st.set_page_config(page_title="Blood Inventory System", layout="wide", initial_sidebar_state="expanded")

HOSPITALS    = ["Hospital_A","Hospital_B","Hospital_C"]
BLOOD_GROUPS = ["O_positive","O_negative"]
COMBOS       = [(h,bg) for h in HOSPITALS for bg in BLOOD_GROUPS]

def _dos_color(dos):
    return "#e74c3c" if dos<2 else ("#f39c12" if dos<4 else "#27ae60")

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

@st.cache_data(show_spinner="Loading dataset...")
def load_data():
    df    = pd.read_csv(os.path.join(DATA_DIR,"raw","synthetic_demand.csv"), parse_dates=["date"])
    graph = pd.read_csv(os.path.join(DATA_DIR,"raw","hospital_graph.csv"))
    return df, graph

@st.cache_resource(show_spinner="Loading forecasting models...")
def load_forecasters():
    fcs = {}
    for h,bg in COMBOS:
        hf = HybridForecaster(h,bg); hf.load_models(); hf.optimise_weights(load_data()[0])
        fcs[(h,bg)] = hf
    return fcs

@st.cache_resource(show_spinner="Initialising redistribution engine...")
def load_engine(): return RedistributionEngine()

_ensure_log()
df, graph_df = load_data()
forecasters  = load_forecasters()
engine       = load_engine()
MIN_DATE, MAX_DATE = df["date"].min().date(), df["date"].max().date()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/blood-bag.png", width=60)
    st.title("Blood Inventory\nManagement System"); st.divider()
    selected_date = pd.Timestamp(st.date_input("Reference Date", value=MAX_DATE, min_value=MIN_DATE, max_value=MAX_DATE))
    hosp_filter = st.selectbox("Hospital Filter",    ["All"]+HOSPITALS)
    bg_filter   = st.selectbox("Blood Group Filter", ["All"]+BLOOD_GROUPS)
    st.divider(); st.caption(f"Data: {MIN_DATE} to {MAX_DATE}"); st.caption("Phase 1 — Synthetic data")

st.markdown("<div style='background:#c0392b;padding:10px 20px;border-radius:6px;margin-bottom:12px;'>"
            "<span style='color:white;font-size:16px;font-weight:bold;'>&#9888; All transfer decisions require human approval</span></div>",
            unsafe_allow_html=True)

tabs = st.tabs(["Overview","Inventory","Forecast","Transfers","History","Metrics","MLOps & Drift","Live Simulation"])

# =============================================================================
# TAB 1 — OVERVIEW
# =============================================================================
with tabs[0]:
    st.subheader("Network Overview")
    snap = df[df["date"]<=selected_date].groupby(["hospital","blood_group"]).last().reset_index()
    opos = snap[snap["blood_group"]=="O_positive"]["stock_on_hand"].sum()
    oneg = snap[snap["blood_group"]=="O_negative"]["stock_on_hand"].sum()
    exp  = snap["units_expiring_soon"].sum()
    c1,c2,c3 = st.columns(3)
    c1.metric("Total O+ Stock (network)", f"{int(opos):,} units")
    c2.metric("Total O- Stock (network)", f"{int(oneg):,} units")
    c3.metric("Units Expiring Soon (<=3d)", f"{int(exp):,} units",
              delta=None if exp==0 else "Action needed", delta_color="inverse")
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
        hs = snap[snap["hospital"]==h]
        ts = int(hs["stock_on_hand"].sum()); md = df[df["hospital"]==h]["demand"].mean()
        dos = ts/md if md>0 else 0
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
    st.plotly_chart(fig, width='stretch')
    st.subheader("Hospital Summary")
    summ_df = pd.DataFrame(summary_rows)
    st.dataframe(summ_df.style.apply(lambda col:[
        f"background-color:{'#fadbd8' if v=='CRITICAL' else '#fef9e7' if v=='LOW' else '#d5f5e3'}"
        for v in col] if col.name=="Status" else [""]*len(col),axis=0),width='stretch',hide_index=True)

# =============================================================================
# TAB 2 — INVENTORY
# =============================================================================
with tabs[1]:
    st.subheader("Inventory Status")
    snap2 = df[df["date"]<=selected_date].groupby(["hospital","blood_group"]).last().reset_index()
    inv_rows = []
    for _,r in snap2.iterrows():
        h,bg = r["hospital"],r["blood_group"]
        if hosp_filter!="All" and h!=hosp_filter: continue
        if bg_filter!="All" and bg!=bg_filter: continue
        md = df[(df["hospital"]==h)&(df["blood_group"]==bg)]["demand"].mean()
        dos = r["stock_on_hand"]/md if md>0 else 0
        inv_rows.append({"Hospital":h,"Blood Group":bg,"Stock on Hand":int(r["stock_on_hand"]),
            "Expiring Soon":int(r["units_expiring_soon"]),"Days of Supply":round(dos,1),
            "Mean Daily Demand":round(md,1),"Status":"CRITICAL" if dos<2 else ("LOW" if dos<4 else "HEALTHY")})
    inv_df = pd.DataFrame(inv_rows)
    if not inv_df.empty:
        def _cs(v):
            if v=="CRITICAL": return "background-color:#fadbd8;color:#c0392b;font-weight:bold"
            if v=="LOW":      return "background-color:#fef9e7;color:#d35400;font-weight:bold"
            return "background-color:#d5f5e3;color:#1e8449"
        st.dataframe(inv_df.style.applymap(_cs, subset=["Status"]), width='stretch', hide_index=True)
        fig_inv = px.bar(inv_df,x="Hospital",y="Stock on Hand",color="Blood Group",barmode="group",
            color_discrete_map={"O_positive":"#e74c3c","O_negative":"#3498db"},
            title=f"Stock Levels by Hospital and Blood Group  (as of {selected_date.date()})")
        fig_inv.update_layout(height=350); st.plotly_chart(fig_inv, width='stretch')
    else:
        st.info("No data for selected filters.")

# =============================================================================
# TAB 3 — FORECAST
# =============================================================================
with tabs[2]:
    st.subheader("7-Day Demand Forecast")
    fc_hosp = st.selectbox("Hospital", HOSPITALS, key="fc_hosp")
    fc_bg   = st.selectbox("Blood Group", BLOOD_GROUPS, key="fc_bg")
    sub_fc  = df[(df["hospital"]==fc_hosp)&(df["blood_group"]==fc_bg)].sort_values("date")
    last14  = sub_fc[sub_fc["date"]<=selected_date].tail(14)
    hf = forecasters[(fc_hosp,fc_bg)]
    hf._prophet._last_date = selected_date
    pfc = hf._prophet.predict(periods=7)
    lfc = hf._lstm.predict(sub_fc[sub_fc["date"]<=selected_date].tail(30))
    hfc = hf.predict(df, target_date=selected_date)

    fig_fc = go.Figure()
    fig_fc.add_trace(go.Scatter(x=last14["date"],y=last14["demand"],mode="lines+markers",
        name="Actual (14d)",line=dict(color="#2980b9",width=2)))
    fig_fc.add_trace(go.Scatter(x=pfc["ds"],y=pfc["yhat"],mode="lines+markers",
        name="Prophet",line=dict(color="#8e44ad",width=1.5,dash="dash")))
    fig_fc.add_trace(go.Scatter(x=hfc["date"],y=hfc["lstm_pred"],mode="lines+markers",
        name="LSTM",line=dict(color="#e67e22",width=1.5,dash="dash")))
    fig_fc.add_trace(go.Scatter(x=hfc["date"],y=hfc["hybrid_pred"],mode="lines+markers",
        name=f"Hybrid (a={hf.alpha:.2f})",line=dict(color="#27ae60",width=3)))
    fig_fc.add_vline(x=selected_date.timestamp()*1000,line_width=1.5,line_dash="dot",
        line_color="gray",annotation_text="Today",annotation_position="top left")
    fig_fc.add_vrect(x0=hfc["date"].min(),x1=hfc["date"].max(),fillcolor="#27ae60",opacity=0.05,
        annotation_text="7-day window",annotation_position="top left")
    fig_fc.update_layout(title=f"Demand Forecast — {fc_hosp}  |  {fc_bg.replace('_',' ')}",
        xaxis_title="Date",yaxis_title="Demand (units)",height=420,hovermode="x unified")
    st.plotly_chart(fig_fc, width='stretch')

    st.subheader("Day-by-Day Forecast Values")
    fc_tbl = hfc.copy(); fc_tbl["Prophet"] = pfc["yhat"].values; fc_tbl["LSTM"] = lfc
    fc_tbl["Day"] = [d.strftime("%a %d %b") for d in fc_tbl["date"]]
    fc_tbl = fc_tbl[["Day","Prophet","LSTM","hybrid_pred"]].rename(columns={"hybrid_pred":"Hybrid"})
    fc_tbl[["Prophet","LSTM","Hybrid"]] = fc_tbl[["Prophet","LSTM","Hybrid"]].round(1)
    st.dataframe(fc_tbl, width='stretch', hide_index=True)

# =============================================================================
# TAB 4 — TRANSFERS
# =============================================================================
with tabs[3]:
    st.subheader("Transfer Recommendations")
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

# =============================================================================
# TAB 5 — HISTORY
# =============================================================================
with tabs[4]:
    st.subheader("Transfer Decision History")
    _ensure_log(); log_df = pd.read_csv(LOG_PATH)
    if log_df.empty:
        st.info("No transfer decisions logged yet. Use the Transfers tab to approve or reject.")
    else:
        ta = (log_df["decision"]=="APPROVED").sum(); tr = (log_df["decision"]=="REJECTED").sum()
        h1,h2,h3 = st.columns(3)
        h1.metric("Total Approved", int(ta)); h2.metric("Total Rejected", int(tr))
        h3.metric("Approval Rate", f"{100*ta/len(log_df):.1f}%")
        st.dataframe(log_df, width='stretch', hide_index=True)
        if len(log_df) > 1:
            log_df["date"] = pd.to_datetime(log_df["date"])
            log_df["week"] = log_df["date"].dt.to_period("W").astype(str)
            weekly = log_df.groupby(["week","decision"]).size().reset_index(name="count")
            fig_h = px.bar(weekly,x="week",y="count",color="decision",barmode="group",
                color_discrete_map={"APPROVED":"#27ae60","REJECTED":"#e74c3c"},
                title="Approvals vs Rejections by Week")
            fig_h.update_layout(height=320); st.plotly_chart(fig_h, width='stretch')

# =============================================================================
# TAB 6 — METRICS
# =============================================================================
with tabs[5]:
    st.subheader("Model Accuracy Metrics")
    def _load_m(name):
        p = os.path.join(REPORTS_DIR,name); return pd.read_csv(p) if os.path.exists(p) else pd.DataFrame()
    compare_m = _load_m("model_comparison.csv")
    if not compare_m.empty:
        hr = compare_m[compare_m["model"]=="Hybrid"]
        mc1,mc2,mc3 = st.columns(3)
        mc1.metric("Best Hybrid MAE",  f"{hr['MAE'].min():.2f} units")
        mc2.metric("Best Hybrid RMSE", f"{hr['RMSE'].min():.2f} units")
        mc3.metric("Best Hybrid MAPE", f"{hr['MAPE'].min():.2f}%"); st.divider()
        fig_m = px.bar(compare_m,x=compare_m["hospital"]+" "+compare_m["blood_group"],y="MAE",
            color="model",barmode="group",
            color_discrete_map={"Prophet":"#8e44ad","LSTM":"#e67e22","Hybrid":"#27ae60","MovingAverage":"#95a5a6"},
            title="MAE Comparison — All Models and Hospital/Blood-Group Combinations")
        fig_m.update_layout(height=380,xaxis_title="Hospital + Blood Group"); st.plotly_chart(fig_m, width='stretch')
        st.subheader("Full Comparison Table"); st.dataframe(compare_m, width='stretch', hide_index=True)
    else:
        st.warning("Run src/hybrid_ensemble.py first to generate model_comparison.csv")

# =============================================================================
# TAB 7 — MLOps & Drift
# =============================================================================
with tabs[6]:
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
            st.caption("No runs yet. Click 'Trigger Retraining Now' or run `python src/mlops_monitor.py --simulate-check`")
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
            fig_d.update_layout(height=380,hovermode="x unified"); st.plotly_chart(fig_d, width='stretch')
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
        st.info("No monitoring history yet. Run the monitor to populate this view.")
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
            st.info("No route adjustments yet (insufficient transfer log data).")
    else:
        st.info("No route weights file found. Run the MLOps monitor to generate it.")
    st.divider()

    st.subheader("Latest Health Report")
    if os.path.exists(HEALTH_TXT):
        with open(HEALTH_TXT, encoding="utf-8") as f: st.code(f.read(), language=None)
    else:
        st.info("No health report yet. Click 'Trigger Retraining Now'.")

# =============================================================================
# TAB 8 — LIVE SIMULATION
# =============================================================================
with tabs[7]:
    st.subheader("Live Blood Inventory Simulation")

    # ── Simulation constants ───────────────────────────────────────────────────
    _SIM_H   = ["Hospital_A","Hospital_B","Hospital_C"]
    _SIM_BG  = ["O_positive","O_negative"]
    _DIST    = {("Hospital_A","Hospital_B"):44,("Hospital_B","Hospital_A"):44,
                ("Hospital_A","Hospital_C"):62,("Hospital_C","Hospital_A"):62,
                ("Hospital_B","Hospital_C"):38,("Hospital_C","Hospital_B"):38}
    _MEAN_HR = {"O_positive":35/24, "O_negative":15/24}   # mean demand units/hour
    _STD_HR  = {"O_positive":8/24,  "O_negative":5/24}
    _DAY_D   = {"O_positive":35,    "O_negative":15}       # units/day for DOS calc
    _INIT    = {"Hospital_A":{"O_positive":80,"O_negative":35},
                "Hospital_B":{"O_positive":60,"O_negative":25},
                "Hospital_C":{"O_positive":70,"O_negative":30}}
    _MAX_STK = 200
    _DONOR_DOS_THR = 8
    _RECIP_DOS_THR = 9

    # ── Session-state helpers ─────────────────────────────────────────────────
    def _sim_reset():
        st.session_state.sim_running   = False
        st.session_state.sim_time      = pd.Timestamp("2024-01-01 06:00:00")
        st.session_state.sim_inventory = {(h,bg): float(_INIT[h][bg])
                                           for h in _SIM_H for bg in _SIM_BG}
        st.session_state.sim_log       = []      # transfer events
        st.session_state.sim_history   = []      # inventory snapshots
        st.session_state.sim_last_tick = None

    if "sim_running" not in st.session_state:
        _sim_reset()

    # ── Controls ──────────────────────────────────────────────────────────────
    cc1, cc2, cc3, cc4 = st.columns([1,1,3,2])
    with cc1:
        if st.session_state.sim_running:
            if st.button("⏸  Pause",  type="secondary", key="sim_pause"):
                st.session_state.sim_running = False
                st.rerun()
        else:
            if st.button("▶  Start",  type="primary",   key="sim_start"):
                st.session_state.sim_running   = True
                st.session_state.sim_last_tick = time.time()
                st.rerun()
    with cc2:
        if st.button("↺  Reset", key="sim_reset"):
            _sim_reset(); st.rerun()
    with cc3:
        sim_speed = st.slider("Speed — sim hours per real second",
                              min_value=0.5, max_value=24.0, value=1.0, step=0.5,
                              help="1.0 = 1 real second → 1 sim hour  |  24.0 = 1 real second → 1 sim day",
                              key="sim_speed")
    with cc4:
        sim_refresh = st.slider("Refresh interval (real seconds)",
                                min_value=1, max_value=5, value=1, key="sim_refresh")

    st.divider()

    # ── Advance simulation on each rerun ─────────────────────────────────────
    if st.session_state.sim_running:
        now_real     = time.time()
        last_tick    = st.session_state.sim_last_tick
        elapsed_real = (now_real - last_tick) if last_tick is not None else sim_speed
        sim_hrs      = elapsed_real * sim_speed          # simulated hours this tick
        st.session_state.sim_last_tick = now_real

        rng = np.random.default_rng(int(now_real * 1000) % 2**32)

        # 1. Consume demand + add collections for every combo
        for h in _SIM_H:
            for bg in _SIM_BG:
                stock     = st.session_state.sim_inventory[(h, bg)]
                mean_d    = _MEAN_HR[bg] * sim_hrs
                std_d     = _STD_HR[bg]  * np.sqrt(max(sim_hrs, 0.01))
                demand    = max(0.0, float(rng.normal(mean_d, std_d)))
                collect   = max(0.0, float(rng.normal(mean_d * 1.1, std_d * 0.8)))
                # occasional donation spike (~5% chance scaled to tick size)
                if rng.random() < 0.05 * (sim_hrs / 24):
                    collect += float(rng.uniform(5, 20))
                stock = min(_MAX_STK, max(0.0, stock - demand + collect))
                st.session_state.sim_inventory[(h, bg)] = round(stock, 1)

        # 2. Snapshot for history chart
        snap_t = st.session_state.sim_time
        for h in _SIM_H:
            for bg in _SIM_BG:
                st.session_state.sim_history.append({
                    "sim_time": snap_t, "hospital": h,
                    "blood_group": bg,
                    "stock": st.session_state.sim_inventory[(h, bg)]
                })
        # keep at most ~7 sim-days of history snapshots
        if len(st.session_state.sim_history) > 6 * 168:
            st.session_state.sim_history = st.session_state.sim_history[-6*168:]

        # 3. Advance sim clock
        st.session_state.sim_time += pd.Timedelta(hours=sim_hrs)

        # 4. Check transfer triggers
        for bg in _SIM_BG:
            stocks = {h: st.session_state.sim_inventory[(h, bg)] for h in _SIM_H}
            dos    = {h: stocks[h] / _DAY_D[bg] for h in _SIM_H}
            recips = sorted([h for h in _SIM_H if dos[h] < _RECIP_DOS_THR], key=lambda x: dos[x])
            donors = sorted([h for h in _SIM_H if dos[h] > _DONOR_DOS_THR], key=lambda x: -dos[x])
            recent_keys = {(e["donor"], e["recipient"], e["blood_group"])
                           for e in st.session_state.sim_log[-12:]}
            for recip in recips:
                for donor in donors:
                    if donor == recip: continue
                    if (donor, recip, bg) in recent_keys: continue
                    qty = min(
                        int((dos[donor] - _DONOR_DOS_THR) * _DAY_D[bg] * 0.5),
                        int((_RECIP_DOS_THR - dos[recip])  * _DAY_D[bg])
                    )
                    if qty < 1: continue
                    dist = _DIST.get((donor, recip), 50)
                    st.session_state.sim_log.append({
                        "sim_time":    st.session_state.sim_time.strftime("%Y-%m-%d %H:%M"),
                        "donor":       donor, "recipient":  recip,
                        "blood_group": bg,    "qty":        qty,
                        "distance_km": dist,  "transit_hr": round(dist / 60, 1),
                        "donor_dos":   round(dos[donor], 1),
                        "recipient_dos": round(dos[recip], 1),
                    })
                    st.session_state.sim_inventory[(donor, bg)] = max(
                        0, st.session_state.sim_inventory[(donor, bg)] - qty)
                    st.session_state.sim_inventory[(recip, bg)] = min(
                        _MAX_STK, st.session_state.sim_inventory[(recip, bg)] + qty)
                    break  # one donor per recipient per tick

    # ── Status bar ────────────────────────────────────────────────────────────
    sim_label = "● RUNNING" if st.session_state.sim_running else "❚❚ PAUSED"
    sim_clr   = "#27ae60"    if st.session_state.sim_running else "#f39c12"
    st.markdown(
        f"<div style='display:flex;align-items:center;gap:20px;padding:8px 0;'>"
        f"<span style='background:{sim_clr};color:white;padding:3px 14px;"
        f"border-radius:12px;font-weight:bold;font-size:13px;'>{sim_label}</span>"
        f"<span style='font-size:20px;font-weight:bold;'>"
        f"Sim Time: {st.session_state.sim_time.strftime('%Y-%m-%d  %H:%M')}</span>"
        f"<span style='color:gray;font-size:12px;'>"
        f"Speed: {sim_speed}× &nbsp;|&nbsp; Transfers logged: {len(st.session_state.sim_log)}</span>"
        f"</div>", unsafe_allow_html=True)

    # ── Inventory gauges ──────────────────────────────────────────────────────
    st.subheader("Live Inventory")
    gcols = st.columns(len(_SIM_H))
    for col, h in zip(gcols, _SIM_H):
        with col:
            st.markdown(f"**{h}**")
            for bg in _SIM_BG:
                stock = st.session_state.sim_inventory[(h, bg)]
                dos   = stock / _DAY_D[bg]
                clr   = "#e74c3c" if dos < 2 else ("#f39c12" if dos < 4 else "#27ae60")
                pct   = min(100, stock / _MAX_STK * 100)
                label = bg.replace("_", " ")
                st.markdown(
                    f"<div style='margin-bottom:10px;'>"
                    f"<div style='display:flex;justify-content:space-between;font-size:12px;margin-bottom:2px;'>"
                    f"<span>{label}</span>"
                    f"<span style='color:{clr};font-weight:bold;'>{stock:.0f} u &nbsp;|&nbsp; {dos:.1f} d</span>"
                    f"</div>"
                    f"<div style='background:#e0e0e0;border-radius:5px;height:16px;'>"
                    f"<div style='background:{clr};width:{pct:.1f}%;height:16px;border-radius:5px;'></div>"
                    f"</div></div>",
                    unsafe_allow_html=True)

    # ── Inventory history chart ───────────────────────────────────────────────
    if st.session_state.sim_history:
        st.subheader("Inventory Over Sim Time")
        hist_sim = pd.DataFrame(st.session_state.sim_history)
        hist_sim["label"] = (hist_sim["hospital"].str.replace("Hospital_","H.") + " " +
                              hist_sim["blood_group"].str.replace("_"," "))
        fig_sim = go.Figure()
        for lbl in hist_sim["label"].unique():
            sub = hist_sim[hist_sim["label"] == lbl]
            fig_sim.add_trace(go.Scatter(x=sub["sim_time"], y=sub["stock"],
                                          mode="lines", name=lbl, line=dict(width=1.5)))
        fig_sim.add_hline(y=_DAY_D["O_positive"]*_RECIP_DOS_THR, line_dash="dot",
                           line_color="#e74c3c", opacity=0.4,
                           annotation_text="O+ transfer trigger", annotation_position="top left")
        fig_sim.add_hline(y=_DAY_D["O_negative"]*_RECIP_DOS_THR, line_dash="dot",
                           line_color="#3498db", opacity=0.4,
                           annotation_text="O- transfer trigger", annotation_position="bottom left")
        fig_sim.update_layout(height=300, hovermode="x unified",
                               xaxis_title="Sim Time", yaxis_title="Stock (units)",
                               margin=dict(l=0,r=0,t=20,b=0))
        st.plotly_chart(fig_sim, width="stretch")

    # ── Transfer event log ────────────────────────────────────────────────────
    st.subheader(f"Transfer Events ({len(st.session_state.sim_log)} total)")
    if st.session_state.sim_log:
        for ev in reversed(st.session_state.sim_log[-20:]):
            bg_clr = "#fde8e8" if ev["blood_group"] == "O_negative" else "#e8f8e8"
            st.markdown(
                f"<div style='background:{bg_clr};padding:5px 12px;border-radius:6px;"
                f"margin-bottom:4px;font-size:13px;'>"
                f"🕐 <b>{ev['sim_time']}</b> &nbsp;·&nbsp; "
                f"<b>{ev['donor'].replace('Hospital_','H.')}</b> → "
                f"<b>{ev['recipient'].replace('Hospital_','H.')}</b> &nbsp;·&nbsp; "
                f"{ev['blood_group'].replace('_',' ')} &nbsp;·&nbsp; "
                f"<b>{ev['qty']} units</b> &nbsp;·&nbsp; "
                f"{ev['distance_km']} km / {ev['transit_hr']} hr &nbsp;·&nbsp; "
                f"Donor DOS {ev['donor_dos']}d → Recip DOS {ev['recipient_dos']}d"
                f"</div>",
                unsafe_allow_html=True)
    else:
        st.info("No transfers yet — transfers fire when any hospital drops below 9 days of supply.")

    # ── Auto-rerun loop ───────────────────────────────────────────────────────
    if st.session_state.sim_running:
        time.sleep(sim_refresh)
        st.rerun()
