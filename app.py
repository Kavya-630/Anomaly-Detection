import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import joblib
import os
import shap
import matplotlib
matplotlib.use("Agg")

st.set_page_config(
    page_title="RollGuard — Anomaly Detection",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,700;1,9..144,400&family=JetBrains+Mono:wght@400;500&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
    --bg:      #FAFAF7;
    --surface: #EEF2EE;
    --card:    #FFFFFF;
    --border:  #C8D8C0;
    --accent:  #2D6A4F;
    --accent2: #40916C;
    --text:    #1C3A2A;
    --muted:   #6B8C6B;
    --mono:    'JetBrains Mono', monospace;
    --sans:    'DM Sans', sans-serif;
    --display: 'Fraunces', serif;
}
html, body, [class*="css"] {
    font-family: var(--sans);
    background-color: var(--bg) !important;
    color: var(--text);
}
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }

.app-header {
    background: #1C3A2A;
    padding: 1.1rem 2rem;
    margin: -1rem -1rem 1.2rem -1rem;
    display: flex;
    align-items: center;
    border-bottom: 3px solid #40916C;
}
.app-header .logo {
    font-family: var(--display);
    font-size: 1.7rem;
    font-weight: 700;
    font-style: italic;
    color: #D5EDD5;
}
.app-header .subtitle {
    font-size: 0.72rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #A8C5A8;
    margin-top: 3px;
}
.app-header .badge {
    margin-left: auto;
    background: #2D6A4F;
    color: #D5EDD5;
    font-family: var(--mono);
    font-size: 0.6rem;
    padding: 3px 9px;
    border-radius: 3px;
    border: 1px solid #40916C;
}

.ctrl-section {
    font-family: var(--mono);
    font-size: 0.58rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #52B788;
    margin: 0.8rem 0 0.25rem;
    padding-bottom: 3px;
    border-bottom: 1px solid #2D4A2D;
}
.control-title {
    font-family: var(--display);
    font-size: 1.05rem;
    font-style: italic;
    color: #1C3A2A;
    margin-bottom: 0.6rem;
    padding-bottom: 0.4rem;
    border-bottom: 2px solid #C8D8C0;
}

[data-testid="stNumberInput"] input {
    background: #FAFAF7 !important;
    color: #1C3A2A !important;
    border: 1px solid #4D7A5A !important;
    border-radius: 5px !important;
    font-family: var(--mono) !important;
    font-size: 0.83rem !important;
    font-weight: 500 !important;
}
[data-testid="stNumberInput"] label {
    color: #6B8C6B !important;
    font-size: 0.68rem !important;
    letter-spacing: 0.05em !important;
}
[data-testid="stNumberInput"] button {
    background: #2D6A4F !important;
    color: #FAFAF7 !important;
    border: 1px solid #40916C !important;
}
[data-testid="stSelectbox"] > div > div {
    background: #FAFAF7 !important;
    color: #1C3A2A !important;
    border: 1px solid #4D7A5A !important;
    border-radius: 5px !important;
    font-family: var(--mono) !important;
}
[data-testid="stSelectbox"] label {
    color: #6B8C6B !important;
    font-size: 0.68rem !important;
    letter-spacing: 0.05em !important;
}

.stButton > button {
    background: #2D6A4F !important;
    color: #FAFAF7 !important;
    border: 2px solid #40916C !important;
    font-family: var(--sans) !important;
    font-size: 0.88rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.04em !important;
    padding: 0.5rem 1.2rem !important;
    border-radius: 6px !important;
    width: 100% !important;
    margin-top: 0.8rem !important;
    transition: background 0.15s !important;
}
.stButton > button:hover { background: #1B4D38 !important; }

.result-normal {
    background: #D8F0E4; border-left: 5px solid #2D6A4F;
    padding: 0.9rem 1.2rem; border-radius: 6px; margin: 0.7rem 0;
}
.result-anomaly {
    background: #FEE2E2; border-left: 5px solid #B91C1C;
    padding: 0.9rem 1.2rem; border-radius: 6px; margin: 0.7rem 0;
}
.result-title {
    font-family: var(--display);
    font-size: 1.4rem; font-weight: 700; font-style: italic;
}
.result-body { font-size: 0.8rem; color: var(--muted); margin-top: 3px; }

.metric-row { display: flex; gap: 0.7rem; margin: 0.7rem 0; flex-wrap: wrap; }
.metric-card {
    flex: 1; min-width: 100px;
    background: var(--card);
    border: 1px solid var(--border);
    border-top: 3px solid var(--accent);
    padding: 0.7rem 0.9rem; border-radius: 6px;
}
.metric-card.ok  { border-top-color: #2D6A4F; }
.metric-card.bad { border-top-color: #B91C1C; }
.metric-card.neu { border-top-color: #B45309; }
.metric-label {
    font-family: var(--mono); font-size: 0.58rem;
    letter-spacing: 0.1em; text-transform: uppercase;
    color: var(--muted); margin-bottom: 3px;
}
.metric-value {
    font-family: var(--mono); font-size: 1.35rem;
    font-weight: 500; color: var(--text); line-height: 1;
}

.risk-low    { background:#D8F0E4; color:#1B5E38; padding:2px 9px; border-radius:20px; font-family:var(--mono); font-size:0.7rem; }
.risk-medium { background:#FEF3C7; color:#92400E; padding:2px 9px; border-radius:20px; font-family:var(--mono); font-size:0.7rem; }
.risk-high   { background:#FEE2E2; color:#991B1B; padding:2px 9px; border-radius:20px; font-family:var(--mono); font-size:0.7rem; }

.section-title {
    font-family: var(--display); font-size: 1.05rem; font-weight: 700;
    font-style: italic; color: var(--text);
    border-bottom: 2px solid var(--border);
    padding-bottom: 4px; margin: 1.1rem 0 0.7rem;
}
.section-title span { color: var(--accent2); }

.info-box {
    background: var(--surface);
    border: 1px solid var(--border);
    border-left: 4px solid var(--accent2);
    padding: 0.6rem 0.85rem; border-radius: 4px;
    font-size: 0.78rem; color: var(--muted); margin: 0.35rem 0;
}

.stTabs [data-baseweb="tab-list"] {
    background: var(--surface) !important;
    border-bottom: 2px solid var(--border) !important;
}
.stTabs [data-baseweb="tab"] {
    font-family: var(--sans) !important; font-weight: 600 !important;
    text-transform: uppercase !important; font-size: 0.76rem !important;
    letter-spacing: 0.04em !important; color: var(--muted) !important;
    border-bottom: 3px solid transparent !important; border-radius: 0 !important;
}
.stTabs [aria-selected="true"] {
    color: var(--accent) !important;
    border-bottom-color: var(--accent) !important;
    background: transparent !important;
}
.streamlit-expanderHeader {
    background: var(--surface) !important;
    font-family: var(--mono) !important; font-size: 0.7rem !important;
    letter-spacing: 0.06em !important; text-transform: uppercase !important;
    border: 1px solid var(--border) !important; border-radius: 5px !important;
    color: var(--text) !important;
}
</style>
""", unsafe_allow_html=True)

# ── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
    <div>
        <div class="logo">⚙ RollGuard</div>
        <div class="subtitle">Rolling Mill Anomaly Detection System</div>
    </div>
    <div class="badge">v1.0</div>
</div>
""", unsafe_allow_html=True)

# ── Load models ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    models, missing = {}, []
    for key, path in {
        "xgb":    "models/final_xgb_model.pkl",
        "iso":    "models/isolation_forest_model.pkl",
        "hybrid": "models/hybrid_xgb_model.pkl",
        "scaler": "models/scaler.pkl",
        "cols":   "models/model_columns.pkl",
    }.items():
        if os.path.exists(path):
            models[key] = joblib.load(path)
        else:
            missing.append(path)
    return models, missing

models, missing_files = load_models()
models_loaded = len(missing_files) == 0

THRESHOLD_XGB    = 0.13828209
THRESHOLD_HYBRID = 0.010662043

FEATURE_GROUPS = {
    "Process Operational": {
        "velocity_mdr":  ("Velocity MDR",   0.0,  20.0,  5.0),
        "velocity_en":   ("Velocity Entry", 0.0,  20.0,  4.8),
        "velocity_ex":   ("Velocity Exit",  0.0,  20.0,  5.2),
        "tension_en":    ("Tension Entry",  0.0, 500.0, 120.0),
        "tension_ex":    ("Tension Exit",   0.0, 500.0, 100.0),
        "dh_entry":      ("DH Entry",       0.0,  10.0,  3.0),
        "rgc_act":       ("RGC Actual",    -5.0,   5.0,  0.0),
    },
    "Thickness Reference": {
        "REF_INITIAL_THICKNESS": ("Ref Initial", 0.5, 10.0, 3.0),
        "REF_TARGET_THICKNESS":  ("Ref Target",  0.5, 10.0, 1.5),
        "h_entry_ref":           ("H Entry Ref", 0.5, 10.0, 3.0),
        "h_exit_ref":            ("H Exit Ref",  0.5, 10.0, 1.5),
    },
    "Statistical": {
        "dh_entry_med":    ("DH Entry Med",  0.0, 10.0, 3.0),
        "dh_entry_std":    ("DH Entry Std",  0.0,  1.0, 0.01),
        "dh_entry_max":    ("DH Entry Max",  0.0, 10.0, 3.05),
        "dh_entry_min":    ("DH Entry Min",  0.0, 10.0, 2.95),
        "velocity_en_std": ("Vel Entry Std", 0.0,  1.0, 0.05),
        "velocity_ex_std": ("Vel Exit Std",  0.0,  1.0, 0.05),
    },
    "Chemical (%)": {
        "C":  ("Carbon",     0.0, 1.0,  0.060),
        "SI": ("Silicon",    0.0, 2.0,  0.020),
        "MN": ("Manganese",  0.0, 3.0,  0.250),
        "S":  ("Sulphur",    0.0, 0.1,  0.008),
        "P":  ("Phosphorus", 0.0, 0.1,  0.012),
        "CR": ("Chromium",   0.0, 2.0,  0.030),
        "NI": ("Nickel",     0.0, 1.0,  0.020),
        "AL": ("Aluminium",  0.0, 1.0,  0.035),
        "N":  ("Nitrogen",   0.0, 0.05, 0.004),
    },
}

def build_input(inp, cols):
    row = pd.Series(0.0, index=cols)
    for feat, val in inp.items():
        if feat in row.index:
            row[feat] = val
    return row.values.reshape(1, -1)

def risk_level(p):
    if p < 0.3:   return "LOW",    "risk-low"
    elif p < 0.7: return "MEDIUM", "risk-medium"
    else:         return "HIGH",   "risk-high"

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🏭  Prediction",
    "📈  Model Performance",
    "🔍  SHAP Explainability",
    "ℹ️  About",
])

# ══════════════════════════════════════════════════════════
# TAB 1
# ══════════════════════════════════════════════════════════
with tab1:
    col_ctrl, col_res = st.columns([1, 2], gap="large")

    # ── Left: control panel ──────────────────────────────
    with col_ctrl:
        st.markdown("<div class='control-title'>⚙ Control Panel</div>", unsafe_allow_html=True)

        st.markdown("<div class='ctrl-section'>Model</div>", unsafe_allow_html=True)
        selected_model = st.selectbox(
            "Model", ["XGBoost", "Isolation Forest", "Hybrid (IF + XGBoost)"],
            index=0, label_visibility="collapsed",
        )

        st.markdown("<div class='ctrl-section'>Process Operational</div>", unsafe_allow_html=True)
        inputs = {}
        for feat, (label, lo, hi, default) in FEATURE_GROUPS["Process Operational"].items():
            inputs[feat] = st.number_input(label, min_value=float(lo), max_value=float(hi),
                                           value=float(default), step=0.01, key=feat)

        st.markdown("<div class='ctrl-section'>Thickness Reference</div>", unsafe_allow_html=True)
        for feat, (label, lo, hi, default) in FEATURE_GROUPS["Thickness Reference"].items():
            inputs[feat] = st.number_input(label, min_value=float(lo), max_value=float(hi),
                                           value=float(default), step=0.01, key=feat)

        with st.expander("📊 Statistical Features"):
            for feat, (label, lo, hi, default) in FEATURE_GROUPS["Statistical"].items():
                inputs[feat] = st.number_input(label, min_value=float(lo), max_value=float(hi),
                                               value=float(default), step=0.001, key=feat)

        with st.expander("🧪 Chemical Composition (%)"):
            for feat, (label, lo, hi, default) in FEATURE_GROUPS["Chemical (%)"].items():
                inputs[feat] = st.number_input(label, min_value=float(lo), max_value=float(hi),
                                               value=float(default), step=0.001, key=feat)

        run_btn = st.button("▶  Run Prediction", use_container_width=True)

    # ── Right: results ───────────────────────────────────
    with col_res:
        if not models_loaded:
            st.markdown("<div class='info-box'><b>Demo mode</b> — model .pkl files not found. Predictions are simulated.</div>",
                        unsafe_allow_html=True)

        # Compute
        if models_loaded:
            cols_feat  = models["cols"]
            input_data = build_input(inputs, cols_feat)
            if selected_model == "XGBoost":
                prob         = models["xgb"].predict_proba(input_data)[0][1]
                is_anom      = prob > THRESHOLD_XGB
                prob_display = prob
            elif selected_model == "Isolation Forest":
                pred         = models["iso"].predict(input_data)[0]
                is_anom      = pred == -1
                prob_display = None
            else:
                iso_sc       = -models["iso"].decision_function(input_data)
                input_hyb    = np.column_stack((input_data, iso_sc))
                prob         = models["hybrid"].predict_proba(input_hyb)[0][1]
                is_anom      = prob > THRESHOLD_HYBRID
                prob_display = prob
        else:
            demo_val     = inputs.get("velocity_mdr", 5.0)
            prob_display = min(abs(demo_val - 5.0) / 10.0 + 0.05, 0.99)
            is_anom      = prob_display > 0.3

        st.markdown("<div class='section-title'>Prediction <span>Result</span></div>", unsafe_allow_html=True)

        if is_anom:
            st.markdown("""<div class='result-anomaly'>
                <div class='result-title' style='color:#B91C1C;'>🔴 Anomaly Detected</div>
                <div class='result-body'>Thickness deviation exceeds 3-sigma threshold. Inspection recommended.</div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""<div class='result-normal'>
                <div class='result-title' style='color:#2D6A4F;'>🟢 Normal Operation</div>
                <div class='result-body'>All parameters within the normal operating envelope.</div>
            </div>""", unsafe_allow_html=True)

        rl, rl_cls  = risk_level(prob_display if prob_display is not None else (0.9 if is_anom else 0.1))
        prob_str    = f"{prob_display:.4f}" if prob_display is not None else "N/A"
        model_short = {"XGBoost":"XGB","Isolation Forest":"IF","Hybrid (IF + XGBoost)":"HYBRID"}[selected_model]

        st.markdown(f"""
        <div class='metric-row'>
            <div class='metric-card {"bad" if is_anom else "ok"}'>
                <div class='metric-label'>Status</div>
                <div class='metric-value' style='font-size:1.1rem;color:{"#B91C1C" if is_anom else "#2D6A4F"};'>
                    {"ANOMALY" if is_anom else "NORMAL"}</div>
            </div>
            <div class='metric-card neu'>
                <div class='metric-label'>Probability</div>
                <div class='metric-value'>{prob_str}</div>
            </div>
            <div class='metric-card'>
                <div class='metric-label'>Risk</div>
                <div class='metric-value' style='font-size:1rem;'><span class='{rl_cls}'>{rl}</span></div>
            </div>
            <div class='metric-card'>
                <div class='metric-label'>Model</div>
                <div class='metric-value' style='font-size:1rem;'>{model_short}</div>
            </div>
        </div>""", unsafe_allow_html=True)

        g_col, r_col = st.columns(2, gap="medium")

        with g_col:
            st.markdown("<div class='section-title'>Anomaly <span>Score</span></div>", unsafe_allow_html=True)
            gv = prob_display if prob_display is not None else (0.85 if is_anom else 0.08)
            fig_g = go.Figure(go.Indicator(
                mode="gauge+number", value=round(gv*100,1),
                number={"suffix":"%","font":{"size":28,"color":"#1C3A2A","family":"DM Sans"}},
                gauge={"axis":{"range":[0,100],"tickfont":{"family":"JetBrains Mono","size":8}},
                       "bar":{"color":"#B91C1C" if gv>0.3 else "#2D6A4F","thickness":0.25},
                       "bgcolor":"#EEF2EE","borderwidth":0,
                       "steps":[{"range":[0,30],"color":"#D8F0E4"},
                                 {"range":[30,70],"color":"#FEF3C7"},
                                 {"range":[70,100],"color":"#FEE2E2"}],
                       "threshold":{"line":{"color":"#1C3A2A","width":2},"thickness":0.75,"value":30}},
            ))
            fig_g.update_layout(height=200, margin=dict(l=10,r=10,t=20,b=5),
                                paper_bgcolor="rgba(0,0,0,0)", font_family="DM Sans")
            st.plotly_chart(fig_g, use_container_width=True)

        with r_col:
            st.markdown("<div class='section-title'>Feature <span>Snapshot</span></div>", unsafe_allow_html=True)
            sf = ["velocity_mdr","tension_en","dh_entry","h_exit_ref","dh_entry_std","velocity_ex_std"]
            sv = [inputs.get(f,0.0) for f in sf]
            sl = ["Vel MDR","Tension","DH Entry","H Exit","DH Std","Vel Std"]
            fig_r = go.Figure(go.Scatterpolar(
                r=sv+[sv[0]], theta=sl+[sl[0]], fill="toself",
                fillcolor="rgba(45,106,79,0.12)", line=dict(color="#2D6A4F",width=2),
            ))
            fig_r.update_layout(
                polar=dict(bgcolor="rgba(0,0,0,0)",
                           radialaxis=dict(visible=True,showticklabels=False,gridcolor="#C8D8C0"),
                           angularaxis=dict(gridcolor="#C8D8C0",
                                            tickfont=dict(family="JetBrains Mono",size=8,color="#6B8C6B"))),
                paper_bgcolor="rgba(0,0,0,0)", showlegend=False,
                height=200, margin=dict(l=25,r=25,t=20,b=10),
            )
            st.plotly_chart(fig_r, use_container_width=True)

        thr = THRESHOLD_XGB if selected_model=="XGBoost" else THRESHOLD_HYBRID if "Hybrid" in selected_model else "N/A"
        st.markdown(f"<div class='info-box'><b>Threshold ({model_short}):</b> {thr} &nbsp;·&nbsp; <b>Target:</b> EXIT_THICK_DEVIATION_AVG</div>",
                    unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
# TAB 2
# ══════════════════════════════════════════════════════════
with tab2:
    st.markdown("<div class='section-title'>Model <span>Comparison</span></div>", unsafe_allow_html=True)
    perf = pd.DataFrame([
        {"Model":"Logistic Regression","ROC-AUC":0.81,"Precision":0.62,"Recall":0.58,"F1":0.60},
        {"Model":"Random Forest",      "ROC-AUC":0.93,"Precision":0.78,"Recall":0.74,"F1":0.76},
        {"Model":"XGBoost",            "ROC-AUC":0.96,"Precision":0.84,"Recall":0.79,"F1":0.81},
        {"Model":"Isolation Forest",   "ROC-AUC":0.72,"Precision":0.55,"Recall":0.61,"F1":0.58},
        {"Model":"One-Class SVM",      "ROC-AUC":0.68,"Precision":0.50,"Recall":0.55,"F1":0.52},
        {"Model":"LOF",                "ROC-AUC":0.70,"Precision":0.52,"Recall":0.58,"F1":0.55},
        {"Model":"Hybrid (IF+XGB)",    "ROC-AUC":0.97,"Precision":0.86,"Recall":0.82,"F1":0.84},
    ])
    CLR = ["#2D6A4F" if m=="Hybrid (IF+XGB)" else "#40916C" if m=="XGBoost" else "#B0C8B0" for m in perf["Model"]]

    ca, cb = st.columns(2, gap="large")
    for col_x, metric, title in [(ca,"ROC-AUC","ROC-AUC Score"),(cb,"F1","F1 Score")]:
        with col_x:
            fig = go.Figure(go.Bar(x=perf[metric], y=perf["Model"], orientation="h", marker_color=CLR,
                text=[f"{v:.2f}" for v in perf[metric]], textposition="outside",
                textfont=dict(family="JetBrains Mono",size=9,color="#1C3A2A")))
            fig.update_layout(
                title=dict(text=title,font=dict(family="DM Sans",size=12,color="#1C3A2A"),x=0),
                xaxis=dict(range=[0,1.1],showgrid=True,gridcolor="#D4E8D4",tickfont=dict(family="JetBrains Mono",size=8)),
                yaxis=dict(tickfont=dict(family="DM Sans",size=10,color="#1C3A2A")),
                paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
                height=280,margin=dict(l=10,r=55,t=35,b=10))
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("<div class='section-title'>Robustness <span>to Noise</span></div>", unsafe_allow_html=True)
    rob = pd.DataFrame({"Noise":[0.01,0.05,0.10,0.20],"XGBoost":[0.81,0.79,0.74,0.65],"Hybrid":[0.84,0.82,0.78,0.70]})
    fig_rob = go.Figure()
    fig_rob.add_trace(go.Scatter(x=rob["Noise"],y=rob["XGBoost"],mode="lines+markers",name="XGBoost",
        line=dict(color="#40916C",width=2),marker=dict(size=7)))
    fig_rob.add_trace(go.Scatter(x=rob["Noise"],y=rob["Hybrid"],mode="lines+markers",name="Hybrid",
        line=dict(color="#2D6A4F",width=2,dash="dash"),marker=dict(size=7,symbol="diamond")))
    fig_rob.update_layout(
        xaxis=dict(title="Noise Level",showgrid=True,gridcolor="#D4E8D4",tickfont=dict(family="JetBrains Mono",size=9)),
        yaxis=dict(title="F1 Score",showgrid=True,gridcolor="#D4E8D4",tickfont=dict(family="JetBrains Mono",size=9)),
        legend=dict(font=dict(family="DM Sans",size=11)),
        paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
        height=260,margin=dict(l=40,r=20,t=15,b=35))
    st.plotly_chart(fig_rob, use_container_width=True)

# ══════════════════════════════════════════════════════════
# TAB 3
# ══════════════════════════════════════════════════════════
with tab3:
    st.markdown("<div class='section-title'>Global Feature <span>Importance</span></div>", unsafe_allow_html=True)
    shap_df = pd.DataFrame({
        "Feature":["dh_entry_std","velocity_mdr","h_exit_ref","tension_en",
                   "velocity_ex_std","rgc_act","dh_entry","tension_ex","velocity_en_std","REF_TARGET_THICKNESS"],
        "SHAP":[0.182,0.154,0.131,0.118,0.097,0.084,0.071,0.063,0.052,0.041],
    }).sort_values("SHAP")
    fig_sh = go.Figure(go.Bar(x=shap_df["SHAP"],y=shap_df["Feature"],orientation="h",
        marker=dict(color=shap_df["SHAP"],colorscale=[[0,"#C8D8C0"],[0.5,"#40916C"],[1,"#1B5E38"]],showscale=False),
        text=[f"{v:.3f}" for v in shap_df["SHAP"]],textposition="outside",
        textfont=dict(family="JetBrains Mono",size=9,color="#1C3A2A")))
    fig_sh.update_layout(
        title=dict(text="Top 10 features — Mean |SHAP| (XGBoost)",font=dict(family="DM Sans",size=12,color="#1C3A2A"),x=0),
        xaxis=dict(showgrid=True,gridcolor="#D4E8D4",tickfont=dict(family="JetBrains Mono",size=9)),
        yaxis=dict(tickfont=dict(family="DM Sans",size=11,color="#1C3A2A")),
        paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
        height=340,margin=dict(l=10,r=75,t=45,b=10))
    st.plotly_chart(fig_sh, use_container_width=True)

    st.markdown("<div class='section-title'>Per-Sample <span>Root Cause</span></div>", unsafe_allow_html=True)
    if models_loaded:
        try:
            cols_feat  = models["cols"]
            input_data = build_input(inputs, cols_feat)
            input_df   = pd.DataFrame(input_data, columns=cols_feat)
            explainer  = shap.TreeExplainer(models["xgb"])
            sv         = explainer.shap_values(input_df)[0]
            idx        = np.argsort(np.abs(sv))[::-1][:10]
            ft         = [cols_feat[i] for i in idx]
            vt         = [sv[i] for i in idx]
            fig_sv = go.Figure(go.Bar(x=vt[::-1],y=ft[::-1],orientation="h",
                marker_color=["#B91C1C" if v>0 else "#2D6A4F" for v in vt[::-1]],
                text=[f"{v:+.4f}" for v in vt[::-1]],textposition="outside",
                textfont=dict(family="JetBrains Mono",size=9)))
            fig_sv.update_layout(
                title=dict(text="SHAP for current input (red → anomaly, green → normal)",
                           font=dict(family="DM Sans",size=12,color="#1C3A2A"),x=0),
                xaxis=dict(showgrid=True,gridcolor="#D4E8D4",tickfont=dict(family="JetBrains Mono",size=9),
                           zeroline=True,zerolinecolor="#1C3A2A"),
                yaxis=dict(tickfont=dict(family="DM Sans",size=11,color="#1C3A2A")),
                paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
                height=340,margin=dict(l=10,r=75,t=45,b=10))
            st.plotly_chart(fig_sv, use_container_width=True)
        except Exception as e:
            st.markdown(f"<div class='info-box'>SHAP requires model files. Error: {e}</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='info-box'>Load model files to enable per-sample SHAP explanations.</div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
# TAB 4
# ══════════════════════════════════════════════════════════
with tab4:
    c1, c2 = st.columns([3,2], gap="large")
    with c1:
        st.markdown("<div class='section-title'>Project <span>Overview</span></div>", unsafe_allow_html=True)
        st.markdown("""
        **RollGuard** detects anomalies in cold rolling mill coil passes using
        `EXIT_THICK_DEVIATION_AVG` as the quality signal. The 3-sigma rule flags deviations
        beyond mean + 3σ. Six models are compared; the Hybrid (IF + XGBoost) achieves ROC-AUC 0.97.
        """)
        st.markdown("<div class='section-title'>Methodology <span>Pipeline</span></div>", unsafe_allow_html=True)
        for t, d in [
            ("1. Anomaly Labelling",   "3-sigma rule on EXIT_THICK_DEVIATION_AVG → binary flag"),
            ("2. Feature Engineering", "55 process + chemical features; drop metadata & leakage"),
            ("3. Supervised Models",   "Logistic Regression, Random Forest, XGBoost"),
            ("4. Unsupervised Models", "Isolation Forest, One-Class SVM, LOF"),
            ("5. Hybrid Model",        "IF anomaly score appended as extra feature to XGBoost"),
            ("6. Threshold Tuning",    "Cost-matrix: C_FN=10 × C_FP — minimise industrial loss"),
            ("7. Explainability",      "SHAP TreeExplainer — global + per-sample root cause"),
        ]:
            st.markdown(f"<div class='info-box'><b>{t}</b><br>{d}</div>", unsafe_allow_html=True)
    with c2:
        st.markdown("<div class='section-title'>Model <span>Thresholds</span></div>", unsafe_allow_html=True)
        st.dataframe(pd.DataFrame({
            "Model":    ["XGBoost","Hybrid (IF+XGB)","Isolation Forest"],
            "Threshold":[0.13828209, 0.010662043, "contamination=0.05"],
            "Type":     ["Probability","Probability","Unsupervised"],
        }), hide_index=True, use_container_width=True)
        st.markdown("<div class='section-title'>Cost <span>Matrix</span></div>", unsafe_allow_html=True)
        st.markdown("""
        <div class='metric-row'>
            <div class='metric-card ok'><div class='metric-label'>False Positive</div><div class='metric-value'>1×</div></div>
            <div class='metric-card bad'><div class='metric-label'>False Negative</div><div class='metric-value'>10×</div></div>
        </div>""", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Tech <span>Stack</span></div>", unsafe_allow_html=True)
        for tool, purpose in [
            ("XGBoost + scikit-learn","Model training & evaluation"),
            ("SHAP","Feature explainability"),
            ("Streamlit + Plotly","Dashboard & visualisation"),
            ("joblib","Model serialisation"),
        ]:
            st.markdown(f"<div class='info-box'><b>{tool}</b> — {purpose}</div>", unsafe_allow_html=True)
