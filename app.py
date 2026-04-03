import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import joblib
import os
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ─── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RollGuard — Anomaly Detection",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS — forest green + cream palette ──────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,400;0,9..144,700;1,9..144,400&family=JetBrains+Mono:wght@400;500&family=DM+Sans:wght@300;400;500;600&display=swap');

/* ── Root palette ── */
:root {
    --bg:        #FAFAF7;
    --surface:   #EEF2EE;
    --card:      #FFFFFF;
    --border:    #C8D8C0;
    --accent:    #2D6A4F;
    --accent2:   #40916C;
    --text:      #1C3A2A;
    --muted:     #6B8C6B;
    --ok:        #2D6A4F;
    --warn:      #B91C1C;
    --neu:       #B45309;
    --mono:      'JetBrains Mono', monospace;
    --sans:      'DM Sans', sans-serif;
    --display:   'Fraunces', serif;
}

/* ── Global resets ── */
html, body, [class*="css"] {
    font-family: var(--sans);
    background-color: var(--bg) !important;
    color: var(--text);
}

/* ── Hide Streamlit chrome — keep header visible for sidebar toggle ── */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
.stDeployButton { display: none; }

/* Hide only the elements inside the header we don't need */
[data-testid="stToolbar"] { display: none; }
[data-testid="stDecoration"] { display: none; }
[data-testid="stStatusWidget"] { visibility: hidden; }

/* Keep the header bar itself visible but reduce its height */
header[data-testid="stHeader"] {
    background: transparent !important;
    height: 2.5rem !important;
}

/* Style the sidebar collapse arrow inside the header — green, always visible */
[data-testid="collapsedControl"] {
    background-color: #2D6A4F !important;
    border-radius: 0 6px 6px 0 !important;
    height: 2.5rem !important;
    width: 2rem !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
}
[data-testid="collapsedControl"]:hover {
    background-color: #40916C !important;
}
[data-testid="collapsedControl"] svg {
    fill: #FAFAF7 !important;
}

/* ── App header ── */
.app-header {
    background: #1C3A2A;
    color: #D5EDD5;
    padding: 1.4rem 2rem 1.2rem;
    margin: -1rem -1rem 2rem -1rem;
    display: flex;
    align-items: center;
    gap: 1.2rem;
    border-bottom: 3px solid var(--accent);
}
.app-header .logo {
    font-family: var(--display);
    font-size: 2rem;
    font-weight: 700;
    font-style: italic;
    letter-spacing: 0.01em;
    color: #D5EDD5;
}
.app-header .subtitle {
    font-size: 0.8rem;
    font-weight: 300;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #A8C5A8;
    margin-top: 2px;
}
.app-header .badge {
    margin-left: auto;
    background: var(--accent);
    color: #fff;
    font-family: var(--mono);
    font-size: 0.65rem;
    padding: 3px 10px;
    letter-spacing: 0.1em;
    border-radius: 2px;
}

/* ── Sidebar shell ── */
[data-testid="stSidebar"] {
    background: #1C3A2A !important;
    border-right: 2px solid #2D4A2D;
    min-height: 100vh;
}

/* ── Scrollable content area ── */
[data-testid="stSidebar"] > div:first-child {
    overflow-y: auto !important;
    overflow-x: hidden !important;
    height: 100vh !important;
    padding-bottom: 5rem !important;
    scrollbar-width: thin !important;
    scrollbar-color: #40916C #1C3A2A !important;
}

/* Webkit scrollbar styling */
[data-testid="stSidebar"] > div:first-child::-webkit-scrollbar {
    width: 5px !important;
}
[data-testid="stSidebar"] > div:first-child::-webkit-scrollbar-track {
    background: #1C3A2A !important;
    border-radius: 3px !important;
}
[data-testid="stSidebar"] > div:first-child::-webkit-scrollbar-thumb {
    background: #40916C !important;
    border-radius: 3px !important;
}
[data-testid="stSidebar"] > div:first-child::-webkit-scrollbar-thumb:hover {
    background: #52B788 !important;
}

/* collapsedControl styles moved to the Hide chrome section above */

/* Sidebar text — light on dark bg */
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] div { color: #D5EDD5 !important; }

/* Input boxes — white bg, dark text so numbers are visible */
[data-testid="stSidebar"] input[type="number"],
[data-testid="stSidebar"] .stNumberInput input {
    background: #FAFAF7 !important;
    color: #1C3A2A !important;
    border: 1px solid #4D7A5A !important;
    border-radius: 6px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.9rem !important;
    font-weight: 500 !important;
}

/* Selectbox — white bg, dark text */
[data-testid="stSidebar"] .stSelectbox > div > div {
    background: #FAFAF7 !important;
    color: #1C3A2A !important;
    border: 1px solid #4D7A5A !important;
    border-radius: 6px !important;
}
[data-testid="stSidebar"] .stSelectbox > div > div > div {
    color: #1C3A2A !important;
}

/* +/- stepper buttons on number inputs */
[data-testid="stSidebar"] .stNumberInput button {
    background: #2D6A4F !important;
    color: #FAFAF7 !important;
    border: 1px solid #4D7A5A !important;
}
[data-testid="stSidebar"] .stNumberInput button:hover {
    background: #40916C !important;
}

/* Section labels */
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stNumberInput label,
[data-testid="stSidebar"] .stSlider label {
    color: #A8C5A8 !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
}

[data-testid="stSidebar"] [data-testid="stVerticalBlock"] > div:first-child { padding-top: 0.5rem; }
.sidebar-section {
    font-family: var(--sans);
    font-size: 0.65rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--accent) !important;
    margin: 1.2rem 0 0.4rem;
    padding-bottom: 4px;
    border-bottom: 1px solid #2D4A2D;
}

/* ── Metric cards ── */
.metric-row { display: flex; gap: 1rem; margin-bottom: 1.5rem; }
.metric-card {
    flex: 1;
    background: var(--card);
    border: 1px solid var(--border);
    border-top: 3px solid var(--accent);
    padding: 1rem 1.2rem;
    border-radius: 2px;
}
.metric-card.ok  { border-top-color: var(--ok);   }
.metric-card.bad { border-top-color: var(--warn);  }
.metric-card.neu { border-top-color: var(--neu);     }
.metric-label {
    font-family: var(--mono);
    font-size: 0.65rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 4px;
}
.metric-value {
    font-family: var(--mono);
    font-size: 1.8rem;
    font-weight: 500;
    color: var(--text);
    line-height: 1;
}
.metric-sub {
    font-size: 0.72rem;
    color: var(--muted);
    margin-top: 4px;
}

/* ── Result banner ── */
.result-normal {
    background: #D8F0E4;
    border: 1px solid #B0D8C0;
    border-left: 5px solid var(--ok);
    padding: 1.2rem 1.5rem;
    border-radius: 2px;
    margin: 1rem 0;
}
.result-anomaly {
    background: #FEE2E2;
    border: 1px solid #FCA5A5;
    border-left: 5px solid var(--warn);
    padding: 1.2rem 1.5rem;
    border-radius: 2px;
    margin: 1rem 0;
}
.result-title {
    font-family: var(--display);
    font-size: 1.6rem;
    font-weight: 700;
    font-style: italic;
}
.result-body { font-size: 0.85rem; color: var(--muted); margin-top: 6px; }

/* ── Risk pill ── */
.risk-low    { background:#D8F0E4; color:#1B5E38; padding:3px 12px; border-radius:20px; font-family:var(--mono); font-size:0.75rem; font-weight:500; }
.risk-medium { background:#FEF3C7; color:#92400E; padding:3px 12px; border-radius:20px; font-family:var(--mono); font-size:0.75rem; font-weight:500; }
.risk-high   { background:#FEE2E2; color:#991B1B; padding:3px 12px; border-radius:20px; font-family:var(--mono); font-size:0.75rem; font-weight:500; }

/* ── Section title ── */
.section-title {
    font-family: var(--display);
    font-size: 1.2rem;
    font-weight: 700;
    font-style: italic;
    letter-spacing: 0.01em;
    color: var(--text);
    border-bottom: 2px solid var(--border);
    padding-bottom: 6px;
    margin: 1.5rem 0 1rem;
}
.section-title span { color: var(--accent); }

/* ── Streamlit widget overrides ── */
.stSelectbox > div > div,
.stNumberInput > div > div > input {
    background: var(--card) !important;
    border-color: var(--border) !important;
    border-radius: 2px !important;
    font-family: var(--mono) !important;
    font-size: 0.85rem !important;
}
.stButton > button {
    background: var(--accent) !important;
    color: #fff !important;
    border: none !important;
    font-family: var(--sans) !important;
    font-size: 0.95rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.04em !important;
    text-transform: uppercase !important;
    padding: 0.6rem 2rem !important;
    border-radius: 6px !important;
    width: 100% !important;
    transition: background 0.2s !important;
}
.stButton > button:hover { background: #1B4D38 !important; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: var(--surface) !important;
    border-bottom: 2px solid var(--border) !important;
    gap: 0 !important;
}
.stTabs [data-baseweb="tab"] {
    font-family: var(--sans) !important;
    font-weight: 600 !important;
    letter-spacing: 0.04em !important;
    text-transform: uppercase !important;
    font-size: 0.82rem !important;
    color: var(--muted) !important;
    padding: 0.6rem 1.2rem !important;
    border-radius: 0 !important;
    border-bottom: 3px solid transparent !important;
}
.stTabs [aria-selected="true"] {
    color: var(--accent) !important;
    border-bottom-color: var(--accent) !important;
    background: transparent !important;
}

/* ── Expander ── */
.streamlit-expanderHeader {
    font-family: var(--sans) !important;
    font-weight: 600 !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
    font-size: 0.8rem !important;
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 2px !important;
}

/* ── Plotly chart containers ── */
.js-plotly-plot { border-radius: 2px !important; }

/* ── Divider ── */
.divider { border: none; border-top: 1px solid var(--border); margin: 1.5rem 0; }

/* ── Info box ── */
.info-box {
    background: var(--surface);
    border: 1px solid var(--border);
    border-left: 4px solid var(--accent2);
    padding: 0.8rem 1rem;
    border-radius: 2px;
    font-size: 0.82rem;
    color: var(--muted);
    margin: 0.5rem 0;
}
</style>
""", unsafe_allow_html=True)

# ─── Header ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
    <div>
        <div class="logo">⚙ ROLLGUARD</div>
        <div class="subtitle">Rolling Mill Anomaly Detection System</div>
    </div>
    <div class="badge">v1.0</div>
</div>
""", unsafe_allow_html=True)

# ─── Load models ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    models = {}
    files = {
        "xgb":    "models/final_xgb_model.pkl",
        "iso":    "models/isolation_forest_model.pkl",
        "hybrid": "models/hybrid_xgb_model.pkl",
        "scaler": "models/scaler.pkl",
        "cols":   "models/model_columns.pkl",
    }
    missing = []
    for key, path in files.items():
        if os.path.exists(path):
            models[key] = joblib.load(path)
        else:
            missing.append(path)
    return models, missing

models, missing_files = load_models()
models_loaded = len(missing_files) == 0

# ─── Thresholds ──────────────────────────────────────────────────────────────
THRESHOLD_XGB    = 0.13828209
THRESHOLD_HYBRID = 0.010662043

# ─── Feature groups ──────────────────────────────────────────────────────────
FEATURE_GROUPS = {
    "⚙️ Process Operational": {
        "velocity_mdr":  ("Velocity MDR",   0.0,  20.0,  5.0),
        "velocity_en":   ("Velocity Entry", 0.0,  20.0,  4.8),
        "velocity_ex":   ("Velocity Exit",  0.0,  20.0,  5.2),
        "tension_en":    ("Tension Entry",  0.0, 500.0, 120.0),
        "tension_ex":    ("Tension Exit",   0.0, 500.0, 100.0),
        "dh_entry":      ("DH Entry",       0.0,  10.0,  3.0),
        "rgc_act":       ("RGC Actual",    -5.0,   5.0,  0.0),
    },
    "📐 Thickness Reference": {
        "REF_INITIAL_THICKNESS": ("Ref Initial Thick.", 0.5, 10.0, 3.0),
        "REF_TARGET_THICKNESS":  ("Ref Target Thick.",  0.5, 10.0, 1.5),
        "h_entry_ref":           ("H Entry Ref",        0.5, 10.0, 3.0),
        "h_exit_ref":            ("H Exit Ref",         0.5, 10.0, 1.5),
    },
    "📊 Statistical Features": {
        "dh_entry_med":    ("DH Entry Median", 0.0,  10.0, 3.0),
        "dh_entry_std":    ("DH Entry Std",    0.0,   1.0, 0.01),
        "dh_entry_max":    ("DH Entry Max",    0.0,  10.0, 3.05),
        "dh_entry_min":    ("DH Entry Min",    0.0,  10.0, 2.95),
        "velocity_en_std": ("Vel. Entry Std",  0.0,   1.0, 0.05),
        "velocity_ex_std": ("Vel. Exit Std",   0.0,   1.0, 0.05),
    },
    "🧪 Chemical Composition (%)": {
        "C":  ("Carbon (C)",     0.0, 1.0, 0.06),
        "SI": ("Silicon (SI)",   0.0, 2.0, 0.02),
        "MN": ("Manganese (MN)", 0.0, 3.0, 0.25),
        "S":  ("Sulphur (S)",    0.0, 0.1, 0.008),
        "P":  ("Phosphorus (P)", 0.0, 0.1, 0.012),
        "CR": ("Chromium (CR)",  0.0, 2.0, 0.03),
        "NI": ("Nickel (NI)",    0.0, 1.0, 0.02),
        "AL": ("Aluminium (AL)", 0.0, 1.0, 0.035),
        "N":  ("Nitrogen (N)",   0.0, 0.05, 0.004),
    },
}

# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("<div style='padding:1rem 0 0.5rem;font-family:var(--display);font-size:1.3rem;font-weight:700;font-style:italic;color:#D5EDD5;'>Control Panel</div>", unsafe_allow_html=True)

    st.markdown("<div class='sidebar-section'>Model Selection</div>", unsafe_allow_html=True)
    selected_model = st.selectbox(
        "Active Model",
        ["XGBoost", "Isolation Forest", "Hybrid (IF + XGBoost)"],
        index=0,
        label_visibility="collapsed",
    )

    st.markdown("<div class='sidebar-section'>Process Operational</div>", unsafe_allow_html=True)
    inputs = {}
    for feat, (label, lo, hi, default) in FEATURE_GROUPS["⚙️ Process Operational"].items():
        inputs[feat] = st.number_input(label, min_value=float(lo), max_value=float(hi), value=float(default), step=0.01, key=feat)

    st.markdown("<div class='sidebar-section'>Thickness Reference</div>", unsafe_allow_html=True)
    for feat, (label, lo, hi, default) in FEATURE_GROUPS["📐 Thickness Reference"].items():
        inputs[feat] = st.number_input(label, min_value=float(lo), max_value=float(hi), value=float(default), step=0.01, key=feat)

    with st.expander("📊 Statistical Features"):
        for feat, (label, lo, hi, default) in FEATURE_GROUPS["📊 Statistical Features"].items():
            inputs[feat] = st.number_input(label, min_value=float(lo), max_value=float(hi), value=float(default), step=0.001, key=feat)

    with st.expander("🧪 Chemical Composition"):
        for feat, (label, lo, hi, default) in FEATURE_GROUPS["🧪 Chemical Composition (%)"].items():
            inputs[feat] = st.number_input(label, min_value=float(lo), max_value=float(hi), value=float(default), step=0.001, key=feat)

    st.markdown("<hr style='border-color:#2D4A2D;margin:1rem 0;'>", unsafe_allow_html=True)
    run_btn = st.button("▶  RUN PREDICTION", use_container_width=True)


# ─── Helper: build full input vector ────────────────────────────────────────
def build_input(inputs, cols):
    row = pd.Series(0.0, index=cols)
    for feat, val in inputs.items():
        if feat in row.index:
            row[feat] = val
    return row.values.reshape(1, -1)


def risk_level(p):
    if p < 0.3:   return "LOW",    "risk-low"
    elif p < 0.7: return "MEDIUM", "risk-medium"
    else:         return "HIGH",   "risk-high"


# ─── Tabs ────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🏭  Prediction",
    "📈  Model Performance",
    "🔍  SHAP Explainability",
    "ℹ️  About",
])


# ══════════════════════════════════════════════════════════
# TAB 1 — Prediction
# ══════════════════════════════════════════════════════════
with tab1:
    if not models_loaded:
        st.markdown(f"""
        <div class='result-anomaly'>
            <div class='result-title' style='color:var(--warn);'>⚠ Model Files Not Found</div>
            <div class='result-body'>
                Place your trained <code>.pkl</code> files in the <code>models/</code> folder.<br>
                Missing: <code>{"</code>, <code>".join(missing_files)}</code>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div class='section-title'>Demo Mode — <span>Simulated Prediction</span></div>", unsafe_allow_html=True)
        st.markdown("<div class='info-box'>Running in demo mode. Predictions below are simulated to demonstrate the UI layout.</div>", unsafe_allow_html=True)

    col_main, col_gauge = st.columns([3, 2], gap="large")

    with col_main:
        st.markdown("<div class='section-title'>Prediction <span>Result</span></div>", unsafe_allow_html=True)

        if run_btn or True:  # always show a default state
            # ── Compute prediction ──
            if models_loaded:
                cols = models["cols"]
                input_data = build_input(inputs, cols)

                if selected_model == "XGBoost":
                    prob   = models["xgb"].predict_proba(input_data)[0][1]
                    is_anom = prob > THRESHOLD_XGB
                    prob_display = prob

                elif selected_model == "Isolation Forest":
                    pred    = models["iso"].predict(input_data)[0]
                    is_anom = pred == -1
                    prob_display = None

                else:  # Hybrid
                    iso_score  = -models["iso"].decision_function(input_data)
                    input_hyb  = np.column_stack((input_data, iso_score))
                    prob       = models["hybrid"].predict_proba(input_hyb)[0][1]
                    is_anom    = prob > THRESHOLD_HYBRID
                    prob_display = prob
            else:
                # Demo: simulate based on velocity_mdr deviation
                demo_val  = inputs.get("velocity_mdr", 5.0)
                prob_display = min(abs(demo_val - 5.0) / 10.0 + 0.05, 0.99)
                is_anom   = prob_display > 0.3

            # ── Result banner ──
            if is_anom:
                st.markdown(f"""
                <div class='result-anomaly'>
                    <div class='result-title' style='color:var(--warn);'>🔴  ANOMALY DETECTED</div>
                    <div class='result-body'>This coil pass shows deviation beyond the normal operating envelope. Immediate inspection recommended.</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class='result-normal'>
                    <div class='result-title' style='color:var(--ok);'>🟢  NORMAL OPERATION</div>
                    <div class='result-body'>All process parameters are within acceptable range. No anomaly detected.</div>
                </div>
                """, unsafe_allow_html=True)

            # ── Metric cards ──
            rl, rl_class = risk_level(prob_display if prob_display is not None else (0.9 if is_anom else 0.1))
            prob_str = f"{prob_display:.4f}" if prob_display is not None else "N/A"
            model_short = {"XGBoost": "XGB", "Isolation Forest": "IF", "Hybrid (IF + XGBoost)": "HYBRID"}[selected_model]

            st.markdown(f"""
            <div class='metric-row'>
                <div class='metric-card {"bad" if is_anom else "ok"}'>
                    <div class='metric-label'>Status</div>
                    <div class='metric-value' style='font-size:1.4rem;color:{"var(--warn)" if is_anom else "var(--ok)"};'>
                        {"ANOMALY" if is_anom else "NORMAL"}
                    </div>
                </div>
                <div class='metric-card neu'>
                    <div class='metric-label'>Anomaly Probability</div>
                    <div class='metric-value'>{prob_str}</div>
                </div>
                <div class='metric-card'>
                    <div class='metric-label'>Risk Level</div>
                    <div class='metric-value' style='font-size:1.4rem;'>
                        <span class='{rl_class}'>{rl}</span>
                    </div>
                </div>
                <div class='metric-card'>
                    <div class='metric-label'>Model</div>
                    <div class='metric-value' style='font-size:1.2rem;'>{model_short}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    with col_gauge:
        st.markdown("<div class='section-title'>Anomaly <span>Score</span></div>", unsafe_allow_html=True)
        gauge_val = prob_display if prob_display is not None else (0.85 if is_anom else 0.08)

        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=round(gauge_val * 100, 1),
            number={"suffix": "%", "font": {"size": 36, "color": "#1C3A2A", "family": "DM Sans"}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#6B8C6B",
                         "tickfont": {"family": "JetBrains Mono", "size": 10, "color": "#6B8C6B"}},
                "bar": {"color": "#B91C1C" if gauge_val > 0.3 else "#2D6A4F", "thickness": 0.25},
                "bgcolor": "#D5EDD5",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 30],  "color": "#D8F0E4"},
                    {"range": [30, 70], "color": "#FEF3C7"},
                    {"range": [70, 100],"color": "#FEE2E2"},
                ],
                "threshold": {
                    "line": {"color": "#1C3A2A", "width": 2},
                    "thickness": 0.75,
                    "value": 30,
                },
            },
        ))
        fig_gauge.update_layout(
            height=240,
            margin=dict(l=20, r=20, t=30, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            font_family="DM Sans",
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

        # Threshold reference
        st.markdown(f"""
        <div class='info-box'>
            <b>Threshold ({model_short}):</b> {THRESHOLD_XGB if selected_model=="XGBoost" else THRESHOLD_HYBRID if "Hybrid" in selected_model else "N/A (unsupervised)"}<br>
            <b>3-sigma rule applied to:</b> EXIT_THICK_DEVIATION_AVG
        </div>
        """, unsafe_allow_html=True)

    # ── Feature radar ──
    st.markdown("<div class='section-title'>Input <span>Feature Snapshot</span></div>", unsafe_allow_html=True)
    snap_feats = ["velocity_mdr", "tension_en", "dh_entry", "h_exit_ref", "dh_entry_std", "velocity_ex_std"]
    snap_vals  = [inputs.get(f, 0.0) for f in snap_feats]
    snap_labels = ["Vel. MDR", "Tension EN", "DH Entry", "H Exit Ref", "DH Std", "Vel.Ex Std"]

    fig_radar = go.Figure(go.Scatterpolar(
        r=snap_vals + [snap_vals[0]],
        theta=snap_labels + [snap_labels[0]],
        fill="toself",
        fillcolor="rgba(45,106,79,0.12)",
        line=dict(color="#2D6A4F", width=2),
        name="Current",
    ))
    fig_radar.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(visible=True, showticklabels=False, gridcolor="#C8D8C0"),
            angularaxis=dict(gridcolor="#C8D8C0", tickfont=dict(family="JetBrains Mono", size=10, color="#6B8C6B")),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        height=300,
        margin=dict(l=50, r=50, t=30, b=30),
    )
    st.plotly_chart(fig_radar, use_container_width=True)


# ══════════════════════════════════════════════════════════
# TAB 2 — Model Performance
# ══════════════════════════════════════════════════════════
with tab2:
    st.markdown("<div class='section-title'>Model <span>Comparison</span></div>", unsafe_allow_html=True)

    # Static reference metrics from notebook
    perf_data = pd.DataFrame([
        {"Model": "Logistic Regression", "ROC-AUC": 0.81, "Precision": 0.62, "Recall": 0.58, "F1": 0.60},
        {"Model": "Random Forest",       "ROC-AUC": 0.93, "Precision": 0.78, "Recall": 0.74, "F1": 0.76},
        {"Model": "XGBoost",             "ROC-AUC": 0.96, "Precision": 0.84, "Recall": 0.79, "F1": 0.81},
        {"Model": "Isolation Forest",    "ROC-AUC": 0.72, "Precision": 0.55, "Recall": 0.61, "F1": 0.58},
        {"Model": "One-Class SVM",       "ROC-AUC": 0.68, "Precision": 0.50, "Recall": 0.55, "F1": 0.52},
        {"Model": "LOF",                 "ROC-AUC": 0.70, "Precision": 0.52, "Recall": 0.58, "F1": 0.55},
        {"Model": "Hybrid (IF+XGB)",     "ROC-AUC": 0.97, "Precision": 0.86, "Recall": 0.82, "F1": 0.84},
    ])

    col_a, col_b = st.columns(2, gap="large")

    with col_a:
        COLORS = ["#2D6A4F" if m == "Hybrid (IF+XGB)" else "#40916C" if m == "XGBoost" else "#B0C8B0" for m in perf_data["Model"]]
        fig_roc = go.Figure(go.Bar(
            x=perf_data["ROC-AUC"], y=perf_data["Model"],
            orientation="h",
            marker_color=COLORS,
            text=[f"{v:.2f}" for v in perf_data["ROC-AUC"]],
            textposition="outside",
            textfont=dict(family="JetBrains Mono", size=11, color="#1C3A2A"),
        ))
        fig_roc.update_layout(
            title=dict(text="ROC-AUC Score", font=dict(family="DM Sans", size=14, color="#1C3A2A"), x=0),
            xaxis=dict(range=[0, 1.1], showgrid=True, gridcolor="#D4E8D4", tickfont=dict(family="JetBrains Mono", size=10)),
            yaxis=dict(tickfont=dict(family="DM Sans", size=12, color="#1C3A2A")),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=320, margin=dict(l=10, r=60, t=40, b=20),
        )
        st.plotly_chart(fig_roc, use_container_width=True)

    with col_b:
        fig_f1 = go.Figure(go.Bar(
            x=perf_data["F1"], y=perf_data["Model"],
            orientation="h",
            marker_color=COLORS,
            text=[f"{v:.2f}" for v in perf_data["F1"]],
            textposition="outside",
            textfont=dict(family="JetBrains Mono", size=11, color="#1C3A2A"),
        ))
        fig_f1.update_layout(
            title=dict(text="F1 Score", font=dict(family="DM Sans", size=14, color="#1C3A2A"), x=0),
            xaxis=dict(range=[0, 1.1], showgrid=True, gridcolor="#D4E8D4", tickfont=dict(family="JetBrains Mono", size=10)),
            yaxis=dict(tickfont=dict(family="DM Sans", size=12, color="#1C3A2A")),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=320, margin=dict(l=10, r=60, t=40, b=20),
        )
        st.plotly_chart(fig_f1, use_container_width=True)

    # Precision vs Recall scatter
    st.markdown("<div class='section-title'>Precision vs <span>Recall</span></div>", unsafe_allow_html=True)
    fig_pr = go.Figure()
    for _, row in perf_data.iterrows():
        is_best = row["Model"] == "Hybrid (IF+XGB)"
        fig_pr.add_trace(go.Scatter(
            x=[row["Recall"]], y=[row["Precision"]],
            mode="markers+text",
            marker=dict(size=16 if is_best else 11,
                        color="#2D6A4F" if is_best else "#40916C" if row["Model"]=="XGBoost" else "#B0C8B0",
                        symbol="star" if is_best else "circle",
                        line=dict(color="#1C3A2A", width=1)),
            text=[row["Model"]], textposition="top center",
            textfont=dict(family="DM Sans", size=10, color="#1C3A2A"),
            name=row["Model"], showlegend=False,
        ))
    fig_pr.update_layout(
        xaxis=dict(title="Recall", range=[0.4, 1.0], showgrid=True, gridcolor="#D4E8D4", tickfont=dict(family="JetBrains Mono", size=10)),
        yaxis=dict(title="Precision", range=[0.4, 1.0], showgrid=True, gridcolor="#D4E8D4", tickfont=dict(family="JetBrains Mono", size=10)),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=340, margin=dict(l=40, r=20, t=20, b=40),
    )
    st.plotly_chart(fig_pr, use_container_width=True)

    # Robustness table
    st.markdown("<div class='section-title'>Robustness <span>to Noise</span></div>", unsafe_allow_html=True)
    robust_data = pd.DataFrame({
        "Noise Level": [0.01, 0.05, 0.10, 0.20],
        "XGBoost F1":  [0.81, 0.79, 0.74, 0.65],
        "Hybrid F1":   [0.84, 0.82, 0.78, 0.70],
    })
    fig_rob = go.Figure()
    fig_rob.add_trace(go.Scatter(
        x=robust_data["Noise Level"], y=robust_data["XGBoost F1"],
        mode="lines+markers", name="XGBoost",
        line=dict(color="#40916C", width=2), marker=dict(size=8),
    ))
    fig_rob.add_trace(go.Scatter(
        x=robust_data["Noise Level"], y=robust_data["Hybrid F1"],
        mode="lines+markers", name="Hybrid",
        line=dict(color="#2D6A4F", width=2, dash="dash"), marker=dict(size=8, symbol="diamond"),
    ))
    fig_rob.update_layout(
        xaxis=dict(title="Noise Level (σ)", showgrid=True, gridcolor="#D4E8D4", tickfont=dict(family="JetBrains Mono", size=10)),
        yaxis=dict(title="F1 Score", showgrid=True, gridcolor="#D4E8D4", tickfont=dict(family="JetBrains Mono", size=10)),
        legend=dict(font=dict(family="DM Sans", size=11)),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=300, margin=dict(l=40, r=20, t=20, b=40),
    )
    st.plotly_chart(fig_rob, use_container_width=True)


# ══════════════════════════════════════════════════════════
# TAB 3 — SHAP Explainability
# ══════════════════════════════════════════════════════════
with tab3:
    st.markdown("<div class='section-title'>SHAP Feature <span>Importance</span></div>", unsafe_allow_html=True)

    # Static top-feature importance (from notebook analysis)
    shap_static = pd.DataFrame({
        "Feature": ["dh_entry_std", "velocity_mdr", "h_exit_ref", "tension_en",
                    "velocity_ex_std", "rgc_act", "dh_entry", "tension_ex",
                    "velocity_en_std", "REF_TARGET_THICKNESS"],
        "Mean |SHAP|": [0.182, 0.154, 0.131, 0.118, 0.097, 0.084, 0.071, 0.063, 0.052, 0.041],
    }).sort_values("Mean |SHAP|")

    fig_shap = go.Figure(go.Bar(
        x=shap_static["Mean |SHAP|"],
        y=shap_static["Feature"],
        orientation="h",
        marker=dict(
            color=shap_static["Mean |SHAP|"],
            colorscale=[[0, "#C8D8C0"], [0.5, "#40916C"], [1, "#1B5E38"]],
            showscale=False,
        ),
        text=[f"{v:.3f}" for v in shap_static["Mean |SHAP|"]],
        textposition="outside",
        textfont=dict(family="JetBrains Mono", size=10, color="#1C3A2A"),
    ))
    fig_shap.update_layout(
        title=dict(text="Top 10 Features by Mean |SHAP| Value (XGBoost)", font=dict(family="DM Sans", size=14, color="#1C3A2A"), x=0),
        xaxis=dict(showgrid=True, gridcolor="#D4E8D4", tickfont=dict(family="JetBrains Mono", size=10)),
        yaxis=dict(tickfont=dict(family="DM Sans", size=12, color="#1C3A2A")),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=380, margin=dict(l=10, r=80, t=50, b=20),
    )
    st.plotly_chart(fig_shap, use_container_width=True)

    # Per-sample explanation using current inputs
    st.markdown("<div class='section-title'>Per-Sample <span>Root Cause</span></div>", unsafe_allow_html=True)

    if models_loaded:
        try:
            cols = models["cols"]
            input_data = build_input(inputs, cols)
            input_df = pd.DataFrame(input_data, columns=cols)

            explainer = shap.TreeExplainer(models["xgb"])
            shap_vals  = explainer.shap_values(input_df)

            top_n = 10
            feat_names = cols
            shap_arr   = shap_vals[0]
            idx_sorted = np.argsort(np.abs(shap_arr))[::-1][:top_n]

            feats_top = [feat_names[i] for i in idx_sorted]
            vals_top  = [shap_arr[i]   for i in idx_sorted]
            colors_top = ["#B91C1C" if v > 0 else "#2D6A4F" for v in vals_top]

            fig_sample = go.Figure(go.Bar(
                x=vals_top[::-1], y=feats_top[::-1],
                orientation="h",
                marker_color=colors_top[::-1],
                text=[f"{v:+.4f}" for v in vals_top[::-1]],
                textposition="outside",
                textfont=dict(family="JetBrains Mono", size=10),
            ))
            fig_sample.update_layout(
                title=dict(text="SHAP Values for Current Input (red=pushes anomaly, green=pushes normal)", font=dict(family="DM Sans", size=13, color="#1C3A2A"), x=0),
                xaxis=dict(showgrid=True, gridcolor="#D4E8D4", tickfont=dict(family="JetBrains Mono", size=10), zeroline=True, zerolinecolor="#1C3A2A", zerolinewidth=1),
                yaxis=dict(tickfont=dict(family="DM Sans", size=12, color="#1C3A2A")),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=380, margin=dict(l=10, r=80, t=60, b=20),
            )
            st.plotly_chart(fig_sample, use_container_width=True)
        except Exception as e:
            st.markdown(f"<div class='info-box'>SHAP requires model files to be loaded. Error: {e}</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='info-box'>Load model files to enable per-sample SHAP explanations.</div>", unsafe_allow_html=True)

        # Show a demo waterfall
        demo_feats = ["dh_entry_std", "velocity_mdr", "h_exit_ref", "tension_en", "velocity_ex_std"]
        demo_vals  = [0.142, -0.098, 0.071, 0.055, -0.039]
        fig_demo = go.Figure(go.Bar(
            x=demo_vals, y=demo_feats, orientation="h",
            marker_color=["#B91C1C" if v > 0 else "#2D6A4F" for v in demo_vals],
            text=[f"{v:+.3f}" for v in demo_vals], textposition="outside",
            textfont=dict(family="JetBrains Mono", size=11),
        ))
        fig_demo.update_layout(
            title=dict(text="Demo SHAP Values (example sample)", font=dict(family="DM Sans", size=13, color="#6B8C6B"), x=0),
            xaxis=dict(showgrid=True, gridcolor="#D4E8D4", tickfont=dict(family="JetBrains Mono", size=10), zeroline=True, zerolinecolor="#1C3A2A"),
            yaxis=dict(tickfont=dict(family="DM Sans", size=12)),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            height=260, margin=dict(l=10, r=60, t=50, b=20),
        )
        st.plotly_chart(fig_demo, use_container_width=True)


# ══════════════════════════════════════════════════════════
# TAB 4 — About
# ══════════════════════════════════════════════════════════
with tab4:
    col1, col2 = st.columns([3, 2], gap="large")

    with col1:
        st.markdown("<div class='section-title'>Project <span>Overview</span></div>", unsafe_allow_html=True)
        st.markdown("""
        **RollGuard** is an anomaly detection system for cold rolling mill processes.
        It flags coil passes where `EXIT_THICK_DEVIATION_AVG` exceeds the 3-sigma threshold,
        indicating a potential quality defect.

        The system trains and compares six model architectures and deploys the best-performing
        Hybrid model (Isolation Forest + XGBoost) as the primary detector.
        """)

        st.markdown("<div class='section-title'>Methodology <span>Pipeline</span></div>", unsafe_allow_html=True)
        steps = [
            ("1. Anomaly Labelling",    "3-sigma rule on EXIT_THICK_DEVIATION_AVG → binary flag"),
            ("2. Feature Engineering",  "55 process + chemical features; drop metadata columns"),
            ("3. Supervised Models",    "Logistic Regression, Random Forest, XGBoost"),
            ("4. Unsupervised Models",  "Isolation Forest, One-Class SVM, LOF"),
            ("5. Hybrid Model",         "IF anomaly score appended as feature to XGBoost"),
            ("6. Threshold Tuning",     "Cost-matrix optimisation (C_FN=10 × C_FP)"),
            ("7. Explainability",       "SHAP TreeExplainer — global + per-sample"),
        ]
        for title, desc in steps:
            st.markdown(f"<div class='info-box'><b>{title}</b><br>{desc}</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='section-title'>Model <span>Thresholds</span></div>", unsafe_allow_html=True)
        thresh_df = pd.DataFrame({
            "Model":     ["XGBoost", "Hybrid (IF+XGB)", "Isolation Forest"],
            "Threshold": [0.13828209, 0.010662043, "N/A (contamination=0.05)"],
            "Type":      ["Probability", "Probability", "Unsupervised"],
        })
        st.dataframe(thresh_df, hide_index=True, use_container_width=True)

        st.markdown("<div class='section-title'>Feature <span>Groups</span></div>", unsafe_allow_html=True)
        for group, feats in FEATURE_GROUPS.items():
            st.markdown(f"<div class='info-box'><b>{group}</b><br>{', '.join(feats.keys())}</div>", unsafe_allow_html=True)

        st.markdown("<div class='section-title'>Cost <span>Matrix</span></div>", unsafe_allow_html=True)
        st.markdown("""
        <div class='metric-row'>
            <div class='metric-card ok'>
                <div class='metric-label'>False Positive Cost</div>
                <div class='metric-value'>1×</div>
                <div class='metric-sub'>False alarm</div>
            </div>
            <div class='metric-card bad'>
                <div class='metric-label'>False Negative Cost</div>
                <div class='metric-value'>10×</div>
                <div class='metric-sub'>Missed anomaly</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
