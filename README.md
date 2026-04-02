# ⚙️ RollGuard — Rolling Mill Anomaly Detection System

<div align="center">

**A machine learning system for detecting thickness deviation anomalies in cold rolling mill processes**

[![Live App](https://img.shields.io/badge/🚀%20Live%20App-Streamlit-FF4B4B?style=for-the-badge)](https://anomaly-detection-dxpm4mmqdfuvzcz939xdwk.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-Hybrid%20Model-FF6600?style=for-the-badge)](https://xgboost.readthedocs.io)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)

🔗 **[https://anomaly-detection-dxpm4mmqdfuvzcz939xdwk.streamlit.app/](https://anomaly-detection-dxpm4mmqdfuvzcz939xdwk.streamlit.app/)**

</div>

---

## 📌 Table of Contents

1. [Project Overview](#-project-overview)
2. [Problem Statement](#-problem-statement)
3. [Dataset Description](#-dataset-description)
4. [Methodology](#-methodology)
5. [Models & Results](#-models--results)
6. [Streamlit App](#-streamlit-app)
7. [Project Structure](#-project-structure)
8. [Setup & Installation](#-setup--installation)
9. [How to Export Models from Colab](#-how-to-export-models-from-colab)
10. [Deployment](#-deployment)
11. [Key Findings](#-key-findings)
12. [Tech Stack](#-tech-stack)

---

## 🏭 Project Overview

**RollGuard** is an end-to-end anomaly detection pipeline built for cold rolling mill quality control. It identifies coil passes where the thickness deviation exceeds acceptable limits — before defective steel reaches downstream processes.

The system combines supervised learning (XGBoost), unsupervised learning (Isolation Forest), and a hybrid architecture to maximise recall of true anomalies while minimising costly misses. A live Streamlit dashboard allows operators to input process parameters and receive instant anomaly predictions with explainability.

---

## 🎯 Problem Statement

In steel rolling mills, maintaining precise strip thickness is critical. When the output thickness deviates significantly from the target, it signals a process anomaly that can lead to:

- Product rejection and material waste
- Equipment damage from excessive rolling pressure
- Downstream quality failures in stamping or forming

**Goal:** Detect anomalous coil passes from process sensor data and chemical composition measurements, using `EXIT_THICK_DEVIATION_AVG` as the quality signal. Any reading exceeding **mean + 3×standard deviation** is flagged as an anomaly.

---

## 📊 Dataset Description

The dataset contains **60 columns** of rolling mill sensor and lab measurements per coil pass.

| Category | Features | Count |
|----------|----------|-------|
| Metadata | `coil_id`, `pass_nr` | 2 |
| Thickness Reference | `REF_INITIAL_THICKNESS`, `REF_TARGET_THICKNESS`, `h_entry_ref`, `h_exit_ref` | 4 |
| Process Operational | `velocity_mdr/en/ex`, `tension_en/ex`, `dh_entry`, `dm_bur`, `dm_imr1`, `dm_imr2lr`, `dm_imr2m`, `dm_wrbot`, `dm_wrtop`, `rgc_act` | 13 |
| Statistical Features | Entry/exit velocity stats (med, std, max, min), DH entry stats | 12 |
| Chemical Composition | C, SI, MN, S, P, CR, NI, CU, AL, SN, MO, V, TI, NB, W, N, CO, ZR, B, TE, PB, SB, CA, TA, CE, LA | 26 |
| Quality Indicators | `EXIT_THICK_DEVIATION_ABS_AVG`, `EXIT_THICK_DEVIATION_AVG` | 2 |
| **Target (derived)** | `quality_anomaly_flag` (0 = Normal, 1 = Anomaly) | 1 |

**Target construction:** `quality_anomaly_flag = 1` where `EXIT_THICK_DEVIATION_AVG > mean + 3σ`

**Class imbalance:** ~5% anomaly rate, requiring class-weight balancing and careful threshold tuning.

---

## 🔬 Methodology

### 1. Exploratory Data Analysis
- Distribution analysis of `EXIT_THICK_DEVIATION_AVG` — right-skewed with a long anomaly tail
- Pearson correlation with target: process variables (velocity, tension, gap) dominate; chemical composition drives steel grade, not thickness anomaly
- Class imbalance analysis and percentile breakdown to justify the 3-sigma threshold

### 2. Anomaly Labelling (3-Sigma Rule)
```python
deviation_col = 'EXIT_THICK_DEVIATION_AVG'
threshold     = df[deviation_col].mean() + 3 * df[deviation_col].std()
df['quality_anomaly_flag'] = np.where(df[deviation_col] > threshold, 1, 0)
```

### 3. Feature Engineering
- Drop metadata columns: `coil_id`, `pass_nr`
- Drop leakage columns: `EXIT_THICK_DEVIATION_ABS_AVG`, `EXIT_THICK_DEVIATION_AVG`, `class`
- Final feature matrix: **55 features**
- Train/test split: 80/20 stratified

### 4. Supervised Models
All trained on the binary `quality_anomaly_flag` target:

- **Logistic Regression** — scaled features, `class_weight='balanced'`
- **Random Forest** — 200 trees, balanced weights, feature importance extraction
- **XGBoost** — 300 estimators, `scale_pos_weight` for imbalance, threshold tuned via cost matrix

### 5. Unsupervised Models
Trained without labels to learn the normal operating envelope:

- **Isolation Forest** — `contamination=0.05`, 200 trees
- **One-Class SVM** — RBF kernel, `nu=0.05`
- **Local Outlier Factor (LOF)** — 20 neighbours, `novelty=True`

### 6. Hybrid Model (Best Performer)
Isolation Forest anomaly score is appended as an extra feature to XGBoost:
```python
train_iso_scores  = -iso_model.decision_function(X_train)
X_train_hybrid    = np.column_stack((X_train, train_iso_scores))
hybrid_model.fit(X_train_hybrid, y_train)
```
This allows XGBoost to learn from the unsupervised signal, pushing ROC-AUC to **0.97**.

### 7. Threshold Optimisation (Cost-Matrix)
Missing an anomaly costs 10× more than a false alarm:
```python
C_FP = 1    # false positive: unnecessary inspection
C_FN = 10   # false negative: defective product shipped

loss = C_FP * fp + C_FN * fn   # minimise this
```
XGBoost optimal threshold: **0.138** | Hybrid optimal threshold: **0.011**

### 8. Explainability (SHAP)
`shap.TreeExplainer` used for both global feature importance and per-sample root cause analysis.

---

## 📈 Models & Results

### Performance Comparison

| Model | Type | ROC-AUC | Precision | Recall | F1 Score |
|-------|------|---------|-----------|--------|----------|
| Logistic Regression | Supervised | 0.81 | 0.62 | 0.58 | 0.60 |
| Random Forest | Supervised | 0.93 | 0.78 | 0.74 | 0.76 |
| **XGBoost** | Supervised | **0.96** | **0.84** | **0.79** | **0.81** |
| Isolation Forest | Unsupervised | 0.72 | 0.55 | 0.61 | 0.58 |
| One-Class SVM | Unsupervised | 0.68 | 0.50 | 0.55 | 0.52 |
| LOF | Unsupervised | 0.70 | 0.52 | 0.58 | 0.55 |
| **Hybrid (IF + XGB)** | Hybrid | **0.97** | **0.86** | **0.82** | **0.84** |

### Robustness to Sensor Noise

| Noise Level (σ) | XGBoost F1 | Hybrid F1 |
|-----------------|-----------|-----------|
| 0.01 | 0.81 | 0.84 |
| 0.05 | 0.79 | 0.82 |
| 0.10 | 0.74 | 0.78 |
| 0.20 | 0.65 | 0.70 |

The hybrid model degrades more gracefully under noise, making it the recommended production model.

### Top SHAP Features (XGBoost)

| Rank | Feature | Mean \|SHAP\| | Interpretation |
|------|---------|-------------|----------------|
| 1 | `dh_entry_std` | 0.182 | Entry thickness variability — key instability signal |
| 2 | `velocity_mdr` | 0.154 | Mill drive roll speed — affects reduction ratio |
| 3 | `h_exit_ref` | 0.131 | Target exit thickness — reference gap |
| 4 | `tension_en` | 0.118 | Entry strip tension |
| 5 | `velocity_ex_std` | 0.097 | Exit velocity stability |

---

## 🖥️ Streamlit App

**🔗 Live at: [https://anomaly-detection-dxpm4mmqdfuvzcz939xdwk.streamlit.app/](https://anomaly-detection-dxpm4mmqdfuvzcz939xdwk.streamlit.app/)**

The dashboard has four tabs:

### 🏭 Prediction Tab
- Enter process parameters via the sidebar (grouped by category)
- Select model: XGBoost / Isolation Forest / Hybrid
- Get instant result with **anomaly gauge**, **risk level badge** (Low / Medium / High), and **feature radar chart**

### 📈 Model Performance Tab
- ROC-AUC and F1 bar charts for all 7 models
- Precision vs Recall scatter plot
- Robustness curve showing F1 degradation under noise

### 🔍 SHAP Explainability Tab
- Global feature importance bar chart
- Live per-sample SHAP waterfall for the current input — shows exactly which features pushed the prediction toward anomaly or normal

### ℹ️ About Tab
- Full methodology pipeline
- Threshold values and cost matrix
- Feature group descriptions

---

## 📁 Project Structure

```
rollguard/
│
├── app.py                          ← Streamlit dashboard (main entry point)
├── requirements.txt                ← Python dependencies
├── README.md                       ← This file
│
├── models/                         ← Trained model .pkl files
│   ├── final_xgb_model.pkl         ← XGBoost classifier
│   ├── isolation_forest_model.pkl  ← Isolation Forest
│   ├── hybrid_xgb_model.pkl        ← Hybrid XGBoost (IF score as feature)
│   ├── scaler.pkl                  ← StandardScaler for logistic regression
│   └── model_columns.pkl           ← Ordered list of 55 training feature names
│
├── Anomaly_Detection.ipynb     ← Full research notebook (EDA → deployment)
│
└── .streamlit/
    └── config.toml                 ← App theme (ash/warm palette)
```

---

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.10 or higher
- pip

### Steps

```bash
# 1. Clone the repository
git clone [https://github.com/YOUR_USERNAME/rollguard.git](https://github.com/Kavya-630/Anomaly-Detection)
cd rollguard

# 2. (Optional) Create a virtual environment
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add your model files to models/ (see section below)

# 5. Run the app
streamlit run app.py
```

The app will open at `http://localhost:8501`

> **Demo mode:** If `models/` files are missing, the app runs in demo mode with simulated predictions — useful for testing the UI layout.

---

## 📦 How to Export Models from Colab

After running all cells in your notebook, add a new cell and run:

```python
import joblib, os
os.makedirs("models", exist_ok=True)

joblib.dump(xgb_model,    "models/final_xgb_model.pkl")
joblib.dump(iso_model,    "models/isolation_forest_model.pkl")
joblib.dump(hybrid_model, "models/hybrid_xgb_model.pkl")
joblib.dump(scaler,       "models/scaler.pkl")
joblib.dump(X_train.columns.tolist(), "models/model_columns.pkl")

print("✅ All models saved")
```

Then download the `models/` folder from Colab's file browser and place it in your local repo root.

---

## 🚢 Deployment

### Deploy to Streamlit Cloud (Recommended)

1. Push your full repo to GitHub (including `models/` folder)
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in
3. Click **New app** → select your repo
4. Set **Main file path** to `app.py`
5. Click **Deploy**

> **Large model files:** If `.pkl` files exceed GitHub's 100 MB limit, use [Git LFS](https://git-lfs.com/):
> ```bash
> git lfs install
> git lfs track "models/*.pkl"
> git add .gitattributes
> git commit -m "Track model files with LFS"
> ```

### Local `.streamlit/config.toml`
```toml
[theme]
base                = "light"
primaryColor        = "#C0390F"
backgroundColor     = "#F0EDE8"
secondaryBackgroundColor = "#E8E4DE"
textColor           = "#1C1915"
font                = "sans serif"
```

---

## 💡 Key Findings

1. **`EXIT_THICK_DEVIATION_AVG` is the correct anomaly target** — the `class` column in the original dataset represents steel grade (driven by AL and SI chemistry), not process quality.

2. **Process parameters drive anomalies, not chemistry** — entry thickness variability (`dh_entry_std`), mill speed (`velocity_mdr`), and tension (`tension_en`) are the dominant SHAP features.

3. **The Hybrid model outperforms all baselines** — combining Isolation Forest's unsupervised signal with XGBoost's supervised classifier achieves ROC-AUC of 0.97 and is more robust to sensor noise.

4. **Threshold tuning is critical** — the default 0.5 threshold misses too many anomalies. The cost-optimised threshold (0.138 for XGB, 0.011 for Hybrid) reflects the industrial reality that a missed defect costs 10× more than a false alarm.

5. **Only 5% of passes are anomalous** — class imbalance requires `scale_pos_weight` in XGBoost and stratified splitting.

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| `pandas` / `numpy` | Data manipulation |
| `scikit-learn` | Preprocessing, RF, IF, SVM, LOF, metrics |
| `xgboost` | Gradient boosting classifier |
| `shap` | Model explainability |
| `streamlit` | Web dashboard |
| `plotly` | Interactive charts (gauge, radar, bar, scatter) |
| `joblib` | Model serialisation |
| `matplotlib` | SHAP static plots |

---

## 👤 Author

Built as part of an industrial ML project on rolling mill quality control.

---

<div align="center">
<b>⚙️ RollGuard</b> · Rolling Mill Anomaly Detection<br>
<a href="https://anomaly-detection-dxpm4mmqdfuvzcz939xdwk.streamlit.app/">Live App</a>
</div>
