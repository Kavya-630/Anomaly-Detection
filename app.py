import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Steel Quality Anomaly Detector",
    page_icon="📊",
    layout="wide"
)

# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_model():
    return joblib.load("final_xgb_model.pkl")

model = load_model()

# Get expected feature names from trained model
expected_features = model.get_booster().feature_names

# -----------------------------
# UI Design
# -----------------------------
st.markdown("""
    <style>
    .main {
        background-color: #f4f8fb;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        border-radius: 10px;
        height: 3em;
        width: 100%;
    }
    </style>
""", unsafe_allow_html=True)

st.title("📊 Steel Quality Anomaly Detection")
st.markdown("### Enter Key Process Parameters")

col1, col2, col3 = st.columns(3)

with col1:
    h_exit_ref = st.number_input("Exit Height Ref", value=1.0)
    velocity_mdr = st.number_input("Velocity MDR", value=0.7)
    tension_en = st.number_input("Tension Entry", value=100.0)

with col2:
    velocity_en = st.number_input("Velocity Entry", value=1.0)
    velocity_ex = st.number_input("Velocity Exit", value=1.0)

with col3:
    REF_INITIAL_THICKNESS = st.number_input("Initial Thickness", value=4.0)
    REF_TARGET_THICKNESS = st.number_input("Target Thickness", value=4.0)

# -----------------------------
# Predict Button
# -----------------------------
if st.button("🔍 Predict Anomaly"):

    # Create full feature dictionary initialized with 0
    input_dict = {feature: 0 for feature in expected_features}

    # Update with user-provided values
    input_dict.update({
        "h_exit_ref": h_exit_ref,
        "velocity_mdr": velocity_mdr,
        "tension_en": tension_en,
        "velocity_en": velocity_en,
        "velocity_ex": velocity_ex,
        "REF_INITIAL_THICKNESS": REF_INITIAL_THICKNESS,
        "REF_TARGET_THICKNESS": REF_TARGET_THICKNESS
    })

    # Convert to DataFrame
    input_data = pd.DataFrame([input_dict])

    # Ensure correct column order
    input_data = input_data[expected_features]

    # Predict probability
    probability = model.predict_proba(input_data)[0][1]

    threshold = 0.50
    prediction = 1 if probability > threshold else 0

    st.markdown("---")

    if prediction == 1:
        st.error(f"⚠️ Anomaly Detected! Probability: {probability:.2f}")
    else:
        st.success(f"✅ Normal Quality. Probability: {probability:.2f}")

    st.progress(float(probability))
