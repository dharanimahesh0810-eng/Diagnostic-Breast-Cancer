"""
app.py
------
Streamlit web interface for the Diagnostic Breast Cancer prediction model.

Run with:
    streamlit run app.py
"""

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# -----------------------------------------------------------------
# Page config
# -----------------------------------------------------------------
st.set_page_config(
    page_title="Breast Cancer Diagnostic Predictor",
    page_icon="🩺",
    layout="wide",
)

# -----------------------------------------------------------------
# Load model + supporting files (cached so it only loads once)
# -----------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("breast_cancer_model.pkl")
    feature_columns = joblib.load("feature_columns.pkl")
    label_classes = joblib.load("label_classes.pkl")  # e.g. ['Benign' 'Malignant']
    return model, feature_columns, label_classes


try:
    model, feature_columns, label_classes = load_artifacts()
except FileNotFoundError:
    st.error(
        "Model files not found. Please run `python train_model.py` first "
        "to generate breast_cancer_model.pkl, feature_columns.pkl and "
        "label_classes.pkl in this folder."
    )
    st.stop()

# -----------------------------------------------------------------
# Default values taken from the sample record used in the notebook
# (a known Malignant case) - used to pre-fill the form
# -----------------------------------------------------------------
default_values = {
    "Mean Radius": 17.99, "Mean Texture": 10.38, "Mean Perimeter": 122.80,
    "Mean Area": 1001.0, "Mean Smoothness": 0.1184, "Mean Compactness": 0.2776,
    "Mean Concavity": 0.3001, "Mean Concave Points": 0.1471, "Mean Symmetry": 0.2419,
    "Mean Fractal Dimension": 0.0787, "Radius SE": 1.095, "Texture SE": 0.9053,
    "Perimeter SE": 8.589, "Area SE": 153.4, "Smoothness SE": 0.006399,
    "Compactness SE": 0.04904, "Concavity SE": 0.05373, "Concave Points SE": 0.01587,
    "Symmetry SE": 0.03003, "Fractal Dimension SE": 0.006193, "Worst Radius": 25.38,
    "Worst Texture": 17.33, "Worst Perimeter": 184.60, "Worst Area": 2019.0,
    "Worst Smoothness": 0.1622, "Worst Compactness": 0.6656, "Worst Concavity": 0.7119,
    "Worst Concave Points": 0.2654, "Worst Symmetry": 0.4601,
    "Worst Fractal Dimension": 0.1189,
}

# -----------------------------------------------------------------
# Header
# -----------------------------------------------------------------
st.title("🩺 Breast Cancer Diagnostic Predictor")
st.markdown(
    "Enter the cell nuclei measurements below to predict whether a tumor "
    "is **Benign** or **Malignant**, using a Logistic Regression model "
    "trained on the Diagnostic Breast Cancer dataset."
)
st.divider()

# -----------------------------------------------------------------
# Input form, grouped into Mean / SE / Worst tabs for readability
# -----------------------------------------------------------------
groups = {
    "Mean values": [c for c in feature_columns if not c.startswith(("Worst", "Radius SE", "Texture SE", "Perimeter SE", "Area SE", "Smoothness SE", "Compactness SE", "Concavity SE", "Concave Points SE", "Symmetry SE", "Fractal Dimension SE"))],
    "Standard Error (SE) values": [c for c in feature_columns if c.endswith("SE")],
    "Worst values": [c for c in feature_columns if c.startswith("Worst")],
}

user_input = {}

with st.form("prediction_form"):
    tabs = st.tabs(list(groups.keys()))
    for tab, (group_name, cols) in zip(tabs, groups.items()):
        with tab:
            grid = st.columns(3)
            for i, col_name in enumerate(cols):
                with grid[i % 3]:
                    user_input[col_name] = st.number_input(
                        col_name,
                        value=float(default_values.get(col_name, 0.0)),
                        format="%.5f",
                    )

    submitted = st.form_submit_button("🔍 Predict Diagnosis", use_container_width=True)

# -----------------------------------------------------------------
# Prediction
# -----------------------------------------------------------------
if submitted:
    input_df = pd.DataFrame([[user_input[c] for c in feature_columns]], columns=feature_columns)

    prediction = model.predict(input_df)[0]
    probabilities = model.predict_proba(input_df)[0]

    predicted_label = label_classes[prediction]
    confidence = probabilities[prediction] * 100

    st.divider()
    if predicted_label.lower() == "malignant":
        st.error(f"### Prediction: {predicted_label}  \nConfidence: {confidence:.2f}%")
    else:
        st.success(f"### Prediction: {predicted_label}  \nConfidence: {confidence:.2f}%")

    prob_df = pd.DataFrame({
        "Diagnosis": label_classes,
        "Probability": probabilities,
    })
    st.bar_chart(prob_df.set_index("Diagnosis"))

    st.caption(
        "⚠️ This tool is for educational/demo purposes only and is **not** "
        "a substitute for professional medical diagnosis."
    )
