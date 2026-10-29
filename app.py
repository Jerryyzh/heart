
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Load the fitted pipeline
model = joblib.load("heart_disease_model.joblib")

st.set_page_config(
    page_title="Sydän appi",
    page_icon="❤️",
    layout="centered"
)

st.title("Sydän appi")
st.write(
    "Enter the patient characteristics below to obtain "
    "a model-estimated probability of heart disease."
)

st.warning(
    "Educational project only. This model is not a "
    "diagnostic tool and must not be used to make "
    "medical decisions."
)

with st.form("prediction_form"):

    st.subheader("Patient information")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input(
            "Age (years)", 28, 100, 50
        )
        sex = st.selectbox(
            "Sex", ["Male", "Female"]
        )
        cp = st.selectbox(
            "Chest pain type",
            [
                "typical angina",
                "atypical angina",
                "non-anginal",
                "asymptomatic"
            ]
        )
        trestbps = st.number_input(
            "Resting blood pressure (mm Hg)",
            min_value=0, max_value=250, value=120
        )
        chol = st.number_input(
            "Cholesterol (mg/dL)",
            min_value=0, max_value=700, value=240
        )
        fbs = st.selectbox(
            "Fasting blood sugar > 120 mg/dL",
            [False, True]
        )
        restecg = st.selectbox(
            "Resting ECG",
            ["normal", "ST-T abnormality", "lv hypertrophy"]
        )

    with col2:
        thalch = st.number_input(
            "Maximum heart rate achieved",
            min_value=0, max_value=250, value=150
        )
        exang = st.selectbox(
            "Exercise-induced angina", [False, True]
        )
        oldpeak = st.number_input(
            "ST depression (oldpeak)",
            min_value=-5.0, max_value=10.0,
            value=0.0, step=0.1
        )
        slope = st.selectbox(
            "ST segment slope",
            ["upsloping", "flat", "downsloping"]
        )
        ca = st.selectbox(
            "Number of major vessels (ca)",
            [0, 1, 2, 3, "Unknown"]
        )
        thal = st.selectbox(
            "Thalassemia (thal)",
            [
                "normal",
                "fixed defect",
                "reversable defect",
                "Unknown"
            ]
        )

    submitted = st.form_submit_button(
        "Predict probability",
        type="primary"
    )

if submitted:
    # Match the feature names used when training the model
    patient = pd.DataFrame([{
        "age": age,
        "sex": sex,
        "cp": cp,
        "trestbps": np.nan if trestbps == 0 else trestbps,
        "chol": np.nan if chol == 0 else chol,
        "fbs": fbs,
        "restecg": restecg,
        "thalch": thalch,
        "exang": exang,
        "oldpeak": oldpeak,
        "slope": slope,
        "ca": np.nan if ca == "Unknown" else ca,
        "thal": np.nan if thal == "Unknown" else thal
    }])

    probability = model.predict_proba(patient)[0, 1]

    st.subheader("Prediction result")
    st.metric(
        "Model-estimated probability of heart disease",
        f"{probability:.1%}"
    )

    st.caption(
        "This is a model output, not an individual's "
        "clinically established risk or a diagnosis."
    )
