import streamlit as st
import joblib
import numpy as np
import pandas as pd

st.set_page_config(
    page_title="Disease Prediction System",
    page_icon="🩺",
    layout="centered"
)

@st.cache_resource
def load_model():
    model = joblib.load("best_model.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
    symptom_cols = joblib.load("symptom_list.pkl")
    return model, label_encoder, symptom_cols

@st.cache_data
def load_extra_info():
    description = {}
    precautions = {}

    try:
        desc_df = pd.read_csv(
            "symptom_Description.csv",
            header=None,
            names=["Disease", "Description"]
        )
        description = dict(
            zip(desc_df["Disease"].astype(str).str.strip(),
                desc_df["Description"].astype(str))
        )
    except Exception:
        pass

    try:
        prec_df = pd.read_csv(
            "symptom_precaution.csv",
            header=None,
            names=["Disease", "p1", "p2", "p3", "p4"]
        )
        for _, row in prec_df.iterrows():
            disease = str(row["Disease"]).strip()
            precautions[disease] = [
                str(row[c]).strip()
                for c in ["p1", "p2", "p3", "p4"]
                if pd.notna(row[c]) and str(row[c]).strip()
            ]
    except Exception:
        pass

    return description, precautions

model, label_encoder, symptom_cols = load_model()
disease_description, disease_precaution = load_extra_info()

st.title("🩺 Disease Prediction System")
st.write("### Predict a possible disease from selected symptoms")
st.info(
    "Educational machine-learning mini-project. "
    "This is not a medical diagnosis and should not replace professional medical advice."
)

# Convert internal symptom names such as skin_rash into readable labels.
symptom_display = {
    s: s.replace("_", " ").replace("  ", " ").title()
    for s in symptom_cols
}

selected = st.multiselect(
    "Select the symptoms you are experiencing:",
    options=symptom_cols,
    format_func=lambda x: symptom_display[x],
    placeholder="Choose one or more symptoms..."
)

if st.button("🔍 Predict Disease", type="primary", use_container_width=True):
    if not selected:
        st.warning("Please select at least one symptom.")
    else:
        input_vector = np.zeros(len(symptom_cols), dtype=int)

        for symptom in selected:
            input_vector[symptom_cols.index(symptom)] = 1

        input_df = pd.DataFrame([input_vector], columns=symptom_cols)

        pred_encoded = model.predict(input_df)[0]
        disease = label_encoder.inverse_transform([pred_encoded])[0]

        confidence = None
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_df)[0]
            confidence = float(max(probabilities) * 100)

        st.success(f"Predicted Disease: **{disease}**")

        if confidence is not None:
            st.metric("Model Confidence", f"{confidence:.2f}%")

        if disease.strip() in disease_description:
            st.subheader("📋 Description")
            st.write(disease_description[disease.strip()])

        if disease.strip() in disease_precaution:
            st.subheader("🛡️ General Precautions")
            for precaution in disease_precaution[disease.strip()]:
                st.write(f"• {precaution}")

st.divider()
st.caption("Mini Project | Machine Learning Lab | AI & Data Science")
