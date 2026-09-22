import streamlit as st
import joblib
import numpy as np
import pandas as pd

# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------

st.set_page_config(
    page_title="Disease Prediction System",
    page_icon="🩺",
    layout="centered"
)

# ---------------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------------

st.markdown("""
<style>

.main {
    background-color: #f4fbff;
}

h1 {
    color: #155a75;
    text-align: center;
}

.subtitle {
    text-align: center;
    color: #4b7180;
    font-size: 18px;
    margin-bottom: 25px;
}

.warning-box {
    background-color: #fff7e6;
    border-left: 5px solid #f0a500;
    padding: 15px;
    border-radius: 8px;
    color: #5c4700;
    margin-bottom: 20px;
}

.result-box {
    background-color: #e8f7ef;
    border-left: 5px solid #20a060;
    padding: 18px;
    border-radius: 8px;
    margin-top: 20px;
}

.result-title {
    color: #176b45;
    font-size: 22px;
    font-weight: bold;
}

.info-box {
    background-color: #eef8fc;
    border-left: 5px solid #2495c7;
    padding: 15px;
    border-radius: 8px;
    margin-top: 15px;
}

.footer {
    text-align: center;
    color: #6b8994;
    font-size: 12px;
    margin-top: 35px;
}

</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# LOAD TRAINED MODEL
# ---------------------------------------------------------

@st.cache_resource
def load_model():

    model = joblib.load("best_model.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
    symptom_cols = joblib.load("symptom_list.pkl")

    return model, label_encoder, symptom_cols


# ---------------------------------------------------------
# LOAD DISEASE INFORMATION
# ---------------------------------------------------------

@st.cache_data
def load_extra_information():

    disease_description = {}
    disease_precaution = {}

    # Disease descriptions
    try:

        desc_df = pd.read_csv(
            "symptom_Description.csv",
            header=None,
            names=["Disease", "Description"]
        )

        disease_description = dict(
            zip(
                desc_df["Disease"].astype(str).str.strip(),
                desc_df["Description"].astype(str)
            )
        )

    except Exception:
        pass

    # Disease precautions
    try:

        precaution_df = pd.read_csv(
            "symptom_precaution.csv",
            header=None,
            names=["Disease", "p1", "p2", "p3", "p4"]
        )

        for _, row in precaution_df.iterrows():

            disease = str(row["Disease"]).strip()

            precautions = []

            for column in ["p1", "p2", "p3", "p4"]:

                if pd.notna(row[column]):

                    precaution = str(row[column]).strip()

                    if precaution:
                        precautions.append(precaution)

            disease_precaution[disease] = precautions

    except Exception:
        pass

    return disease_description, disease_precaution


# Load everything

model, label_encoder, symptom_cols = load_model()

disease_description, disease_precaution = load_extra_information()


# ---------------------------------------------------------
# HEADER
# ---------------------------------------------------------

st.title("🩺 Disease Prediction System")

st.markdown(
    '<div class="subtitle">'
    'Machine Learning Based Symptom Classification'
    '</div>',
    unsafe_allow_html=True
)


# ---------------------------------------------------------
# MEDICAL DISCLAIMER
# ---------------------------------------------------------

st.markdown("""
<div class="warning-box">

<b>⚠ Educational Project</b><br><br>

This system is developed as a Machine Learning mini-project.
The prediction is based on patterns learned from the project dataset
and <b>must not be considered a medical diagnosis</b>.

For health concerns, consult a qualified healthcare professional.

</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# SYMPTOM SELECTION
# ---------------------------------------------------------

st.subheader("Select Your Symptoms")

st.write(
    "Choose the symptoms from the list below that best match the input."
)


# Convert internal names into readable names

symptom_display = {
    symptom:
    symptom.replace("_", " ").title()
    for symptom in symptom_cols
}


selected_symptoms = st.multiselect(
    "Symptoms",
    options=symptom_cols,
    format_func=lambda x: symptom_display[x],
    placeholder="Search and select symptoms..."
)


# ---------------------------------------------------------
# PREDICTION BUTTON
# ---------------------------------------------------------

if st.button(
    "🔍 Predict Possible Disease",
    type="primary",
    use_container_width=True
):

    if not selected_symptoms:

        st.warning(
            "Please select at least one symptom before making a prediction."
        )

    else:

        # Create 132-feature binary vector

        input_vector = np.zeros(
            len(symptom_cols),
            dtype=int
        )

        for symptom in selected_symptoms:

            index = symptom_cols.index(symptom)

            input_vector[index] = 1


        # Convert into DataFrame

        input_df = pd.DataFrame(
            [input_vector],
            columns=symptom_cols
        )


        # -------------------------------------------------
        # MODEL PREDICTION
        # -------------------------------------------------

        predicted_encoded = model.predict(input_df)[0]

        predicted_disease = label_encoder.inverse_transform(
            [predicted_encoded]
        )[0]


        # -------------------------------------------------
        # DISPLAY RESULT
        # -------------------------------------------------

        st.markdown(
            f"""
            <div class="result-box">

            <div class="result-title">
            Possible Prediction: {predicted_disease}
            </div>

            <br>

            The selected symptoms most closely match this
            disease class according to the trained Machine Learning model.

            </div>
            """,
            unsafe_allow_html=True
        )


        # -------------------------------------------------
        # DISEASE DESCRIPTION
        # -------------------------------------------------

        disease_key = predicted_disease.strip()

        if disease_key in disease_description:

            st.subheader("📋 Disease Information")

            st.write(
                disease_description[disease_key]
            )


        # -------------------------------------------------
        # PRECAUTIONS
        # -------------------------------------------------

        if disease_key in disease_precaution:

            st.subheader("🛡️ General Precautions")

            for precaution in disease_precaution[disease_key]:

                st.write(
                    f"• {precaution}"
                )


        # -------------------------------------------------
        # FINAL WARNING
        # -------------------------------------------------

        st.markdown("""
        <div class="info-box">

        <b>Important:</b> This result is generated from the
        Machine Learning dataset used in this academic project.
        It is not a clinical diagnosis.

        </div>
        """, unsafe_allow_html=True)


# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------

st.markdown(
    """
    <div class="footer">
    Disease Prediction Using Machine Learning |
    Machine Learning Lab |
    AI & Data Science
    </div>
    """,
    unsafe_allow_html=True
)
