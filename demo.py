"""
Standalone demo - run this AFTER disease_prediction.py has been run once
(it needs best_model.pkl, label_encoder.pkl, symptom_list.pkl to exist).

Usage:
    python demo.py
"""

import joblib
import numpy as np
import pandas as pd

model = joblib.load("best_model.pkl")
le = joblib.load("label_encoder.pkl")
symptom_cols = joblib.load("symptom_list.pkl")

# optional extra info
try:
    desc_df = pd.read_csv("symptom_Description.csv", header=None, names=["Disease", "Description"])
    disease_description = dict(zip(desc_df["Disease"].str.strip(), desc_df["Description"]))
except FileNotFoundError:
    disease_description = {}

try:
    prec_df = pd.read_csv("symptom_precaution.csv", header=None, names=["Disease", "p1", "p2", "p3", "p4"])
    disease_precaution = {
        row["Disease"].strip(): [row[c] for c in ["p1", "p2", "p3", "p4"] if pd.notna(row[c])]
        for _, row in prec_df.iterrows()
    }
except FileNotFoundError:
    disease_precaution = {}


def predict_disease(user_symptoms):
    input_vector = np.zeros(len(symptom_cols))
    matched, unmatched = [], []

    for s in user_symptoms:
        s_clean = s.strip().lower().replace(" ", "_")
        if s_clean in symptom_cols:
            input_vector[symptom_cols.index(s_clean)] = 1
            matched.append(s_clean)
        else:
            unmatched.append(s)

    input_df = pd.DataFrame([input_vector], columns=symptom_cols)
    pred_encoded = model.predict(input_df)[0]
    disease = le.inverse_transform([pred_encoded])[0]

    confidence = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(input_df)[0]
        confidence = round(max(proba) * 100, 2)

    return disease, confidence, matched, unmatched


if __name__ == "__main__":
    print("=" * 60)
    print(" DISEASE PREDICTION SYSTEM - Interactive Demo")
    print("=" * 60)
    print("Type symptoms separated by commas, e.g.:")
    print("  itching, skin_rash, high_fever")
    print("Type 'list' to see all valid symptom names, or 'quit' to exit.\n")

    while True:
        raw = input("Enter symptoms: ").strip()
        if raw.lower() == "quit":
            print("Goodbye!")
            break
        if raw.lower() == "list":
            print(", ".join(symptom_cols))
            continue
        if not raw:
            continue

        symptoms = [s.strip() for s in raw.split(",") if s.strip()]
        disease, confidence, matched, unmatched = predict_disease(symptoms)

        print(f"\nMatched symptoms   : {matched if matched else 'None recognized'}")
        if unmatched:
            print(f"Unrecognized input : {unmatched}  (type 'list' to see valid names)")

        print(f"\n>> Predicted disease: {disease}")
        if confidence is not None:
            print(f">> Confidence       : {confidence}%")

        if disease.strip() in disease_description:
            print(f">> Description      : {disease_description[disease.strip()]}")
        if disease.strip() in disease_precaution:
            print(f">> Precautions      : {', '.join(disease_precaution[disease.strip()])}")

        print("\n(Note: this is a college mini-project demo, not medical advice.)\n")
