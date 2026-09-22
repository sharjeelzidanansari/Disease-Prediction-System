"""
Disease Prediction System (Symptoms -> Disease)
------------------------------------------------
Mini Project - ML Lab (5th Sem, AI & Data Science)

Dataset : Training.csv / Testing.csv
          132 binary symptom columns + 1 target column ('prognosis')
          41 possible diseases

Pipeline:
  1. Load data
  2. Encode target labels (disease names -> numbers)
  3. Train 3 models: Decision Tree, Random Forest, Naive Bayes
  4. Evaluate each on the held-out Testing.csv
  5. Save the best model to disk
  6. Interactive prediction: user selects symptoms -> model predicts disease
     (+ description + precautions, if available)
"""

import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import matplotlib
matplotlib.use("Agg")          # no GUI needed, we just save the plot to a file
import matplotlib.pyplot as plt
import seaborn as sns


# --------------------------------------------------------------------------
# 1. LOAD DATA
# --------------------------------------------------------------------------
train_df = pd.read_csv("Training.csv")
test_df = pd.read_csv("Testing.csv")

# Training.csv sometimes ships with a stray unnamed empty column at the end -
# drop it if present, so it never gets treated as a feature.
train_df = train_df.loc[:, ~train_df.columns.str.contains("^Unnamed")]
test_df = test_df.loc[:, ~test_df.columns.str.contains("^Unnamed")]

symptom_cols = [c for c in train_df.columns if c != "prognosis"]

X_train = train_df[symptom_cols]
y_train_raw = train_df["prognosis"]

X_test = test_df[symptom_cols]
y_test_raw = test_df["prognosis"]

print(f"Training samples : {X_train.shape[0]}")
print(f"Testing samples  : {X_test.shape[0]}")
print(f"Symptom features : {X_train.shape[1]}")
print(f"Diseases (classes): {y_train_raw.nunique()}\n")


# --------------------------------------------------------------------------
# 2. ENCODE LABELS  (disease name -> integer)
# --------------------------------------------------------------------------
le = LabelEncoder()
y_train = le.fit_transform(y_train_raw)
y_test = le.transform(y_test_raw)


# --------------------------------------------------------------------------
# 3. TRAIN MULTIPLE MODELS
# --------------------------------------------------------------------------
models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Naive Bayes": GaussianNB(),
}

results = {}
trained_models = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    results[name] = acc
    trained_models[name] = model
    print(f"{name:<15} -> Test Accuracy: {acc * 100:.2f}%")

print()


# --------------------------------------------------------------------------
# 4. PICK BEST MODEL & SHOW DETAILED REPORT
# --------------------------------------------------------------------------
best_name = max(results, key=results.get)
best_model = trained_models[best_name]
best_preds = best_model.predict(X_test)

print(f"Best model: {best_name} ({results[best_name] * 100:.2f}% accuracy)\n")
print("Classification report on Testing.csv:")
print(classification_report(y_test, best_preds, target_names=le.classes_, zero_division=0))


# --------------------------------------------------------------------------
# 5. SAVE ACCURACY COMPARISON CHART + CONFUSION MATRIX (for the report)
# --------------------------------------------------------------------------
plt.figure(figsize=(6, 4))
plt.bar(results.keys(), [v * 100 for v in results.values()], color=["#4C72B0", "#55A868", "#C44E52"])
plt.ylabel("Test Accuracy (%)")
plt.title("Model Comparison - Disease Prediction")
plt.ylim(0, 105)
for i, (k, v) in enumerate(results.items()):
    plt.text(i, v * 100 + 1, f"{v*100:.1f}%", ha="center")
plt.tight_layout()
plt.savefig("model_comparison.png", dpi=150)
plt.close()

cm = confusion_matrix(y_test, best_preds)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, cmap="Blues", cbar=True)
plt.title(f"Confusion Matrix - {best_name}")
plt.xlabel("Predicted disease (encoded)")
plt.ylabel("Actual disease (encoded)")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.close()

print("Saved: model_comparison.png, confusion_matrix.png")


# --------------------------------------------------------------------------
# 6. SAVE THE BEST MODEL + LABEL ENCODER + SYMPTOM LIST (for reuse / app)
# --------------------------------------------------------------------------
joblib.dump(best_model, "best_model.pkl")
joblib.dump(le, "label_encoder.pkl")
joblib.dump(symptom_cols, "symptom_list.pkl")
print("Saved: best_model.pkl, label_encoder.pkl, symptom_list.pkl\n")


# --------------------------------------------------------------------------
# 7. HELPER: load extra info (description / precaution), if available
# --------------------------------------------------------------------------
def load_extra_info():
    desc_map, prec_map = {}, {}
    try:
        desc_df = pd.read_csv("symptom_Description.csv", header=None,
                               names=["Disease", "Description"])
        desc_map = dict(zip(desc_df["Disease"].str.strip(), desc_df["Description"]))
    except FileNotFoundError:
        pass
    try:
        prec_df = pd.read_csv("symptom_precaution.csv", header=None,
                               names=["Disease", "p1", "p2", "p3", "p4"])
        for _, row in prec_df.iterrows():
            prec_map[row["Disease"].strip()] = [row[c] for c in ["p1", "p2", "p3", "p4"] if pd.notna(row[c])]
    except FileNotFoundError:
        pass
    return desc_map, prec_map


disease_description, disease_precaution = load_extra_info()


# --------------------------------------------------------------------------
# 8. PREDICTION FUNCTION (reusable, also used by the CLI demo below)
# --------------------------------------------------------------------------
def predict_disease(user_symptoms, model=best_model):
    """
    user_symptoms : list of symptom strings (must match column names in symptom_cols)
    returns       : (predicted_disease, confidence_or_None, matched_symptoms, unmatched_symptoms)
    """
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


# --------------------------------------------------------------------------
# 9. SIMPLE INTERACTIVE CLI DEMO
# --------------------------------------------------------------------------
def run_cli_demo():
    print("=" * 60)
    print(" DISEASE PREDICTION SYSTEM - Interactive Demo")
    print("=" * 60)
    print(f"Model in use: {best_name} ({results[best_name]*100:.2f}% test accuracy)\n")
    print("Type your symptoms separated by commas (underscore style like")
    print("'skin_rash, joint_pain, high_fever' or close English words).")
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


if __name__ == "__main__":
    # Quick sanity-check prediction (non-interactive) so the script always
    # shows one worked example even when run non-interactively / for grading.
    example_symptoms = ["itching", "skin_rash", "nodal_skin_eruptions"]
    d, c, m, u = predict_disease(example_symptoms)
    print("Example prediction")
    print(f"  Input symptoms : {example_symptoms}")
    print(f"  Predicted      : {d}  (confidence: {c}%)\n")

    # Uncomment the line below to try the live interactive demo in a terminal:
    # run_cli_demo()
