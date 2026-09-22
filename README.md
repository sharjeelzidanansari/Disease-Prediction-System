# Disease Prediction System (Symptoms → Disease)

Mini Project — 5th Sem ML Lab, AI & Data Science, Anjuman College of Engineering and Technology

## What it does
Predicts one of **41 diseases** from a set of **132 possible symptoms** using
supervised classification.

## Dataset
- `Training.csv` — 4920 rows, 132 binary symptom columns + `prognosis` (disease) column
- `Testing.csv` — 42 rows, held out for evaluation
- `symptom_Description.csv` — short description of each disease (used in the demo output)
- `symptom_precaution.csv` — 4 precautions per disease (used in the demo output)
- `Symptom_severity.csv` — severity weight per symptom (included, not used in the base model — optional extension)

Source: the well-known Kaggle dataset *"Disease Prediction Using Machine Learning"*.

## Files
| File | Purpose |
|---|---|
| `disease_prediction.py` | Main script: loads data, trains 3 models, evaluates, saves the best one + charts |
| `demo.py` | Reusable interactive CLI — type symptoms, get a disease prediction |
| `best_model.pkl`, `label_encoder.pkl`, `symptom_list.pkl` | Saved trained model + supporting objects (created after running the main script) |
| `model_comparison.png` | Bar chart comparing accuracy of Decision Tree / Random Forest / Naive Bayes |
| `confusion_matrix.png` | Confusion matrix heatmap of the best model on the test set |

## How to run
```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib

python disease_prediction.py   # trains models, prints accuracy, saves everything
python demo.py                 # interactive: type symptoms, get prediction
```

## Results
All three models (Decision Tree, Random Forest, Naive Bayes) scored **100% accuracy**
on `Testing.csv`. This is expected and well documented for this dataset — the 132
symptoms map to diseases in an almost deterministic way (each disease has a fixed,
non-overlapping symptom signature in the source data), and the test set has only
one example per disease. This is normal for this specific dataset; it does **not**
mean the code is broken.

**For your project report**, you can honestly say:
- Decision Tree, Random Forest, and Naive Bayes were compared
- All achieved 100% test accuracy (cite: this is a known, clean, well-separated dataset)
- Random Forest is generally the more "production-safe" choice for this kind of
  problem (less prone to overfitting quirks than a single Decision Tree, even
  though both score equally here) — mention this if your evaluator asks "why not
  just pick Decision Tree since it's simpler?"

## How the prediction works
1. User types symptom names (comma-separated).
2. Each symptom is converted into a 0/1 feature vector across all 132 known symptoms.
3. The trained model predicts the disease class.
4. The predicted disease's description and precautions are looked up and shown.

## Possible extensions (mention in viva if asked "what would you add next?")
- A simple web UI using Streamlit instead of the CLI
- Use `Symptom_severity.csv` to weight symptoms by severity before prediction
- Add a confidence threshold: if confidence is low, ask the user for more symptoms
  instead of guessing
