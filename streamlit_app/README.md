# Neonatal ARDS Research Classifier — Streamlit App

Interactive research app that serves the **final locked random-forest model**
for distinguishing neonatal ARDS from pooled non-ARDS respiratory conditions
within the study cohort of mechanically ventilated late-preterm and term
neonates with respiratory distress. Its six maternal predictors were available
immediately before delivery.

This is **not an antenatal screening tool** for unselected pregnancies. The
target-population restriction is established from the neonatal clinical course,
so the interface is a retrospective research implementation rather than a
prospectively validated clinical workflow.

## Model summary

| Item | Value |
|------|-------|
| Algorithm | Random forest (`n_estimators=500`, `min_samples_leaf=5`, `max_features="sqrt"`, `random_state=42`) |
| Feature set | `Clinical_plus_Inflammation` (6 predictors) |
| Predictors | AI (antenatal infection within 14 days before delivery), ACS (antenatal corticosteroids), MA (maternal age), GH (gestational hypothyroidism), GDM (gestational diabetes), SII (systemic immune-inflammation index) |
| Primary threshold | Youden-optimal = **0.424** |
| Internal test AUC / AP / Brier | 0.827 / 0.828 / 0.175 |
| Internal test sensitivity / specificity | 0.79 / 0.78 |

> Research prototype only. Internal evaluation from the same cohort, uncertain
> calibration, and **no external validation** or prospective clinical-impact
> evaluation. The model score is conditional on the study case mix and is not a
> transportable absolute risk estimate, a medical device, or a substitute for
> clinical judgement.

## Files

```
streamlit_app/
├── app.py                              # Streamlit web app
├── train_export_model.py               # Re-train + export the model artifacts
├── ards_rf_model.joblib                # Fitted sklearn Pipeline (generated)
├── model_metadata.json                 # Features, threshold, metrics (generated)
├── dataset_with_train_test_split.csv   # Training data (local copy)
├── requirements.txt
└── README.md
```

## 1. Install

```bash
pip install -r requirements.txt
```

## 2. (Re)generate the model artifact

Already generated, but to reproduce:

```bash
python train_export_model.py
```

This fits the pipeline on the **training split only** and writes
`ards_rf_model.joblib` + `model_metadata.json`.

## 3. Run locally

```bash
streamlit run app.py
```

Open the URL shown in the terminal (default http://localhost:8501).

## 4. Deploy to Streamlit Community Cloud

1. Push this `streamlit_app/` folder to a GitHub repository.
2. Go to https://share.streamlit.io → **New app**.
3. Select the repo/branch and set **Main file path** to `streamlit_app/app.py`
   (or `app.py` if this folder is the repo root).
4. Streamlit Cloud installs `requirements.txt` automatically and launches the app.

> Important: keep `scikit-learn==1.7.2` pinned in `requirements.txt` so the
> saved model loads without version-mismatch warnings.

## Notes

- `GH` is the dataset/model code for **gestational hypothyroidism**. Gestational
  hypertension (`CHT`) is not an input to the deployed six-predictor model.
- SII can be entered directly or computed from PLT × NEU / LYM inside the app.
- Each prediction shows an individual **SHAP contribution** chart explaining
  which predictors pushed the risk up or down.
