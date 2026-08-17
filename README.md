# Neonatal ARDS Prediction Model

This repository accompanies the manuscript:

> **Predictive value of maternal prenatal blood immune-inflammatory markers for neonatal acute respiratory distress syndrome based on machine learning**

## Project overview

This study develops and internally evaluates prediction models for distinguishing neonatal acute respiratory distress syndrome (ARDS) from other respiratory conditions among mechanically ventilated late-preterm and term neonates with respiratory distress.

The primary model is a six-predictor random forest using:

- antenatal infection within 14 days before delivery;
- antenatal corticosteroid use;
- maternal age;
- gestational hypothyroidism;
- gestational diabetes mellitus; and
- systemic immune-inflammation index (SII).

The model was developed for retrospective research within this selected neonatal respiratory-distress cohort. It is **not** an antenatal screening tool for unselected pregnancies, has not undergone external validation or prospective clinical-impact evaluation, and must not be used for clinical decision-making.

## Repository status

The associated manuscript is currently under peer review.

To preserve a stable review record, the complete analysis code, model-development notebooks, model artifacts, and interface source code will be released in this repository after the article has been accepted for publication. The release will include documentation sufficient to reproduce the reported analyses and regenerate the main tables and figures.

No patient-level clinical data will be placed in this public repository. Access to de-identified data may be considered for non-commercial academic research on reasonable request, subject to ethics approval and an appropriate data-use agreement.

## Planned software requirements

The analysis and research interface use Python and the following principal packages:

- Python 3.10 or later
- JupyterLab / Jupyter Notebook
- NumPy
- pandas
- SciPy
- scikit-learn
- XGBoost
- SHAP
- Matplotlib
- openpyxl
- joblib
- Streamlit

A version-pinned `requirements.txt` or equivalent environment file will be provided with the post-acceptance code release.

## Planned post-acceptance release

The repository is expected to include:

- data-quality-control and feature-screening workflows;
- model training and internal-validation code;
- bootstrap calibration and performance analyses;
- benchmark logistic-regression analyses;
- scripts used to generate manuscript tables and figures;
- the locked research model and metadata;
- source code for the research-use web interface; and
- installation and reproducibility instructions.

## Citation

Citation details will be added after publication.

## Disclaimer

This repository and the associated model are provided for research and reproducibility purposes only. They are not a medical device and must not be used to diagnose disease, estimate transportable individual risk, guide antenatal corticosteroid administration, or replace clinical judgment.
