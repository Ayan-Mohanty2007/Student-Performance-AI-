# Student Performance AI — Prediction & Learning Recommendation System

A polished Streamlit prototype for a 10-day AI & ML major project. It predicts a broad performance category and provides transparent, rule-based study suggestions.

## Features
- Interactive Streamlit dashboard
- Random Forest classifier with preprocessing pipeline
- 80/20 stratified train/test split
- Synthetic dataset with documented missing values
- Dataset exploration, class distribution and confusion matrix
- Rule-based recommendations separated from model predictions
- Saved model pipeline and evaluation outputs

## Quick start
Use Python 3.11 or 3.12 recommended (avoid cutting-edge Python versions until all dependencies support them).

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

If your shell uses `python3`, replace `python` with `python3`.

## Project structure
```text
student_performance_ai_ml_project/
├── app.py
├── requirements.txt
├── README.md
├── data/
│   └── student_performance_synthetic.csv
├── models/
│   └── performance_pipeline.joblib
├── reports/
│   ├── classification_report.csv
│   ├── confusion_matrix.png
│   └── performance_distribution.png
└── docs/
    └── project_report.md
```

## Dataset and ethics
The CSV is a synthetic educational dataset generated for this project (n=1200); it does not contain real student records. The target categories are created from a deliberately simple, noisy scoring rule using study hours, attendance, previous marks, assignment completion, sleep, internet access, and participation. Therefore, this is a demonstration of the ML workflow—not evidence of real-world student prediction ability. The model output must not be used for high-stakes academic decisions.

## Model evaluation
- Test accuracy: 0.613
- Test macro-F1: 0.507
- Majority-class baseline accuracy: 0.542

Metrics are from a stratified 20% held-out split with random_state=42. They describe performance on synthetic data only.

## Reproduce training
The fitted pipeline is included in `models/performance_pipeline.joblib`. For a full reproducible retraining workflow, use the same feature columns and split strategy documented in `docs/project_report.md`; never include the target column as an input feature.

## Deployment
1. Push the contents of this folder to a GitHub repository.
2. On Streamlit Community Cloud, create a new app and select `app.py` as the entry point.
3. Ensure `requirements.txt`, `data/`, `models/`, and `reports/` are committed.
4. Test the deployed app from a clean browser session.

## Suggested demo video (2–4 minutes)
1. Introduce the problem and ethics note.
2. Show dataset preview and class distribution.
3. Change inputs and run a prediction.
4. Explain how recommendations are rule-based.
5. Show model evaluation and limitations.
