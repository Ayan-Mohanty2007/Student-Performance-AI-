# AI-Powered Student Performance Prediction & Learning Recommendation System

## Abstract
This project demonstrates an end-to-end machine-learning workflow that predicts a broad student performance category from study habits and academic indicators. A Streamlit interface lets users enter a learning profile, view the estimated category, inspect class probabilities, and receive simple study suggestions. The project uses a synthetic dataset, so results are educational demonstrations rather than validated claims about real students.

## 1. Problem Statement
Academic support teams may find it difficult to review multiple indicators manually. This prototype explores whether a machine-learning workflow can combine attendance, study hours, previous marks, assignment completion, sleep, internet access and class participation to estimate a broad performance category.

## 2. Objectives
- Prepare a dataset and handle missing values.
- Explore the data using summary statistics and visualizations.
- Train a baseline and a Random Forest classifier.
- Evaluate predictions on held-out test data.
- Build a Streamlit dashboard with predictions and study suggestions.
- Document limitations and responsible-use principles.

## 3. Dataset Description
The project contains 1200 synthetically generated records. Features include study hours per week, attendance percentage, previous exam marks, assignment completion percentage, sleep hours per day, internet access and class participation. The target is `performance_category`, with categories Needs Support, Developing and On Track. Small amounts of missingness were deliberately added to selected input columns to exercise preprocessing. No real or identifiable student data is used.

The labels were generated using a noisy formula combining the input variables, then grouped into three categories. This makes the dataset convenient for a reproducible demonstration but creates a major limitation: the learned relationships reflect the synthetic formula and should not be assumed to represent actual educational outcomes.

## 4. Methodology
The target was separated from the input features before splitting. An 80/20 stratified train/test split was created with random state 42. Numerical features use median imputation and standard scaling. Categorical features use most-frequent imputation and one-hot encoding. These transformations are fitted inside a scikit-learn pipeline using training data only, helping avoid data leakage. A Random Forest classifier with 250 trees was trained. A majority-class DummyClassifier provides a simple baseline.

## 5. Exploratory Data Analysis
The dashboard shows a preview of the dataset, counts of performance categories, and numerical-feature correlations. The saved figure `reports/performance_distribution.png` visualizes the target distribution. Since data is synthetic, patterns should be interpreted as properties of the generator rather than as findings about real students.

## 6. Model Development and Evaluation
The model was assessed on a held-out test set using accuracy, macro-F1, per-class precision/recall and a confusion matrix.

| Metric | Result |
|---|---:|
| Random Forest test accuracy | 0.613 |
| Random Forest macro-F1 | 0.507 |
| Majority-class baseline accuracy | 0.542 |

Detailed class-level metrics are in `reports/classification_report.csv`, and the confusion matrix is in `reports/confusion_matrix.png`. A model can perform well on synthetic data because its target is generated from a known formula; these metrics are not evidence of real-world validity.

## 7. Application
The Streamlit app provides three tabs: prediction and recommendations, dataset exploration, and model information. Users can change seven input features and request a prediction. The app displays class probabilities as model outputs, then separately applies explicit rules to suggest practical learning habits. The rules are intentionally transparent and do not claim to be model explanations.

## 8. Limitations and Ethics
- Synthetic data does not capture the complexity or diversity of real student experiences.
- The target labels are derived from a simulated formula and may encode its assumptions.
- Predictions can be wrong and must not be treated as definitive judgments.
- No identifiable student records should be uploaded.
- The prototype must not be used for grading, admissions, discipline, or other high-stakes decisions.
- Real deployment would require consent, privacy safeguards, representative data, domain-expert review, fairness evaluation, and ongoing monitoring.

## 9. Conclusion
The project demonstrates the main steps of a basic AI/ML application: data preparation, exploratory analysis, model training, held-out evaluation, and a Streamlit interface. The synthetic dataset makes the workflow reproducible and safe for a student demonstration. Future work could evaluate a legally shareable, anonymized public dataset and compare additional models under careful ethical review.

## 10. References and Tools
- Python
- Pandas and NumPy
- Scikit-learn
- Matplotlib
- Streamlit
- Joblib

Dataset: generated synthetically for this educational project; no external student dataset used.
