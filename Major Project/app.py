import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "models" / "performance_pipeline.joblib"
DATA_PATH = ROOT / "data" / "student_performance_synthetic.csv"

st.set_page_config(page_title="Student Performance AI", page_icon="🎓", layout="wide")
st.markdown("""
<style>
.main {background-color:#f7f9fc;}
.hero {padding:1.2rem 1.5rem;border-radius:18px;background:linear-gradient(120deg,#102a43,#2463a6);color:white;}
.hero h1 {color:white;margin-bottom:.3rem;}
.small-note {color:#52606d;font-size:.92rem;}
div[data-testid="stMetric"] {background:white;padding:14px;border-radius:12px;border:1px solid #e6edf5;}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

st.markdown('<div class="hero"><h1>🎓 Student Performance AI</h1><p>Performance prediction + practical learning recommendations</p></div>', unsafe_allow_html=True)
st.write("")
st.info("Educational prototype only. Predictions are estimates based on a synthetic demonstration dataset—not a judgment of a real student.")
model = load_model()
data = load_data()

tab1, tab2, tab3 = st.tabs(["🔮 Predict performance", "📊 Explore dataset", "🧠 About the model"])
with tab1:
    left, right = st.columns([1,1.1], gap="large")
    with left:
        st.subheader("Student learning profile")
        study = st.slider("Study hours per week", 0.0, 35.0, 12.0, 0.5)
        attendance = st.slider("Attendance (%)", 35.0, 100.0, 80.0, 1.0)
        previous = st.slider("Previous exam marks (%)", 10.0, 100.0, 65.0, 1.0)
        assignments = st.slider("Assignments completed (%)", 0.0, 100.0, 75.0, 1.0)
        sleep = st.slider("Average sleep (hours/day)", 3.0, 10.0, 7.0, 0.5)
        internet = st.selectbox("Reliable internet access", ["Yes", "No"])
        participation = st.selectbox("Class participation", ["Low", "Medium", "High"], index=1)
        run = st.button("Predict & get study plan", type="primary", use_container_width=True)
    with right:
        st.subheader("Your result")
        if run:
            row = pd.DataFrame([{
                "study_hours_per_week":study,
                "attendance_percentage":attendance,
                "previous_exam_marks":previous,
                "assignment_completion_percentage":assignments,
                "sleep_hours_per_day":sleep,
                "internet_access":internet,
                "class_participation":participation
            }])
            outcome = model.predict(row)[0]
            probs = model.predict_proba(row)[0]
            classes = model.named_steps["classifier"].classes_
            confidence = float(probs.max())*100
            color = {"Needs Support":"🔴","Developing":"🟠","On Track":"🟢"}.get(outcome,"🔵")
            st.metric("Estimated performance category", f"{color} {outcome}")
            st.caption(f"Highest model class probability: {confidence:.1f}% (not a guarantee of correctness)")
            st.write("**Model class probabilities**")
            st.bar_chart(pd.DataFrame({"Probability":probs}, index=classes))
            recs=[]
            if study < 8: recs.append("Try a realistic routine of 30–45 minutes of focused study on most days, then gradually build up.")
            if attendance < 75: recs.append("Aim to attend more classes and ask for notes or support for missed topics.")
            if previous < 60: recs.append("Review previous mistakes and practise a small set of questions each day.")
            if assignments < 80: recs.append("Break assignments into smaller tasks and set an earlier personal deadline.")
            if sleep < 6.5: recs.append("Protect a consistent sleep schedule; rest supports learning and concentration.")
            if participation == "Low": recs.append("Ask one question or contribute one idea in class when comfortable.")
            if not recs: recs.append("Keep your routine steady. Use weekly practice tests and review topics you find difficult.")
            st.markdown("### Suggested learning plan")
            for r in recs: st.write("• " + r)
            st.caption("Suggestions are simple rules, separate from the machine-learning prediction.")
        else:
            st.write("Adjust the inputs and press **Predict & get study plan** to see an estimated category and tailored suggestions.")
            image_path = ROOT / "reports" / "performance_distribution.png"

image_path = ROOT / "reports" / "performance_distribution.png"

if image_path.exists():
    st.image(str(image_path), width="stretch")
else:
    st.info("Performance distribution chart is not available yet.")
with tab2:
    st.subheader("Synthetic demonstration dataset")
    c1,c2,c3=st.columns(3)
    c1.metric("Records",f"{len(data):,}")
    c2.metric("Input features", "7")
    c3.metric("Missing input cells", f"{int(data.drop(columns=['performance_category']).isna().sum().sum())}")
    st.write("The data is synthetically generated for learning and testing. It is not real student data.")
    st.dataframe(data.head(20), use_container_width=True)
    st.markdown("#### Performance category distribution")
    counts=data["performance_category"].value_counts()
    st.bar_chart(counts)
    st.markdown("#### Feature relationships")
    numeric=data.select_dtypes(include=np.number)
    st.dataframe(numeric.corr().round(2), use_container_width=True)
with tab3:
    st.subheader("How it works")
    st.markdown("""
    1. **Preprocessing:** median imputation and scaling for numerical columns; most-frequent imputation and one-hot encoding for categories.
    2. **Split:** 80% training and 20% held-out testing with stratification.
    3. **Model:** Random Forest classifier compared with a majority-class baseline.
    4. **Evaluation:** accuracy, macro-F1, per-class precision/recall and confusion matrix are saved in `reports/`.
    5. **Recommendations:** transparent rules based on input values; these are not generated by the classifier.

    **Important limitation:** the dataset is synthetic and its labels are generated from a simple formula. Strong test results on this data do not prove that the model will perform well on real students. Do not use it for grading, admissions, discipline, or other high-stakes decisions.
    """)
    image_path = ROOT / "reports" / "confusion_matrix.png"

if image_path.exists():
    st.image(
        str(image_path),
        caption="Held-out test confusion matrix",
        width="stretch"
    )
else:
    st.info("Confusion matrix chart is not available yet.")
    st.write("Model file: `models/performance_pipeline.joblib`")
st.divider()
st.caption("Built with Python • Pandas • Scikit-learn • Streamlit | AI/ML major project prototype")