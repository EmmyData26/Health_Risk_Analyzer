import streamlit as st
import pandas as pd
import joblib

# Load model once when app starts
model_pipeline = joblib.load("gradient_boosting_health_risk_model.joblib")

# Page configuration
st.set_page_config(
    page_title="Health Risk Analyzer",
    page_icon="🏥",
    layout="centered"
)

st.title("Patient Health Risk Diagnosis Tool")
st.write(
    "Adjust the clinical parameters below to extract real-time predictive "
    "health analytics from the Gradient Boosting model."
)

st.markdown("---")

# Layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("Vital Stats")

    age = st.slider(
        "Age",
        min_value=1,
        max_value=100,
        value=35
    )

    weight = st.number_input(
        "Weight (kg)",
        min_value=30.0,
        max_value=200.0,
        value=70.0,
        step=0.1
    )

    height = st.number_input(
        "Height (cm)",
        min_value=100.0,
        max_value=250.0,
        value=170.0,
        step=0.1
    )

    sleep = st.slider(
        "Average Daily Sleep (Hours)",
        min_value=3.0,
        max_value=12.0,
        value=7.0,
        step=0.5
    )

    # BMI calculation
    height_m = height / 100
    bmi = round(weight / (height_m ** 2), 1)

    st.info(f"**Calculated BMI:** {bmi}")

with col2:
    st.subheader("Lifestyle Variables")

    smoking = st.selectbox(
        "Do you smoke?",
        ["no", "yes"]
    )

    alcohol = st.selectbox(
        "Do you consume alcohol?",
        ["no", "yes"]
    )

    married = st.selectbox(
        "Are you married?",
        ["no", "yes"]
    )

    profession = st.selectbox(
        "Profession",
        ["engineer", "doctor", "teacher", "other"]
    )

    exercise = st.selectbox(
        "Exercise Frequency",
        ["none", "low", "medium", "high"]
    )

    sugar_intake = st.selectbox(
        "Sugar Intake Level",
        ["low", "medium", "high"]
    )

st.markdown("---")

if st.button("Analyze Patient Metrics", use_container_width=True):

    try:
        # Build input exactly like your FastAPI backend
        raw_data = {
            "Age": float(age),
            "Weight": float(weight),
            "Height": float(height),
            "Exercise": str(exercise),
            "Sleep": float(sleep),
            "Sugar_intake": str(sugar_intake),
            "Smoking": str(smoking),
            "Alcohol": str(alcohol),
            "Married": str(married),
            "Profession": str(profession),
            "Bmi": float(bmi)
        }

        input_df = pd.DataFrame([raw_data])

        notebook_order = [
            "Age",
            "Weight",
            "Height",
            "Exercise",
            "Sleep",
            "Sugar_intake",
            "Smoking",
            "Alcohol",
            "Married",
            "Profession",
            "Bmi"
        ]

        input_df = input_df[notebook_order]

        # Predict
        probabilities = model_pipeline.predict_proba(input_df)[0]

        prob_class_0 = float(probabilities[0])
        prob_class_1 = float(probabilities[1])

        # Same logic from FastAPI
        if prob_class_0 > 0.50:
            prediction = "High Risk"
            high_risk_prob = prob_class_0
        else:
            prediction = "Low Risk"
            high_risk_prob = prob_class_0

        prob = high_risk_prob * 100

        st.subheader("Diagnostic Evaluation")

        if prediction == "High Risk":
            st.error(f"**Assessment:** {prediction}")
            st.warning(
                f"**High Risk Probability Score:** {prob:.2f}%"
            )
        else:
            st.success(f"**Assessment:** {prediction}")
            st.info(
                f"**High Risk Probability Score:** {prob:.2f}%"
            )

    except Exception as e:
        st.error(f"Prediction Error: {str(e)}")
