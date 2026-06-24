import streamlit as st
import requests
import multiprocessing
import uvicorn
from app import app  # Import your FastAPI instance safely

# To Automatically run FastAPI backend alongside Streamlit
def run_backend():
    # Running FastAPI on host 127.0.0.1 and port 8000
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="error")

# Starting the backend process once if it isn't already running
if "backend_started" not in st.session_state:
    backend_process = multiprocessing.Process(target=run_backend, daemon=True)
    backend_process.start()
    st.session_state["backend_started"] = True

# Setting page design configuration
st.set_page_config(page_title="Health Risk Analyzer", page_icon="🏥", layout="centered")

st.title("Patient Health Risk Diagnosis Tool")
st.write("Adjust the clinical parameters below to extract real-time predictive health analytics from the Gradient Boosting model.")
st.markdown("---")

# Splitting visual inputs into 2 neat columns
col1, col2 = st.columns(2)

with col1:
    st.subheader("Vital Stats")
    age = st.slider("Age", min_value=1, max_value=100, value=35)
    weight = st.number_input("Weight (kg)", min_value=30.0, max_value=200.0, value=70.0, step=0.1)
    height = st.number_input("Height (cm)", min_value=100.0, max_value=250.0, value=170.0, step=0.1)
    sleep = st.slider("Average Daily Sleep (Hours)", min_value=3.0, max_value=12.0, value=7.0, step=0.5)
    
    # Automatically Calculating BMI for the user
    height_m = height / 100.0
    bmi = round(weight / (height_m ** 2), 1)
    st.info(f"**Calculated BMI:** {bmi}")

with col2:
    st.subheader("Lifestyle Variables")
    smoking = st.selectbox("Do you smoke?", ["no", "yes"])
    alcohol = st.selectbox("Do you consume alcohol?", ["no", "yes"])
    married = st.selectbox("Are you married?", ["no", "yes"])
    profession = st.selectbox("Profession", ["engineer", "doctor", "teacher", "other"])
    exercise = st.selectbox("Exercise Frequency", ["none", "low", "medium", "high"])
    sugar_intake = st.selectbox("Sugar Intake Level", ["low", "medium", "high"])

st.markdown("---")

# Triggering prediction when clicked:
if st.button("Analyze Patient Metrics", use_container_width=True):
    # Enforcing the exact Jupyter Notebook layout sequence;
    patient_profile = {
        "Age": float(age),
        "Weight": float(weight),
        "Height": float(height),
        "Exercise": exercise,
        "Sleep": float(sleep),
        "Sugar_intake": sugar_intake,
        "Smoking": smoking,
        "Alcohol": alcohol,
        "Married": married,
        "Profession": profession,
        "Bmi": float(bmi)  
    }
    
    backend_url = "http://127.0.0.1:8000/predict"
    
    try:
        response = requests.post(backend_url, json=patient_profile)
        
        if response.status_code == 200:
            result = response.json()
            
            if result.get("status") == "error":
                st.error(f"Backend Processing Error: {result.get('message')}")
            else:
                prediction = result["prediction"]
                prob = result["high_risk_probability"] * 100
                
                st.subheader("Diagnostic Evaluation")
                if prediction == "High Risk":
                    st.error(f"**Assessment:** {prediction}")
                    st.warning(f"**High Risk Probability Score:** {prob:.2f}%")
                else:
                    st.success(f"**Assessment:** {prediction}")
                    st.info(f"**High Risk Probability Score:** {prob:.2f}%")
        else:
            st.error(f"Backend API returned an error code: {response.status_code}")
            
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to Backend API. Please ensure your FastAPI server is running on port 8000.")