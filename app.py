from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI()

# Loading my trained Gradient Boosting model pipeline
model_pipeline = joblib.load("gradient_boosting_health_risk_model.joblib")

class PatientProfile(BaseModel):
    Age: float
    Weight: float
    Height: float
    Exercise: str
    Sleep: float
    Sugar_intake: str
    Smoking: str
    Alcohol: str
    Married: str
    Profession: str
    Bmi: float

@app.post("/predict")
def predict_risk(patient: PatientProfile):
    try:
        # 1. Building the dictionary using the available raw text values
        raw_data = {
            "Age": float(patient.Age),
            "Weight": float(patient.Weight),
            "Height": float(patient.Height),
            "Exercise": str(patient.Exercise),
            "Sleep": float(patient.Sleep),
            "Sugar_intake": str(patient.Sugar_intake),
            "Smoking": str(patient.Smoking),
            "Alcohol": str(patient.Alcohol),
            "Married": str(patient.Married),
            "Profession": str(patient.Profession),
            "Bmi": float(patient.Bmi)
        }
        
        # 2. Converting to DataFrame and matching column order accuately
        input_df = pd.DataFrame([raw_data])
        notebook_order = [
            'Age', 'Weight', 'Height', 'Exercise', 'Sleep', 
            'Sugar_intake', 'Smoking', 'Alcohol', 'Married', 
            'Profession', 'Bmi'
        ]
        input_df = input_df[notebook_order]
        
        # 3. Extracting raw probabilities
        probabilities = model_pipeline.predict_proba(input_df)[0]
        prob_class_0 = float(probabilities[0])  # Model's internal High Risk tracker
        prob_class_1 = float(probabilities[1])  # Model's internal Low Risk tracker
        
        # 4. MODEL LOGIC
        # High score in Class 1 (prob_class_1) = Truly Healthy (Low Risk)
        # High score in Class 0 (prob_class_0) = Truly Unhealthy (High Risk)
        
        if prob_class_0 > 0.50:
            risk_status = "High Risk"
            high_risk_prob = prob_class_0  # Shows the actual high risk severity
        else:
            risk_status = "Low Risk"
            high_risk_prob = prob_class_0  # To Keep it as the true risk score (almost 0%)

        return {
            "status": "success",
            "prediction": risk_status,
            "high_risk_probability": high_risk_prob
        }
        
    except Exception as e:
        return {"status": "error", "message": str(e)}