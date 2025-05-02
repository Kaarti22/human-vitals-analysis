from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(title="Vitals Health Prediction API")

model = joblib.load("vitals_rf_model.pkl")

class VitalSigns(BaseModel):
    age: int
    bloodGroup: str
    hasBpHigh: int
    hasBpLow: int
    gender: str
    height: float
    hasDiabetes: int
    weight: float
    heartRate: int
    SpO2: int
    temperature: float

label_mapping = {0: "normal", 1: "abnormal"}

@app.post("/predict")
async def predict_vitals(vitals: VitalSigns):
    input_data = pd.DataFrame([vitals.dict()])

    prediction = model.predict(input_data)[0]

    probabilities = model.predict_proba(input_data)[0]

    confidence_score = probabilities[prediction]

    result_label = label_mapping[prediction]

    return {
        "prediction": result_label,
        "confidence": round(confidence_score * 100, 2)   
    }