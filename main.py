from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import joblib
import numpy as np
import uvicorn
import os

# Load model and artifacts
model = joblib.load("model/churn_model.pkl")
threshold = joblib.load("model/threshold.pkl")
feature_names = joblib.load("model/feature_names.pkl")

# Initialize API
app = FastAPI(
    title="Customer Churn Prediction API",
    description="Predicts churn probability for telecom customers using XGBoost",
    version="1.0.0"
)

# Allow all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend static files
app.mount("/static", StaticFiles(directory="frontend"), name="static")

# Define customer input schema
class Customer(BaseModel):
    gender: int
    SeniorCitizen: int
    Partner: int
    Dependents: int
    tenure: int
    PhoneService: int
    MultipleLines: int
    InternetService: int
    OnlineSecurity: int
    OnlineBackup: int
    DeviceProtection: int
    TechSupport: int
    StreamingTV: int
    StreamingMovies: int
    Contract: int
    PaperlessBilling: int
    PaymentMethod: int
    MonthlyCharges: float
    TotalCharges: float

# Serve frontend at root
@app.get("/", response_class=FileResponse)
def serve_frontend():
    return FileResponse("frontend/index.html")

# Health check
@app.get("/health")
def health():
    return {
        "status": "online",
        "model": "XGBoost Churn Predictor",
        "version": "1.0.0"
    }

# Prediction endpoint
@app.post("/predict")
def predict(customer: Customer):
    input_data = np.array([[
        customer.gender,
        customer.SeniorCitizen,
        customer.Partner,
        customer.Dependents,
        customer.tenure,
        customer.PhoneService,
        customer.MultipleLines,
        customer.InternetService,
        customer.OnlineSecurity,
        customer.OnlineBackup,
        customer.DeviceProtection,
        customer.TechSupport,
        customer.StreamingTV,
        customer.StreamingMovies,
        customer.Contract,
        customer.PaperlessBilling,
        customer.PaymentMethod,
        customer.MonthlyCharges,
        customer.TotalCharges
    ]])

    churn_probability = model.predict_proba(input_data)[0][1]
    churn_predicted = int(churn_probability >= threshold)

    if churn_probability < 0.3:
        risk_segment = "Low Risk"
    elif churn_probability < 0.6:
        risk_segment = "Medium Risk"
    else:
        risk_segment = "High Risk"

    return {
        "churn_probability": round(float(churn_probability) * 100, 2),
        "churn_predicted": churn_predicted,
        "risk_segment": risk_segment,
        "message": "This customer is likely to churn" if churn_predicted == 1 else "This customer is likely to stay"
    }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)