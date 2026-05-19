from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
import uvicorn

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

# Allow all origins for now
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define what a customer request looks like
class Customer(BaseModel):
    gender: int                  # 0 = Female, 1 = Male
    SeniorCitizen: int           # 0 = No, 1 = Yes
    Partner: int                 # 0 = No, 1 = Yes
    Dependents: int              # 0 = No, 1 = Yes
    tenure: int                  # months as customer
    PhoneService: int            # 0 = No, 1 = Yes
    MultipleLines: int           # 0 = No, 1 = Yes, 2 = No phone
    InternetService: int         # 0 = DSL, 1 = Fiber, 2 = No
    OnlineSecurity: int          # 0 = No, 1 = Yes, 2 = No internet
    OnlineBackup: int            # 0 = No, 1 = Yes, 2 = No internet
    DeviceProtection: int        # 0 = No, 1 = Yes, 2 = No internet
    TechSupport: int             # 0 = No, 1 = Yes, 2 = No internet
    StreamingTV: int             # 0 = No, 1 = Yes, 2 = No internet
    StreamingMovies: int         # 0 = No, 1 = Yes, 2 = No internet
    Contract: int                # 0 = Month-to-month, 1 = One year, 2 = Two year
    PaperlessBilling: int        # 0 = No, 1 = Yes
    PaymentMethod: int           # 0-3 = different payment methods
    MonthlyCharges: float        # monthly bill amount
    TotalCharges: float          # total amount paid

# Health check endpoint
@app.get("/")
def root():
    return {
        "status": "online",
        "model": "XGBoost Churn Predictor",
        "version": "1.0.0"
    }

# Main prediction endpoint
@app.post("/predict")
def predict(customer: Customer):
    # Convert input to array in correct feature order
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

    # Get churn probability
    churn_probability = model.predict_proba(input_data)[0][1]
    churn_predicted = int(churn_probability >= threshold)

    # Assign risk segment
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
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)