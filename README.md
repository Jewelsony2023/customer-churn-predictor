# Customer Churn Prediction System

An end-to-end machine learning system that predicts telecom customer churn with **84% AUC-ROC** using XGBoost, served via a REST API and visualized through a Power BI dashboard.

## Project Overview

Customer churn is one of the most costly problems in the telecom industry. This system identifies at-risk customers before they leave, enabling proactive retention strategies.

## Key Results

- **AUC-ROC: 0.84** across 5-fold cross validation (std: 0.01)
- **1,605 high-risk customers** identified out of 7,043
- **Top churn drivers:** Monthly charges, contract type, tenure, internet service, online security

## Tech Stack

- **Model:** XGBoost with SMOTE for class imbalance
- **Explainability:** SHAP values for feature importance
- **API:** FastAPI + Uvicorn
- **Dashboard:** Power BI
- **Language:** Python 3.11

## Project Structure
customer-churn-predictor/
├── main.py                  # FastAPI application
├── churn_predictor.ipynb    # Full ML pipeline notebook
├── model/
│   ├── churn_model.pkl      # Trained XGBoost model
│   ├── threshold.pkl        # Optimal decision threshold
│   └── feature_names.pkl    # Feature column names
├── dashboard/
│   └── churn_predictions.csv
└── requirements.txt
## API Usage

**Live API:** https://customer-churn-predictor-v7r3.onrender.com/docs

Start the server locally:
```bash
uvicorn main:app --reload
```

Make a prediction:
```bash
curl -X POST "https://customer-churn-predictor-v7r3.onrender.com/predict" \
-H "Content-Type: application/json" \
-d '{
  "gender": 1,
  "SeniorCitizen": 0,
  "Partner": 0,
  "Dependents": 0,
  "tenure": 2,
  "PhoneService": 1,
  "MultipleLines": 0,
  "InternetService": 1,
  "OnlineSecurity": 0,
  "OnlineBackup": 0,
  "DeviceProtection": 0,
  "TechSupport": 0,
  "StreamingTV": 0,
  "StreamingMovies": 0,
  "Contract": 0,
  "PaperlessBilling": 1,
  "PaymentMethod": 2,
  "MonthlyCharges": 85.50,
  "TotalCharges": 171.00
}'
```

Response:
```json
{
  "churn_probability": 84.75,
  "churn_predicted": 1,
  "risk_segment": "High Risk",
  "message": "This customer is likely to churn"
}
```

## Business Insights

| Segment | Customers | Action |
|---------|-----------|--------|
| High Risk | 1,605 | Immediate retention call |
| Medium Risk | 1,432 | Targeted discount offer |
| Low Risk | 4,006 | Standard engagement |

## Setup

```bash
git clone https://github.com/Jewelsony2023/customer-churn-predictor
cd customer-churn-predictor
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload
```