# Customer Churn Prediction System

An end-to-end machine learning project for predicting telecom customer churn. The project trains an XGBoost churn classifier, serves predictions through a FastAPI application, provides a browser-based prediction form, and includes Power BI/dashboard artifacts for business analysis.

The model scores each customer profile, returns a churn probability, assigns a risk segment, and gives a simple retention-oriented message for business users.

## Highlights

- Predicts telecom customer churn from 19 customer, service, contract, and billing features.
- Uses an XGBoost classifier trained on the Telco Customer Churn dataset.
- Handles class imbalance with SMOTE during the modeling workflow.
- Stores trained model artifacts for direct API inference.
- Exposes a FastAPI REST API with automatic Swagger documentation.
- Serves a responsive frontend form from the same backend.
- Includes Power BI and CSV/JSON dashboard outputs for business reporting.
- Includes pytest coverage for health checks, prediction behavior, and validation cases.

## Model Results

- AUC-ROC: `0.84`
- Cross-validation standard deviation: `+/- 0.01`
- Training records: `7,043`
- Features used by the API model: `19`
- High-risk customers identified in the dashboard output: `1,605`
- Overall churn rate in the dataset summary: `28.8%`

## Business Problem

Customer churn is expensive for telecom companies because retaining an existing customer is usually cheaper than acquiring a new one. This project helps identify customers who are most likely to leave so retention teams can prioritize outreach, offers, and service improvements.

The prediction output is designed for practical action:

| Risk Segment | Probability Range | Recommended Action |
| --- | --- | --- |
| Low Risk | Below 30% | Continue standard engagement and monitoring |
| Medium Risk | 30% to below 60% | Schedule proactive outreach or loyalty follow-up |
| High Risk | 60% and above | Prioritize immediate retention action |

## Tech Stack

| Area | Tools |
| --- | --- |
| Language | Python |
| API | FastAPI, Uvicorn, Pydantic |
| Machine Learning | XGBoost, scikit-learn, imbalanced-learn |
| Data Analysis | pandas, NumPy |
| Explainability | SHAP, feature importance plots |
| Testing | pytest, FastAPI TestClient |
| Dashboard | Power BI |
| Frontend | HTML, CSS, JavaScript |

## Project Structure

```text
churn-predictor/
|-- main.py                         # FastAPI app, static frontend serving, prediction endpoint
|-- requirements.txt                # Python dependencies
|-- README.md                       # Project documentation
|-- Telco-Customer-Churn.csv        # Source dataset
|-- churn_predictor.ipynb           # Notebook for training, evaluation, and analysis
|-- Customer_Churn_Dashboard.pbix   # Power BI dashboard file
|-- api/                            # API-related project folder
|-- frontend/
|   `-- index.html                  # Browser UI for churn prediction
|-- model/
|   |-- churn_model.pkl             # Trained XGBoost model
|   |-- threshold.pkl               # Decision threshold for churn classification
|   |-- feature_names.pkl           # Feature order used at inference time
|   |-- feature_importance.png      # Feature importance visualization
|   `-- shap_impact.png             # SHAP impact visualization
|-- dashboard/
|   |-- churn_predictions.csv       # Customer-level prediction output
|   `-- summary_stats.json          # Dashboard summary metrics
|-- tests/
|   |-- test_api.py                 # API, prediction, and validation tests
|   |-- Procfile                    # Deployment helper file
|   `-- runtime.txt                 # Runtime helper file
`-- venv/                           # Local virtual environment, not required in source control
```

## How It Works

1. The notebook loads and prepares the Telco Customer Churn dataset.
2. Categorical customer fields are encoded into numeric values.
3. Class imbalance is handled during training with SMOTE.
4. An XGBoost model is trained and evaluated.
5. Model artifacts are saved in the `model/` directory.
6. `main.py` loads the saved model, threshold, and feature names at startup.
7. The `/predict` endpoint accepts a customer profile and returns churn risk.
8. The frontend posts form values to `/predict` and displays the result.

## Setup

Clone the repository and install dependencies:

```bash
git clone https://github.com/Jewelsony2023/customer-churn-predictor
cd customer-churn-predictor
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

On macOS/Linux, activate the virtual environment with:

```bash
source venv/bin/activate
```

## Run Locally

Start the FastAPI server:

```bash
uvicorn main:app --reload
```

Then open:

```text
http://127.0.0.1:8000
```

API documentation is available at:

```text
http://127.0.0.1:8000/docs
```

## API Endpoints

| Method | Endpoint | Description |
| --- | --- | --- |
| `GET` | `/` | Serves the frontend prediction UI |
| `GET` | `/health` | Returns API/model health information |
| `POST` | `/predict` | Predicts churn risk for one customer |
| `GET` | `/docs` | Interactive Swagger API documentation |

## Example Health Check

Request:

```bash
curl http://127.0.0.1:8000/health
```

Response:

```json
{
  "status": "online",
  "model": "XGBoost Churn Predictor",
  "version": "1.0.0"
}
```

## Prediction Request

The `/predict` endpoint expects numeric encoded feature values in the exact schema used by the API.

Command Prompt example:

```cmd
curl -X POST "http://127.0.0.1:8000/predict" ^
  -H "Content-Type: application/json" ^
  -d "{\"gender\":1,\"SeniorCitizen\":0,\"Partner\":0,\"Dependents\":0,\"tenure\":2,\"PhoneService\":1,\"MultipleLines\":0,\"InternetService\":1,\"OnlineSecurity\":0,\"OnlineBackup\":0,\"DeviceProtection\":0,\"TechSupport\":0,\"StreamingTV\":0,\"StreamingMovies\":0,\"Contract\":0,\"PaperlessBilling\":1,\"PaymentMethod\":2,\"MonthlyCharges\":85.50,\"TotalCharges\":171.00}"
```

PowerShell example:

```powershell
$body = @{
  gender = 1
  SeniorCitizen = 0
  Partner = 0
  Dependents = 0
  tenure = 2
  PhoneService = 1
  MultipleLines = 0
  InternetService = 1
  OnlineSecurity = 0
  OnlineBackup = 0
  DeviceProtection = 0
  TechSupport = 0
  StreamingTV = 0
  StreamingMovies = 0
  Contract = 0
  PaperlessBilling = 1
  PaymentMethod = 2
  MonthlyCharges = 85.50
  TotalCharges = 171.00
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://127.0.0.1:8000/predict" -Method Post -ContentType "application/json" -Body $body
```

Example response:

```json
{
  "churn_probability": 84.75,
  "churn_predicted": 1,
  "risk_segment": "High Risk",
  "message": "This customer is likely to churn"
}
```

## Input Features

| Field | Type | Notes |
| --- | --- | --- |
| `gender` | integer | Encoded customer gender |
| `SeniorCitizen` | integer | `0` = no, `1` = yes |
| `Partner` | integer | `0` = no, `1` = yes |
| `Dependents` | integer | `0` = no, `1` = yes |
| `tenure` | integer | Number of months as a customer |
| `PhoneService` | integer | Encoded phone service status |
| `MultipleLines` | integer | Encoded multiple-lines status |
| `InternetService` | integer | Encoded internet service type |
| `OnlineSecurity` | integer | Encoded online security status |
| `OnlineBackup` | integer | Encoded online backup status |
| `DeviceProtection` | integer | Encoded device protection status |
| `TechSupport` | integer | Encoded tech support status |
| `StreamingTV` | integer | Encoded streaming TV status |
| `StreamingMovies` | integer | Encoded streaming movies status |
| `Contract` | integer | Encoded contract type |
| `PaperlessBilling` | integer | `0` = no, `1` = yes |
| `PaymentMethod` | integer | Encoded payment method |
| `MonthlyCharges` | float | Monthly bill amount |
| `TotalCharges` | float | Total amount charged |

The frontend provides dropdown labels for the encoded categorical values, so users do not need to manually remember the numeric codes.

## Frontend

The frontend is a single-page HTML/CSS/JavaScript interface located at `frontend/index.html`. It is served automatically by FastAPI at the root URL.

Main features:

- Customer profile form grouped by personal, phone, internet, and billing details.
- Client-side validation for tenure and charge fields.
- Prediction result card with churn probability, risk segment, and retention recommendation.
- Model summary stats and top churn driver insights.
- Responsive layout for desktop and smaller screens.

## Dashboard Artifacts

The repository includes business reporting outputs:

- `Customer_Churn_Dashboard.pbix`: Power BI dashboard file.
- `dashboard/churn_predictions.csv`: customer-level churn prediction export.
- `dashboard/summary_stats.json`: summary metrics used for reporting.

Current summary metrics include:

| Metric | Value |
| --- | ---: |
| Total Customers | 7,043 |
| Predicted Churners | 2,029 |
| Churn Rate | 28.8% |
| High Risk Customers | 1,605 |
| Average Monthly Charge, Churners | 73.76 |
| Average Monthly Charge, Retained | 61.12 |
| Average Tenure, Churners | 12.9 months |
| Average Tenure, Retained | 40.2 months |

## Testing

Run the test suite with:

```bash
pytest
```

The tests cover:

- Root frontend route.
- Health check response.
- Prediction endpoint status and response fields.
- Probability range and binary prediction output.
- Risk segment consistency.
- High-risk and low-risk customer examples.
- Missing fields, wrong types, empty request bodies, and extra fields.

## Deployment Notes

The app can run on platforms that support Python web services. `main.py` reads the `PORT` environment variable when executed directly:

```bash
python main.py
```

For ASGI deployment, use:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

Make sure the deployment includes:

- `main.py`
- `requirements.txt`
- `frontend/index.html`
- the full `model/` directory

## Important Notes

- The API expects already-encoded numeric inputs, matching the training pipeline and frontend mappings.
- The saved model artifacts must remain in the `model/` directory unless `main.py` is updated.
- This project is intended as a machine learning portfolio and decision-support system. Real production use should add monitoring, authentication, stricter input validation, retraining workflows, and bias/performance checks across customer segments.

## Author

Developed by Jewel Sony as a customer churn prediction and analytics project.
