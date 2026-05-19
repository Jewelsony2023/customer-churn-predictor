import pytest
from fastapi.testclient import TestClient
import sys
import os

# Make sure Python can find our main.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import app

client = TestClient(app)

# ─── VALID CUSTOMER DATA ──────────────────────────────────────────────────────
# This is a realistic high-risk customer we'll reuse across tests
VALID_CUSTOMER = {
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
}

# A clearly low-risk customer — long tenure, annual contract, low charges
LOW_RISK_CUSTOMER = {
    "gender": 0,
    "SeniorCitizen": 0,
    "Partner": 1,
    "Dependents": 1,
    "tenure": 60,
    "PhoneService": 1,
    "MultipleLines": 1,
    "InternetService": 0,
    "OnlineSecurity": 1,
    "OnlineBackup": 1,
    "DeviceProtection": 1,
    "TechSupport": 1,
    "StreamingTV": 1,
    "StreamingMovies": 1,
    "Contract": 2,
    "PaperlessBilling": 0,
    "PaymentMethod": 0,
    "MonthlyCharges": 45.00,
    "TotalCharges": 2700.00
}


# ─── 1. HEALTH CHECK TESTS ────────────────────────────────────────────────────
class TestHealthCheck:

    def test_root_returns_200(self):
        """API should be online and return 200"""
        response = client.get("/")
        assert response.status_code == 200

    def test_root_returns_correct_fields(self):
        """Health check should return status, model name and version"""
        response = client.get("/")
        data = response.json()
        assert "status" in data
        assert "model" in data
        assert "version" in data

    def test_root_status_is_online(self):
        """Status field should say online"""
        response = client.get("/")
        assert response.json()["status"] == "online"


# ─── 2. PREDICTION ENDPOINT TESTS ─────────────────────────────────────────────
class TestPredictionEndpoint:

    def test_predict_returns_200(self):
        """Valid customer data should return 200"""
        response = client.post("/predict", json=VALID_CUSTOMER)
        assert response.status_code == 200

    def test_predict_returns_required_fields(self):
        """Response must contain all 4 required fields"""
        response = client.post("/predict", json=VALID_CUSTOMER)
        data = response.json()
        assert "churn_probability" in data
        assert "churn_predicted" in data
        assert "risk_segment" in data
        assert "message" in data

    def test_churn_probability_is_percentage(self):
        """Churn probability must be between 0 and 100"""
        response = client.post("/predict", json=VALID_CUSTOMER)
        prob = response.json()["churn_probability"]
        assert 0.0 <= prob <= 100.0

    def test_churn_predicted_is_binary(self):
        """Churn predicted must be 0 or 1"""
        response = client.post("/predict", json=VALID_CUSTOMER)
        pred = response.json()["churn_predicted"]
        assert pred in [0, 1]

    def test_risk_segment_is_valid(self):
        """Risk segment must be one of three valid values"""
        response = client.post("/predict", json=VALID_CUSTOMER)
        segment = response.json()["risk_segment"]
        assert segment in ["Low Risk", "Medium Risk", "High Risk"]

    def test_message_matches_prediction(self):
        """Message should match the churn prediction"""
        response = client.post("/predict", json=VALID_CUSTOMER)
        data = response.json()
        if data["churn_predicted"] == 1:
            assert "likely to churn" in data["message"]
        else:
            assert "likely to stay" in data["message"]


# ─── 3. BUSINESS LOGIC TESTS ──────────────────────────────────────────────────
class TestBusinessLogic:

    def test_high_risk_customer_flagged_correctly(self):
        """Month-to-month, short tenure, high charges = should be high risk"""
        response = client.post("/predict", json=VALID_CUSTOMER)
        data = response.json()
        assert data["churn_probability"] > 50.0
        assert data["risk_segment"] == "High Risk"

    def test_low_risk_customer_flagged_correctly(self):
        """Long tenure, two-year contract = lower churn probability than high risk"""
        high_risk_response = client.post("/predict", json=VALID_CUSTOMER)
        low_risk_response = client.post("/predict", json=LOW_RISK_CUSTOMER)
        
        high_prob = high_risk_response.json()["churn_probability"]
        low_prob = low_risk_response.json()["churn_probability"]
        
        # Low risk customer should always score lower than high risk customer
        assert low_prob < high_risk_response.json()["churn_probability"], \
            f"Expected low risk ({low_prob}%) < high risk ({high_prob}%)"

    def test_risk_segment_matches_probability(self):
        """Risk segment must be consistent with probability value"""
        response = client.post("/predict", json=VALID_CUSTOMER)
        data = response.json()
        prob = data["churn_probability"]
        segment = data["risk_segment"]

        if prob < 30.0:
            assert segment == "Low Risk"
        elif prob < 60.0:
            assert segment == "Medium Risk"
        else:
            assert segment == "High Risk"

    def test_senior_citizen_high_risk(self):
        """Senior citizen on month-to-month with high charges should be high risk"""
        customer = VALID_CUSTOMER.copy()
        customer["SeniorCitizen"] = 1
        customer["MonthlyCharges"] = 95.0
        response = client.post("/predict", json=customer)
        assert response.status_code == 200
        assert response.json()["churn_probability"] > 30.0


# ─── 4. INPUT VALIDATION TESTS ────────────────────────────────────────────────
class TestInputValidation:

    def test_missing_field_returns_422(self):
        """Missing required field should return 422 validation error"""
        incomplete = VALID_CUSTOMER.copy()
        del incomplete["MonthlyCharges"]
        response = client.post("/predict", json=incomplete)
        assert response.status_code == 422

    def test_wrong_type_returns_422(self):
        """Sending string where number expected should return 422"""
        bad_data = VALID_CUSTOMER.copy()
        bad_data["MonthlyCharges"] = "not-a-number"
        response = client.post("/predict", json=bad_data)
        assert response.status_code == 422

    def test_empty_body_returns_422(self):
        """Empty request body should return 422"""
        response = client.post("/predict", json={})
        assert response.status_code == 422

    def test_extra_fields_are_ignored(self):
        """Extra unexpected fields should not crash the API"""
        customer_with_extras = VALID_CUSTOMER.copy()
        customer_with_extras["unknown_field"] = "some_value"
        customer_with_extras["another_field"] = 999
        response = client.post("/predict", json=customer_with_extras)
        assert response.status_code == 200

    def test_zero_tenure_is_valid(self):
        """Brand new customer with 0 tenure should not crash"""
        new_customer = VALID_CUSTOMER.copy()
        new_customer["tenure"] = 0
        new_customer["TotalCharges"] = 0.0
        response = client.post("/predict", json=new_customer)
        assert response.status_code == 200

    def test_very_high_charges_is_valid(self):
        """Unusually high monthly charges should not crash the API"""
        customer = VALID_CUSTOMER.copy()
        customer["MonthlyCharges"] = 500.0
        customer["TotalCharges"] = 30000.0
        response = client.post("/predict", json=customer)
        assert response.status_code == 200