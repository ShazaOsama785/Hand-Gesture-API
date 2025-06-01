# tests/test_main.py

import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

# Sample valid "like" gesture (should match a known class)
valid_like_landmarks = {
    "landmarks": [
        0.12, 0.45, -0.02, 0.14, 0.48, -0.01, 0.16, 0.52, 0.00,
        0.18, 0.55, 0.02, 0.20, 0.57, 0.01, 0.22, 0.53, -0.01,
        0.25, 0.50, -0.03, 0.27, 0.47, -0.02, 0.30, 0.44, -0.01,
        0.32, 0.41, 0.00, 0.34, 0.39, 0.01, 0.36, 0.42, 0.01,
        0.38, 0.45, 0.00, 0.40, 0.48, -0.01, 0.43, 0.50, -0.01,
        0.45, 0.52, 0.00, 0.47, 0.54, 0.01, 0.49, 0.56, 0.01,
        0.51, 0.58, 0.00, 0.53, 0.60, -0.01, 0.55, 0.62, -0.01
    ]
}

def test_home():
    response = client.get("/")
    assert response.status_code == 200
    assert "Hand Gesture Recognition" in response.json()["message"]

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_predict_valid_input():
    response = client.post("/predict", json=valid_like_landmarks)
    assert response.status_code == 200
    assert "gesture" in response.json()

def test_predict_invalid_input():
    # Sending fewer than 63 values
    invalid_landmarks = {"landmarks": [0.1, 0.2, 0.3]}
    response = client.post("/predict", json=invalid_landmarks)
    assert response.status_code == 400
    assert "Expected 63 values" in response.json()["detail"]
