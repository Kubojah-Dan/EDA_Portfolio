import os
import sys
import json
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Use TestClient without starting a real server
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    from app.backend.main import app
    return TestClient(app)


def test_health_endpoint(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "version" in data
    assert "models_loaded" in data
    assert isinstance(data["models_loaded"], list)


def test_docs_available(client):
    resp = client.get("/docs")
    assert resp.status_code == 200


def test_predict_endpoint_structure(client):
    payload = {
        "hour": 8, "day_of_week": 1, "month": 3, "year": 2022,
        "is_weekend": 0, "is_rush_hour": 1, "is_night": 0, "season": 1,
        "duration_min": 30.0, "weather_severity_score": 1.0,
        "road_feature_count": 2,
        "Distance(mi)": 0.5, "Temperature(F)": 55.0,
        "Humidity(%)": 70.0, "Pressure(in)": 29.9,
        "Visibility(mi)": 10.0, "Wind_Speed(mph)": 10.0,
        "Precipitation(in)": 0.0,
    }
    resp = client.post("/predict", json=payload)
    # Either 200 (model loaded) or 503 (model not yet trained)
    assert resp.status_code in (200, 503)
    if resp.status_code == 200:
        data = resp.json()
        assert "prediction" in data
        assert "probability" in data
        assert "severity_label" in data
        assert data["prediction"] in (0, 1)
        assert 0.0 <= data["probability"] <= 1.0


def test_predict_invalid_input(client):
    payload = {"hour": 99}  # invalid hour
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 422  # Unprocessable Entity


def test_stats_endpoint(client):
    resp = client.get("/stats")
    # 200 if data exists, 503 if pipeline not run yet
    assert resp.status_code in (200, 503)
    if resp.status_code == 200:
        data = resp.json()
        assert "total_accidents" in data
        assert "by_severity" in data


def test_metrics_endpoint(client):
    resp = client.get("/metrics")
    assert resp.status_code in (200, 404)
    if resp.status_code == 200:
        data = resp.json()
        assert isinstance(data, list)
        if data:
            assert "model_name" in data[0]
            assert "test_auc" in data[0]


def test_eda_severity_by_hour(client):
    resp = client.get("/eda/severity-by-hour")
    assert resp.status_code in (200, 404, 503)


def test_eda_top_cities_param(client):
    resp = client.get("/eda/top-cities?top_n=5")
    assert resp.status_code in (200, 404, 503)
    if resp.status_code == 200:
        data = resp.json()
        assert len(data) <= 5


def test_eda_top_cities_invalid_param(client):
    resp = client.get("/eda/top-cities?top_n=200")
    assert resp.status_code == 422

