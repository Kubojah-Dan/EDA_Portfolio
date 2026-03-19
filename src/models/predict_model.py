"""
predict_model.py
Inference pipeline: loads best model and returns predictions + probabilities.
Used by the FastAPI backend.
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import json
from typing import Optional
import numpy as np
import pandas as pd
import joblib

from src.utils.helpers import load_config
from src.utils.logger import get_logger

log = get_logger("predict_model")

_model_cache: dict = {}


def _load_best_model(cfg: dict, model_name: Optional[str] = None):
    models_dir = cfg["paths"]["models_dir"]

    if model_name is None:
        # Pick best by test AUC from metrics file
        metrics_path = os.path.join(models_dir, "test_metrics.json")
        if os.path.exists(metrics_path):
            with open(metrics_path) as f:
                data = json.load(f)
            best = max(data["test_metrics"], key=lambda x: x["test_auc"])
            model_name = best["model"]
        else:
            model_name = "xgboost"  # fallback

    cache_key = model_name
    if cache_key not in _model_cache:
        model_path = os.path.join(models_dir, f"model_{model_name}.pkl")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        _model_cache[cache_key] = joblib.load(model_path)
        log.info(f"Loaded model: {model_name}")

    return _model_cache[cache_key], model_name


def predict(features: dict, model_name: Optional[str] = None) -> dict:
    """
    Predict severity class for a single accident record.
    features: dict of feature_name -> value
    Returns: {"prediction": int, "probability": float, "model": str}
    """
    cfg = load_config()
    model, used_name = _load_best_model(cfg, model_name)

    # Load expected feature list
    meta_path = os.path.join(cfg["paths"]["models_dir"], "feature_metadata.json")
    with open(meta_path) as f:
        meta = json.load(f)
    expected_features = meta["features"]

    df = pd.DataFrame([features])
    # Fill missing features with 0
    for feat in expected_features:
        if feat not in df.columns:
            df[feat] = 0
    df = df[expected_features]

    prob = model.predict_proba(df)[0, 1]
    pred = int(prob >= 0.5)

    return {
        "prediction": pred,
        "severity_label": "High" if pred == 1 else "Low",
        "probability": round(float(prob), 4),
        "model": used_name,
    }


def batch_predict(df: pd.DataFrame, model_name: Optional[str] = None) -> pd.DataFrame:
    """Batch prediction for a DataFrame."""
    cfg = load_config()
    model, used_name = _load_best_model(cfg, model_name)

    meta_path = os.path.join(cfg["paths"]["models_dir"], "feature_metadata.json")
    with open(meta_path) as f:
        meta = json.load(f)
    expected_features = meta["features"]

    for feat in expected_features:
        if feat not in df.columns:
            df[feat] = 0
    X = df[expected_features]

    probs = model.predict_proba(X)[:, 1]
    preds = (probs >= 0.5).astype(int)

    result = df.copy()
    result["prediction"] = preds
    result["probability"] = probs
    result["model"] = used_name
    return result


if __name__ == "__main__":
    # Quick smoke test
    sample = {
        "hour": 8, "day_of_week": 1, "month": 3, "year": 2022,
        "is_weekend": 0, "is_rush_hour": 1, "is_night": 0, "season": 1,
        "duration_min": 30.0, "weather_severity_score": 1.0,
        "road_feature_count": 2, "Distance(mi)": 0.5,
        "Temperature(F)": 55.0, "Humidity(%)": 70.0,
        "Pressure(in)": 29.9, "Visibility(mi)": 10.0,
        "Wind_Speed(mph)": 10.0, "Precipitation(in)": 0.0,
    }
    result = predict(sample)
    print(result)

