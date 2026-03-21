import os
import sys
import json
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from app.backend.schemas import (
    PredictionRequest, PredictionResponse,
    AccidentStats, ModelMetricsResponse, HealthResponse,
)
from app.backend.database import engine
from app.backend import models as db_models
from src.utils.helpers import load_config
from src.utils.logger import get_logger

log = get_logger("api")

# ── App-level state ───────────────────────────────────────────────────────────
_state: dict = {
    "df": None,
    "metrics": None,
    "predict_fn": None,
    "models_loaded": [],
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-load everything at startup so first requests are fast."""
    cfg = load_config()

    # DB tables 
    try:
        db_models.Base.metadata.create_all(bind=engine)
    except Exception as e:
        log.warning(f"DB not available, skipping table creation: {e}")

    # Pre-load processed dataframe
    path = cfg["paths"]["processed_data"]
    if os.path.exists(path):
        log.info("Pre-loading processed data...")
        _state["df"] = pd.read_parquet(path)
        log.info(f"Loaded processed data: {_state['df'].shape}")
    else:
        log.warning("Processed data not found. Run the pipeline first.")

    # Pre-load metrics
    metrics_path = os.path.join(cfg["paths"]["models_dir"], "test_metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            _state["metrics"] = json.load(f)["test_metrics"]

    # Pre-load best model
    models_dir = cfg["paths"]["models_dir"]
    if os.path.exists(models_dir):
        _state["models_loaded"] = [
            f.replace("model_", "").replace(".pkl", "")
            for f in os.listdir(models_dir)
            if f.startswith("model_") and f.endswith(".pkl")
        ]
        if _state["models_loaded"]:
            try:
                from src.models.predict_model import predict
                # Warm up: trigger model load
                _state["predict_fn"] = predict
                log.info(f"Models available: {_state['models_loaded']}")
            except Exception as e:
                log.warning(f"Could not pre-load predict function: {e}")

    yield  

    log.info("Shutting down API.")


app = FastAPI(
    title="US Accidents ML API",
    description="EDA Portfolio API - accident severity prediction & analytics",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _get_df() -> pd.DataFrame:
    if _state["df"] is None:
        raise HTTPException(503, "Processed data not loaded. Run the pipeline first.")
    return _state["df"]


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
def health_check():
    return HealthResponse(
        status="ok",
        version="1.0.0",
        models_loaded=_state["models_loaded"],
        timestamp=datetime.utcnow(),
    )


@app.post("/predict", response_model=PredictionResponse, tags=["ML"])
def predict_severity(request: PredictionRequest):
    if _state["predict_fn"] is None:
        raise HTTPException(503, "No trained models found. Run train_model.py first.")
    features = request.model_dump(by_alias=True)
    features.pop("model_name", None)
    features.pop("selected_model", None)
    model_name = request.selected_model
    try:
        result = _state["predict_fn"](features, model_name=model_name)
        return PredictionResponse(**result)
    except FileNotFoundError as e:
        raise HTTPException(503, str(e))
    except Exception as e:
        log.error(f"Prediction error: {e}")
        raise HTTPException(500, f"Prediction failed: {e}")


@app.get("/stats", response_model=AccidentStats, tags=["Analytics"])
def get_stats():
    df = _get_df()
    by_severity = df["Severity"].value_counts().to_dict() if "Severity" in df.columns else {}
    by_state = {}
    if "State" in df.columns:
        by_state = df["State"].value_counts().head(10).to_dict()
    elif "State_enc" in df.columns:
        by_state = df["State_enc"].value_counts().head(10).to_dict()
    return AccidentStats(
        total_accidents=len(df),
        by_severity={str(k): int(v) for k, v in by_severity.items()},
        by_state={str(k): int(v) for k, v in by_state.items()},
        avg_temperature=round(float(df["Temperature(F)"].mean()), 2) if "Temperature(F)" in df.columns else 0.0,
        avg_visibility=round(float(df["Visibility(mi)"].mean()), 2) if "Visibility(mi)" in df.columns else 0.0,
        date_range={
            "start": str(int(df["year"].min())) if "year" in df.columns else "N/A",
            "end": str(int(df["year"].max())) if "year" in df.columns else "N/A",
        },
    )


@app.get("/metrics", response_model=list[ModelMetricsResponse], tags=["ML"])
def get_model_metrics():
    if not _state["metrics"]:
        raise HTTPException(404, "No model metrics found. Run evaluate_model.py first.")
    return [
        ModelMetricsResponse(
            model_name=m.get("model_name") or m.get("model", ""),
            **{k: v for k, v in m.items() if k not in ("model", "model_name", "test_brier")}
        )
        for m in _state["metrics"]
    ]


@app.get("/eda/severity-by-hour", tags=["EDA"])
def severity_by_hour():
    df = _get_df()
    if "hour" not in df.columns:
        raise HTTPException(404, "hour feature not found")
    if "Severity" in df.columns:
        result = df.groupby(["hour", "Severity"]).size().reset_index(name="count")
    else:
        result = df.groupby("hour").size().reset_index(name="count")
    return result.to_dict(orient="records")


@app.get("/eda/severity-by-state", tags=["EDA"])
def severity_by_state():
    df = _get_df()
    if "State" not in df.columns:
        raise HTTPException(404, "State column not found")
    result = df.groupby("State")["Severity"].agg(["mean", "count"]).reset_index()
    result.columns = ["state", "avg_severity", "count"]
    return result.to_dict(orient="records")


@app.get("/eda/accidents-over-time", tags=["EDA"])
def accidents_over_time():
    df = _get_df()
    if "year" not in df.columns or "month" not in df.columns:
        raise HTTPException(404, "Temporal features not found")
    result = df.groupby(["year", "month"]).size().reset_index(name="count")
    result["period"] = result["year"].astype(str) + "-" + result["month"].astype(str).str.zfill(2)
    return result[["period", "count"]].to_dict(orient="records")


@app.get("/eda/weather-impact", tags=["EDA"])
def weather_impact():
    df = _get_df()
    if "weather_severity_score" not in df.columns:
        raise HTTPException(404, "weather_severity_score not found")
    result = df.groupby("weather_severity_score")["Severity"].agg(["mean", "count"]).reset_index()
    result.columns = ["weather_score", "avg_severity", "count"]
    return result.to_dict(orient="records")


@app.get("/eda/top-cities", tags=["EDA"])
def top_cities(top_n: int = Query(15, ge=5, le=50)):
    df = _get_df()
    if "City" not in df.columns:
        raise HTTPException(404, "City column not found")
    result = df.groupby(["City", "State"]).size().reset_index(name="count")
    result["city_state"] = result["City"] + ", " + result["State"]
    return result.nlargest(top_n, "count")[["city_state", "count"]].to_dict(orient="records")


@app.get("/eda/road-features", tags=["EDA"])
def road_features_impact():
    df = _get_df()
    if "road_feature_count" not in df.columns:
        raise HTTPException(404, "road_feature_count column not found")
    dist = df["road_feature_count"].value_counts().sort_index().reset_index()
    dist.columns = ["feature_count", "accident_count"]
    if "Severity" in df.columns:
        avg_sev = (
            df.groupby("road_feature_count")["Severity"]
            .mean()
            .reset_index()
            .rename(columns={"road_feature_count": "feature_count", "Severity": "avg_severity"})
        )
        dist = dist.merge(avg_sev, on="feature_count", how="left")
    else:
        dist["avg_severity"] = 0.0
    dist["avg_severity"] = dist["avg_severity"].round(3)
    return dist.to_dict(orient="records")
