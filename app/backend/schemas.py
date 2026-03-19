from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


class PredictionRequest(BaseModel):
    model_config = {"populate_by_name": True, "protected_namespaces": ()}

    hour: int = Field(ge=0, le=23)
    day_of_week: int = Field(ge=0, le=6)
    month: int = Field(ge=1, le=12)
    year: int = Field(ge=2016, le=2030)
    is_weekend: int = Field(ge=0, le=1)
    is_rush_hour: int = Field(ge=0, le=1)
    is_night: int = Field(ge=0, le=1)
    season: int = Field(ge=0, le=3)
    duration_min: float = Field(ge=0)
    weather_severity_score: float = Field(ge=0, le=5)
    road_feature_count: int = Field(ge=0)
    distance_mi: float = Field(ge=0, alias="Distance(mi)")
    temperature_f: float = Field(alias="Temperature(F)")
    humidity_pct: float = Field(ge=0, le=100, alias="Humidity(%)")
    pressure_in: float = Field(alias="Pressure(in)")
    visibility_mi: float = Field(ge=0, alias="Visibility(mi)")
    wind_speed_mph: float = Field(ge=0, alias="Wind_Speed(mph)")
    precipitation_in: float = Field(ge=0, alias="Precipitation(in)")
    selected_model: Optional[str] = Field(default=None, alias="model_name")


class PredictionResponse(BaseModel):
    prediction: int
    severity_label: str
    probability: float
    model: str


class AccidentStats(BaseModel):
    total_accidents: int
    by_severity: dict
    by_state: dict
    avg_temperature: float
    avg_visibility: float
    date_range: dict


class ModelMetricsResponse(BaseModel):
    model_config = {"protected_namespaces": ()}

    model_name: str
    test_auc: float
    test_f1: float
    test_precision: float
    test_recall: float
    test_accuracy: float
    test_ap: float


class HealthResponse(BaseModel):
    status: str
    version: str
    models_loaded: list[str]
    timestamp: datetime
