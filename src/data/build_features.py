"""
build_features.py
Feature engineering pipeline:
  - Temporal features from Start_Time
  - Duration feature
  - Weather severity index
  - Road infrastructure score
  - Geographic clustering proxy
  - Binary target: Severity >= 3 -> 1 (High), else 0 (Low)
  - Encodes categoricals, saves processed parquet + feature list
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

from src.utils.helpers import load_config
from src.utils.logger import get_logger

log = get_logger("build_features")


def temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    df["Start_Time"] = pd.to_datetime(df["Start_Time"], errors="coerce")
    df["End_Time"] = pd.to_datetime(df["End_Time"], errors="coerce")

    df["hour"] = df["Start_Time"].dt.hour.astype("int8")
    df["day_of_week"] = df["Start_Time"].dt.dayofweek.astype("int8")   # 0=Mon
    df["month"] = df["Start_Time"].dt.month.astype("int8")
    df["year"] = df["Start_Time"].dt.year.astype("int16")
    df["is_weekend"] = (df["day_of_week"] >= 5).astype("int8")
    df["is_rush_hour"] = df["hour"].isin([7, 8, 9, 16, 17, 18]).astype("int8")
    df["is_night"] = df["hour"].apply(lambda h: 1 if h < 6 or h >= 21 else 0).astype("int8")
    df["season"] = df["month"].map({
        12: 0, 1: 0, 2: 0,   # Winter
        3: 1, 4: 1, 5: 1,    # Spring
        6: 2, 7: 2, 8: 2,    # Summer
        9: 3, 10: 3, 11: 3,  # Fall
    }).astype("int8")

    # Duration in minutes
    df["duration_min"] = (
        (df["End_Time"] - df["Start_Time"]).dt.total_seconds() / 60
    ).clip(0, 1440).astype("float32")

    return df


def weather_features(df: pd.DataFrame) -> pd.DataFrame:
    """Composite weather severity index (0–5)."""
    score = pd.Series(0.0, index=df.index)

    if "Temperature(F)" in df.columns:
        score += ((df["Temperature(F)"] < 32) | (df["Temperature(F)"] > 95)).astype(float)

    if "Visibility(mi)" in df.columns:
        score += (df["Visibility(mi)"] < 1.0).astype(float) * 2
        score += (df["Visibility(mi)"].between(1.0, 3.0)).astype(float)

    if "Wind_Speed(mph)" in df.columns:
        score += (df["Wind_Speed(mph)"] > 30).astype(float)

    if "Precipitation(in)" in df.columns:
        score += (df["Precipitation(in)"] > 0.1).astype(float)

    df["weather_severity_score"] = score.clip(0, 5).astype("float32")

    # Simplified weather category
    if "Weather_Condition" in df.columns:
        wc = df["Weather_Condition"].astype(str).str.lower()
        df["weather_category"] = np.select(
            [
                wc.str.contains("snow|sleet|ice|blizzard|wintry"),
                wc.str.contains("rain|drizzle|shower|thunder|storm"),
                wc.str.contains("fog|mist|haze|smoke|dust"),
                wc.str.contains("clear|fair|sunny"),
                wc.str.contains("cloud|overcast|partly"),
            ],
            ["Snow/Ice", "Rain/Storm", "Fog/Haze", "Clear", "Cloudy"],
            default="Other",
        )
        df["weather_category"] = df["weather_category"].astype("category")

    return df


def road_features(df: pd.DataFrame) -> pd.DataFrame:
    """Infrastructure score: count of road features present."""
    infra_cols = [
        "Amenity", "Bump", "Crossing", "Give_Way", "Junction",
        "No_Exit", "Railway", "Roundabout", "Station", "Stop",
        "Traffic_Calming", "Traffic_Signal", "Turning_Loop",
    ]
    present = [c for c in infra_cols if c in df.columns]
    df["road_feature_count"] = df[present].astype(int).sum(axis=1).astype("int8")
    return df


def encode_categoricals(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    le_map = {}
    cat_cols = ["State", "Timezone", "Wind_Direction", "Sunrise_Sunset",
                "Civil_Twilight", "Nautical_Twilight", "Astronomical_Twilight",
                "Source", "weather_category"]
    for col in cat_cols:
        if col not in df.columns:
            continue
        le = LabelEncoder()
        df[col + "_enc"] = le.fit_transform(df[col].astype(str)).astype("int16")
        le_map[col] = list(le.classes_)
    return df, le_map


def build_target(df: pd.DataFrame, threshold: int) -> pd.DataFrame:
    df["target"] = (df["Severity"] >= threshold).astype("int8")
    log.info(f"Target distribution:\n{df['target'].value_counts(normalize=True).round(3)}")
    return df


def select_features(df: pd.DataFrame) -> list[str]:
    exclude = {
        "ID", "Description", "Street", "City", "County", "Zipcode",
        "Airport_Code", "Start_Time", "End_Time", "Weather_Timestamp",
        "Country", "Severity", "target",
        # raw categoricals replaced by _enc versions
        "State", "Timezone", "Wind_Direction", "Sunrise_Sunset",
        "Civil_Twilight", "Nautical_Twilight", "Astronomical_Twilight",
        "Source", "Weather_Condition", "weather_category",
    }
    features = [c for c in df.columns if c not in exclude and df[c].dtype != object]
    return features


def main():
    cfg = load_config()
    interim_path = cfg["paths"]["interim_data"]
    out_path = cfg["paths"]["processed_data"]
    threshold = cfg["preprocessing"]["severity_binary_threshold"]

    log.info(f"Loading cleaned data from {interim_path}")
    df = pd.read_parquet(interim_path)
    log.info(f"Shape: {df.shape}")

    df = temporal_features(df)
    df = weather_features(df)
    df = road_features(df)
    df, le_map = encode_categoricals(df)
    df = build_target(df, threshold)

    features = select_features(df)
    log.info(f"Selected {len(features)} features: {features}")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_parquet(out_path, index=False, engine="pyarrow")
    log.info(f"Saved processed data -> {out_path}")

    # Save feature list and label encoders
    meta = {"features": features, "label_encoders": le_map, "target": "target"}
    meta_path = os.path.join(cfg["paths"]["models_dir"], "feature_metadata.json")
    os.makedirs(cfg["paths"]["models_dir"], exist_ok=True)
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    log.info(f"Saved feature metadata -> {meta_path}")


if __name__ == "__main__":
    main()

