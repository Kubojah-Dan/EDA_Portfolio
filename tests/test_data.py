import os
import sys
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.clean_data import (
    drop_high_missing, impute_missing, remove_outliers,
    validate_coordinates, validate_severity,
)
from src.data.build_features import (
    temporal_features, weather_features, road_features, build_target,
)


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "ID": ["A-1", "A-2", "A-3", "A-4"],
        "Severity": [1, 2, 3, 4],
        "Start_Lat": [39.8, 34.0, 41.5, 25.0],
        "Start_Lng": [-84.0, -118.0, -87.6, -80.0],
        "Start_Time": pd.to_datetime(["2022-03-08 08:00", "2022-06-15 17:30",
                                       "2022-12-01 02:00", "2022-09-20 12:00"]),
        "End_Time": pd.to_datetime(["2022-03-08 08:30", "2022-06-15 18:00",
                                     "2022-12-01 02:45", "2022-09-20 12:20"]),
        "Temperature(F)": [55.0, 85.0, 28.0, 90.0],
        "Humidity(%)": [70.0, 50.0, 90.0, 60.0],
        "Visibility(mi)": [10.0, 8.0, 0.5, 10.0],
        "Wind_Speed(mph)": [10.0, 5.0, 35.0, 8.0],
        "Precipitation(in)": [0.0, 0.0, 0.2, 0.0],
        "Weather_Condition": ["Clear", "Sunny", "Snow", "Partly Cloudy"],
        "Junction": [False, True, False, True],
        "Traffic_Signal": [True, False, False, True],
        "Crossing": [False, False, True, False],
        "Give_Way": [False, False, False, False],
        "Bump": [False, False, False, False],
        "No_Exit": [False, False, False, False],
        "Railway": [False, False, False, False],
        "Roundabout": [False, False, False, False],
        "Station": [False, False, False, False],
        "Stop": [False, False, False, False],
        "Traffic_Calming": [False, False, False, False],
        "Turning_Loop": [False, False, False, False],
        "Amenity": [False, False, False, False],
    })


def test_drop_high_missing(sample_df):
    sample_df["mostly_null"] = [np.nan, np.nan, np.nan, 1.0]
    result = drop_high_missing(sample_df, threshold=0.5)
    assert "mostly_null" not in result.columns


def test_impute_missing(sample_df):
    sample_df.loc[0, "Temperature(F)"] = np.nan
    result = impute_missing(sample_df)
    assert result["Temperature(F)"].isnull().sum() == 0


def test_validate_coordinates(sample_df):
    sample_df.loc[0, "Start_Lat"] = 0.0  # invalid
    result = validate_coordinates(sample_df)
    assert len(result) == 3


def test_validate_severity(sample_df):
    sample_df.loc[0, "Severity"] = 5  # invalid
    result = validate_severity(sample_df)
    assert len(result) == 3


def test_remove_outliers(sample_df):
    # Build a larger series so IQR capping is meaningful
    df_large = pd.concat([sample_df] * 20, ignore_index=True)
    df_large.loc[0, "Wind_Speed(mph)"] = 9999.0
    result = remove_outliers(df_large, factor=1.5)
    assert result["Wind_Speed(mph)"].max() < 9999.0


def test_temporal_features(sample_df):
    result = temporal_features(sample_df)
    assert "hour" in result.columns
    assert "day_of_week" in result.columns
    assert "is_weekend" in result.columns
    assert "is_rush_hour" in result.columns
    assert "duration_min" in result.columns
    assert result["duration_min"].min() >= 0


def test_weather_features(sample_df):
    result = weather_features(sample_df)
    assert "weather_severity_score" in result.columns
    assert result["weather_severity_score"].between(0, 5).all()


def test_road_features(sample_df):
    result = road_features(sample_df)
    assert "road_feature_count" in result.columns
    assert result["road_feature_count"].min() >= 0


def test_build_target(sample_df):
    result = build_target(sample_df, threshold=3)
    assert "target" in result.columns
    assert set(result["target"].unique()).issubset({0, 1})
    # Severity 3 and 4 -> 1
    assert result.loc[result["Severity"] >= 3, "target"].eq(1).all()
    assert result.loc[result["Severity"] < 3, "target"].eq(0).all()

