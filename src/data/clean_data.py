"""
clean_data.py
Cleans the interim dataset:
  - Drops high-missing columns
  - Imputes remaining missing values
  - Removes duplicates & outliers
  - Standardises string columns
  - Saves cleaned parquet
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pandas as pd
import numpy as np

from src.utils.helpers import load_config
from src.utils.logger import get_logger

log = get_logger("clean_data")


def drop_high_missing(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    missing_frac = df.isnull().mean()
    drop_cols = missing_frac[missing_frac > threshold].index.tolist()
    log.info(f"Dropping {len(drop_cols)} columns with >{threshold*100:.0f}% missing: {drop_cols}")
    return df.drop(columns=drop_cols)


def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    num_cols = df.select_dtypes(include=[np.number]).columns
    cat_cols = df.select_dtypes(include=["category", "object", "string"]).columns

    # Numeric: median imputation
    for col in num_cols:
        if df[col].isnull().any():
            median = df[col].median()
            df[col] = df[col].fillna(median)

    # Categorical: mode imputation
    for col in cat_cols:
        if df[col].isnull().any():
            mode = df[col].mode()
            if len(mode) > 0:
                df[col] = df[col].fillna(mode[0])

    # Boolean: fill with False
    bool_cols = df.select_dtypes(include=["boolean", "bool"]).columns
    for col in bool_cols:
        df[col] = df[col].fillna(False)

    return df


def remove_outliers(df: pd.DataFrame, factor: float) -> pd.DataFrame:
    """IQR-based outlier capping (not removal) for key numeric columns."""
    clip_cols = ["Distance(mi)", "Temperature(F)", "Humidity(%)",
                 "Pressure(in)", "Visibility(mi)", "Wind_Speed(mph)", "Precipitation(in)"]
    for col in clip_cols:
        if col not in df.columns:
            continue
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - factor * iqr, q3 + factor * iqr
        df[col] = df[col].clip(lower=lower, upper=upper)
    return df


def clean_strings(df: pd.DataFrame) -> pd.DataFrame:
    for col in ["City", "County", "Street"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.title()
    if "Weather_Condition" in df.columns:
        df["Weather_Condition"] = df["Weather_Condition"].astype(str).str.strip().str.title()
    return df


def validate_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows with impossible lat/lng values."""
    mask = (
        df["Start_Lat"].between(24.0, 50.0) &
        df["Start_Lng"].between(-125.0, -66.0)
    )
    removed = (~mask).sum()
    if removed:
        log.info(f"Removing {removed:,} rows with invalid coordinates")
    return df[mask].reset_index(drop=True)


def validate_severity(df: pd.DataFrame) -> pd.DataFrame:
    mask = df["Severity"].between(1, 4)
    removed = (~mask).sum()
    if removed:
        log.info(f"Removing {removed:,} rows with invalid Severity")
    return df[mask].reset_index(drop=True)


def main():
    cfg = load_config()
    interim_path = cfg["paths"]["interim_data"]
    out_path = cfg["paths"]["interim_data"].replace("accidents_cleaned", "accidents_cleaned_v2")
    # We overwrite the same interim file
    out_path = interim_path

    log.info(f"Loading interim data from {interim_path}")
    df = pd.read_parquet(interim_path)
    log.info(f"Shape before cleaning: {df.shape}")

    df = df.drop_duplicates(subset=["ID"])
    log.info(f"After dedup: {df.shape}")

    df = drop_high_missing(df, cfg["preprocessing"]["drop_threshold_missing"])
    df = validate_coordinates(df)
    df = validate_severity(df)
    df = remove_outliers(df, cfg["preprocessing"]["outlier_iqr_factor"])
    df = impute_missing(df)
    df = clean_strings(df)

    log.info(f"Shape after cleaning: {df.shape}")
    log.info(f"Missing values remaining: {df.isnull().sum().sum()}")

    df.to_parquet(out_path, index=False, engine="pyarrow")
    log.info(f"Saved cleaned data -> {out_path}")


if __name__ == "__main__":
    main()

