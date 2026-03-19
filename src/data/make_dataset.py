"""
make_dataset.py
Loads the raw US Accidents CSV (chunked for memory efficiency) and saves
a typed interim Parquet file.
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pandas as pd
from tqdm import tqdm

from src.utils.helpers import load_config, reduce_mem_usage
from src.utils.logger import get_logger

log = get_logger("make_dataset")

DTYPE_MAP = {
    "ID": "string",
    "Source": "category",
    "Severity": "int8",
    "Start_Lat": "float32",
    "Start_Lng": "float32",
    "End_Lat": "float32",
    "End_Lng": "float32",
    "Distance(mi)": "float32",
    "Description": "string",
    "Street": "string",
    "City": "string",
    "County": "string",
    "State": "category",
    "Zipcode": "string",
    "Country": "category",
    "Timezone": "category",
    "Airport_Code": "string",
    "Temperature(F)": "float32",
    "Wind_Chill(F)": "float32",
    "Humidity(%)": "float32",
    "Pressure(in)": "float32",
    "Visibility(mi)": "float32",
    "Wind_Direction": "category",
    "Wind_Speed(mph)": "float32",
    "Precipitation(in)": "float32",
    "Weather_Condition": "category",
    "Amenity": "bool",
    "Bump": "bool",
    "Crossing": "bool",
    "Give_Way": "bool",
    "Junction": "bool",
    "No_Exit": "bool",
    "Railway": "bool",
    "Roundabout": "bool",
    "Station": "bool",
    "Stop": "bool",
    "Traffic_Calming": "bool",
    "Traffic_Signal": "bool",
    "Turning_Loop": "bool",
    "Sunrise_Sunset": "category",
    "Civil_Twilight": "category",
    "Nautical_Twilight": "category",
    "Astronomical_Twilight": "category",
}

BOOL_COLS = [k for k, v in DTYPE_MAP.items() if v == "bool"]
PARSE_DATES = ["Start_Time", "End_Time", "Weather_Timestamp"]


def load_raw(cfg: dict) -> pd.DataFrame:
    raw_path = cfg["paths"]["raw_data"]
    sample_size = cfg["data"]["sample_size"]
    chunksize = cfg["data"]["chunksize"]

    log.info(f"Reading {raw_path} in chunks of {chunksize:,}...")
    chunks = []
    total_rows = 0

    reader = pd.read_csv(
        raw_path,
        dtype={k: v for k, v in DTYPE_MAP.items() if v not in ("bool",)},
        parse_dates=PARSE_DATES,
        chunksize=chunksize,
        low_memory=False,
        on_bad_lines="skip",
    )

    for chunk in tqdm(reader, desc="Loading chunks"):
        # Convert bool columns (stored as True/False strings)
        for col in BOOL_COLS:
            if col in chunk.columns:
                chunk[col] = chunk[col].map({"True": True, "False": False}).astype("boolean")
        chunks.append(chunk)
        total_rows += len(chunk)
        if sample_size and total_rows >= sample_size:
            break

    df = pd.concat(chunks, ignore_index=True)
    if sample_size:
        df = df.iloc[:sample_size]

    log.info(f"Loaded {len(df):,} rows × {df.shape[1]} columns")
    return df


def save_interim(df: pd.DataFrame, cfg: dict) -> None:
    out_path = cfg["paths"]["interim_data"]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    # Ensure datetime columns are proper dtype before saving
    for col in PARSE_DATES:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    # Convert pandas StringDtype to object for pyarrow compatibility
    for col in df.select_dtypes(include=["string"]).columns:
        df[col] = df[col].astype(object)
    df = reduce_mem_usage(df)
    df.to_parquet(out_path, index=False, engine="pyarrow")
    log.info(f"Saved interim data -> {out_path}  ({os.path.getsize(out_path)/1e6:.1f} MB)")


def main():
    cfg = load_config()
    df = load_raw(cfg)
    save_interim(df, cfg)


if __name__ == "__main__":
    main()

