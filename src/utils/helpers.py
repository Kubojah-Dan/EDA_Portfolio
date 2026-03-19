import os
import yaml
import joblib
import json
import time
from functools import wraps
from typing import Any

import pandas as pd
import numpy as np


def load_config(path: str = "configs/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_model_config(path: str = "configs/model_config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def save_model(model: Any, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)


def load_model(path: str) -> Any:
    return joblib.load(path)


def save_metadata(metadata: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)


def load_metadata(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        print(f"[{func.__name__}] completed in {elapsed:.2f}s")
        return result
    return wrapper


def reduce_mem_usage(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast numeric columns to reduce memory footprint."""
    for col in df.select_dtypes(include=[np.number]).columns:
        col_min, col_max = df[col].min(), df[col].max()
        if pd.api.types.is_integer_dtype(df[col]):
            for dtype in [np.int8, np.int16, np.int32]:
                if col_min >= np.iinfo(dtype).min and col_max <= np.iinfo(dtype).max:
                    df[col] = df[col].astype(dtype)
                    break
        else:
            if col_min >= np.finfo(np.float32).min and col_max <= np.finfo(np.float32).max:
                df[col] = df[col].astype(np.float32)
    return df


def get_project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

