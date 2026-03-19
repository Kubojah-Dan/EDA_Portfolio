"""
train_model.py
Trains multiple classifiers on the processed US Accidents dataset.
Models: Logistic Regression, Random Forest, XGBoost, LightGBM
Handles class imbalance, saves models + metadata.
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import json
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, classification_report
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import joblib

from src.utils.helpers import load_config, load_model_config, save_metadata
from src.utils.logger import get_logger

log = get_logger("train_model")


def load_data(cfg: dict) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    processed_path = cfg["paths"]["processed_data"]
    meta_path = os.path.join(cfg["paths"]["models_dir"], "feature_metadata.json")

    log.info(f"Loading processed data from {processed_path}")
    df = pd.read_parquet(processed_path)

    with open(meta_path) as f:
        meta = json.load(f)

    features = meta["features"]
    # Keep only features that exist in df
    features = [f for f in features if f in df.columns]

    X = df[features].copy()
    y = df["target"].copy()

    # Final safety: drop any remaining NaN
    mask = X.notna().all(axis=1)
    X, y = X[mask], y[mask]

    log.info(f"Dataset: {X.shape[0]:,} rows, {X.shape[1]} features")
    log.info(f"Class distribution: {y.value_counts().to_dict()}")
    return X, y, features


def split_data(X, y, cfg):
    rs = cfg["data"]["random_state"]
    test_size = cfg["data"]["test_size"]
    val_size = cfg["data"]["val_size"]

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=rs, stratify=y
    )
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio, random_state=rs, stratify=y_temp
    )
    log.info(f"Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")
    return X_train, X_val, X_test, y_train, y_val, y_test


def build_models(mcfg: dict) -> dict:
    models = {}

    if mcfg["models"]["logistic_regression"]["enabled"]:
        p = mcfg["models"]["logistic_regression"]["params"]
        models["logistic_regression"] = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(**p)),
        ])

    if mcfg["models"]["random_forest"]["enabled"]:
        p = mcfg["models"]["random_forest"]["params"]
        models["random_forest"] = RandomForestClassifier(**p)

    if mcfg["models"]["xgboost"]["enabled"]:
        p = mcfg["models"]["xgboost"]["params"].copy()
        p.pop("use_label_encoder", None)
        models["xgboost"] = XGBClassifier(**p, tree_method="hist", device="cpu")

    if mcfg["models"]["lightgbm"]["enabled"]:
        p = mcfg["models"]["lightgbm"]["params"]
        models["lightgbm"] = LGBMClassifier(**p)

    return models


def train_and_evaluate(
    name: str,
    model,
    X_train, y_train,
    X_val, y_val,
    models_dir: str,
) -> dict:
    log.info(f"Training {name}...")
    t0 = time.time()

    fit_kwargs = {}
    if name == "xgboost":
        fit_kwargs = {"eval_set": [(X_val, y_val)], "verbose": 100}
    elif name == "lightgbm":
        fit_kwargs = {"eval_set": [(X_val, y_val)]}

    model.fit(X_train, y_train, **fit_kwargs)
    elapsed = time.time() - t0

    # Validation metrics
    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, y_prob)
    report = classification_report(y_val, y_pred, output_dict=True)

    log.info(f"{name} | AUC={auc:.4f} | F1={report['weighted avg']['f1-score']:.4f} | time={elapsed:.1f}s")

    # Save model
    model_path = os.path.join(models_dir, f"model_{name}.pkl")
    joblib.dump(model, model_path)

    return {
        "model_name": name,
        "val_auc": round(auc, 4),
        "val_f1": round(report["weighted avg"]["f1-score"], 4),
        "val_precision": round(report["weighted avg"]["precision"], 4),
        "val_recall": round(report["weighted avg"]["recall"], 4),
        "train_time_sec": round(elapsed, 2),
        "model_path": model_path,
        "n_train": len(y_train),
        "n_val": len(y_val),
    }


def save_test_split(X_test, y_test, cfg):
    """Persist test set for evaluation script."""
    test_path = os.path.join(cfg["paths"]["models_dir"], "test_data.parquet")
    test_df = X_test.copy()
    test_df["target"] = y_test.values
    test_df.to_parquet(test_path, index=False)
    log.info(f"Saved test split -> {test_path}")


def main():
    cfg = load_config()
    mcfg = load_model_config()
    models_dir = cfg["paths"]["models_dir"]
    os.makedirs(models_dir, exist_ok=True)

    X, y, features = load_data(cfg)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y, cfg)
    save_test_split(X_test, y_test, cfg)

    models = build_models(mcfg)
    all_results = []

    for name, model in models.items():
        result = train_and_evaluate(name, model, X_train, y_train, X_val, y_val, models_dir)
        all_results.append(result)

    # Save training summary
    summary = {
        "features": features,
        "n_features": len(features),
        "class_distribution": y.value_counts().to_dict(),
        "results": all_results,
    }
    save_metadata(summary, os.path.join(models_dir, "training_summary.json"))

    log.info("\n=== Training Summary ===")
    for r in sorted(all_results, key=lambda x: x["val_auc"], reverse=True):
        log.info(f"  {r['model_name']:25s} AUC={r['val_auc']:.4f}  F1={r['val_f1']:.4f}")

    best = max(all_results, key=lambda x: x["val_auc"])
    log.info(f"\nBest model: {best['model_name']} (AUC={best['val_auc']:.4f})")


if __name__ == "__main__":
    main()

