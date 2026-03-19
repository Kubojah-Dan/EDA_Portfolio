"""
evaluate_model.py
Evaluates all trained models on the held-out test set.
Generates: ROC curves, confusion matrices, feature importance plots,
           calibration curves, and a final metrics JSON.
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.metrics import (
    roc_auc_score, roc_curve, average_precision_score,
    precision_recall_curve, confusion_matrix,
    classification_report, brier_score_loss,
)
from sklearn.calibration import calibration_curve

from src.utils.helpers import load_config, save_metadata
from src.utils.logger import get_logger

log = get_logger("evaluate_model")
sns.set_theme(style="whitegrid", palette="muted")


def load_test_data(cfg: dict):
    test_path = os.path.join(cfg["paths"]["models_dir"], "test_data.parquet")
    df = pd.read_parquet(test_path)
    y = df.pop("target")
    return df, y


def load_models(cfg: dict) -> dict:
    models_dir = cfg["paths"]["models_dir"]
    models = {}
    for fname in os.listdir(models_dir):
        if fname.startswith("model_") and fname.endswith(".pkl"):
            name = fname.replace("model_", "").replace(".pkl", "")
            models[name] = joblib.load(os.path.join(models_dir, fname))
    return models


def plot_roc_curves(models, X_test, y_test, fig_dir: str):
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, model in models.items():
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        ax.plot(fpr, tpr, lw=2, label=f"{name} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate", title="ROC Curves – Test Set")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "roc_curves.png"), dpi=150)
    plt.close()
    log.info("Saved roc_curves.png")


def plot_pr_curves(models, X_test, y_test, fig_dir: str):
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, model in models.items():
        y_prob = model.predict_proba(X_test)[:, 1]
        prec, rec, _ = precision_recall_curve(y_test, y_prob)
        ap = average_precision_score(y_test, y_prob)
        ax.plot(rec, prec, lw=2, label=f"{name} (AP={ap:.3f})")
    ax.set(xlabel="Recall", ylabel="Precision", title="Precision-Recall Curves – Test Set")
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "pr_curves.png"), dpi=150)
    plt.close()
    log.info("Saved pr_curves.png")


def plot_confusion_matrix(name, model, X_test, y_test, fig_dir: str):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Low", "High"], yticklabels=["Low", "High"])
    ax.set(xlabel="Predicted", ylabel="Actual", title=f"Confusion Matrix – {name}")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, f"cm_{name}.png"), dpi=150)
    plt.close()


def plot_feature_importance(name, model, feature_names, fig_dir: str, top_n: int = 20):
    importances = None
    # Handle Pipeline (LR)
    clf = model.named_steps["clf"] if hasattr(model, "named_steps") else model
    if hasattr(clf, "feature_importances_"):
        importances = clf.feature_importances_
    elif hasattr(clf, "coef_"):
        importances = np.abs(clf.coef_[0])

    if importances is None:
        return

    idx = np.argsort(importances)[-top_n:]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(np.array(feature_names)[idx], importances[idx], color="steelblue")
    ax.set(title=f"Feature Importance – {name} (Top {top_n})", xlabel="Importance")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, f"feat_imp_{name}.png"), dpi=150)
    plt.close()
    log.info(f"Saved feat_imp_{name}.png")


def plot_calibration(models, X_test, y_test, fig_dir: str):
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    for name, model in models.items():
        y_prob = model.predict_proba(X_test)[:, 1]
        frac_pos, mean_pred = calibration_curve(y_test, y_prob, n_bins=10)
        ax.plot(mean_pred, frac_pos, marker="o", lw=2, label=name)
    ax.set(xlabel="Mean Predicted Probability", ylabel="Fraction of Positives",
           title="Calibration Curves")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "calibration_curves.png"), dpi=150)
    plt.close()
    log.info("Saved calibration_curves.png")


def compute_metrics(name, model, X_test, y_test, threshold: float = 0.5) -> dict:
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)
    report = classification_report(y_test, y_pred, output_dict=True)
    return {
        "model": name,
        "test_auc": round(roc_auc_score(y_test, y_prob), 4),
        "test_ap": round(average_precision_score(y_test, y_prob), 4),
        "test_brier": round(brier_score_loss(y_test, y_prob), 4),
        "test_f1": round(report["weighted avg"]["f1-score"], 4),
        "test_precision": round(report["weighted avg"]["precision"], 4),
        "test_recall": round(report["weighted avg"]["recall"], 4),
        "test_accuracy": round(report["accuracy"], 4),
    }


def main():
    cfg = load_config()
    fig_dir = cfg["paths"]["figures_dir"]
    models_dir = cfg["paths"]["models_dir"]
    os.makedirs(fig_dir, exist_ok=True)

    X_test, y_test = load_test_data(cfg)
    models = load_models(cfg)

    if not models:
        log.error("No trained models found. Run train_model.py first.")
        return

    log.info(f"Evaluating {len(models)} models on {len(y_test):,} test samples")

    all_metrics = []
    for name, model in models.items():
        metrics = compute_metrics(name, model, X_test, y_test)
        all_metrics.append(metrics)
        log.info(f"{name}: AUC={metrics['test_auc']} | F1={metrics['test_f1']} | AP={metrics['test_ap']}")
        plot_confusion_matrix(name, model, X_test, y_test, fig_dir)
        plot_feature_importance(name, model, X_test.columns.tolist(), fig_dir)

    plot_roc_curves(models, X_test, y_test, fig_dir)
    plot_pr_curves(models, X_test, y_test, fig_dir)
    plot_calibration(models, X_test, y_test, fig_dir)

    save_metadata(
        {"test_metrics": all_metrics},
        os.path.join(models_dir, "test_metrics.json"),
    )

    best = max(all_metrics, key=lambda x: x["test_auc"])
    log.info(f"\nBest model on test set: {best['model']} (AUC={best['test_auc']})")


if __name__ == "__main__":
    main()

