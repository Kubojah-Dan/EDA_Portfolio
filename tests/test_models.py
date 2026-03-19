import os
import sys
import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture
def binary_dataset():
    X, y = make_classification(
        n_samples=500, n_features=20, n_informative=10,
        n_redundant=5, weights=[0.7, 0.3], random_state=42,
    )
    return pd.DataFrame(X, columns=[f"f{i}" for i in range(20)]), pd.Series(y)


def test_logistic_regression_trains(binary_dataset):
    X, y = binary_dataset
    model = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=200))])
    model.fit(X, y)
    preds = model.predict(X)
    assert len(preds) == len(y)
    assert set(preds).issubset({0, 1})


def test_random_forest_trains(binary_dataset):
    X, y = binary_dataset
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)
    probs = model.predict_proba(X)
    assert probs.shape == (len(y), 2)
    assert np.allclose(probs.sum(axis=1), 1.0)


def test_xgboost_trains(binary_dataset):
    pytest.importorskip("xgboost")
    from xgboost import XGBClassifier
    X, y = binary_dataset
    model = XGBClassifier(n_estimators=10, random_state=42, verbosity=0)
    model.fit(X, y)
    preds = model.predict(X)
    assert len(preds) == len(y)


def test_lightgbm_trains(binary_dataset):
    pytest.importorskip("lightgbm")
    from lightgbm import LGBMClassifier
    X, y = binary_dataset
    model = LGBMClassifier(n_estimators=10, random_state=42, verbose=-1)
    model.fit(X, y)
    preds = model.predict(X)
    assert len(preds) == len(y)


def test_model_auc_above_baseline(binary_dataset):
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split
    X, y = binary_dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)
    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    assert auc > 0.6, f"AUC {auc:.3f} is below acceptable threshold"


def test_feature_importance_available(binary_dataset):
    X, y = binary_dataset
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)
    assert hasattr(model, "feature_importances_")
    assert len(model.feature_importances_) == X.shape[1]


def test_predict_proba_range(binary_dataset):
    X, y = binary_dataset
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)
    probs = model.predict_proba(X)[:, 1]
    assert probs.min() >= 0.0
    assert probs.max() <= 1.0

