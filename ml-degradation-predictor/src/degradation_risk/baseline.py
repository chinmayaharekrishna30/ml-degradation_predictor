"""The monitored business model: a hotel-booking cancellation classifier."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .config import CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Select model inputs with stable dtypes (missing columns become NaN)."""
    X = pd.DataFrame(index=df.index)
    for c in NUMERIC_FEATURES:
        X[c] = pd.to_numeric(df[c], errors="coerce").astype(float) if c in df else np.nan
    for c in CATEGORICAL_FEATURES:
        s = df[c].astype(object) if c in df else pd.Series(np.nan, index=df.index, dtype=object)
        X[c] = s.where(s.notna(), np.nan)
    return X


def build_baseline_pipeline(seed: int = 42) -> Pipeline:
    """Imputation + encoding + gradient boosting, as in report workflow step 2."""
    numeric = Pipeline([("impute", SimpleImputer(strategy="median", keep_empty_features=True))])
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
            ("encode", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=0.005, sparse_output=False)),
        ]
    )
    pre = ColumnTransformer([("num", numeric, NUMERIC_FEATURES), ("cat", categorical, CATEGORICAL_FEATURES)])
    clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, max_leaf_nodes=31, random_state=seed)
    return Pipeline([("pre", pre), ("clf", clf)])


def train_baseline(df_train: pd.DataFrame, seed: int = 42) -> Pipeline:
    model = build_baseline_pipeline(seed)
    model.fit(prepare_features(df_train), df_train[TARGET].to_numpy())
    return model


def classification_report_dict(y, proba, threshold: float = 0.5) -> dict:
    y = np.asarray(y)
    pred = (np.asarray(proba) >= threshold).astype(int)
    return {
        "n": int(len(y)),
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, proba)) if len(np.unique(y)) == 2 else float("nan"),
        "positive_rate": float(y.mean()),
    }
