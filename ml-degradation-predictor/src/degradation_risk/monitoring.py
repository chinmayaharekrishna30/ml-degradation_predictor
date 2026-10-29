"""Monitoring signals: drift, data quality, prediction distribution, realized performance.

One call to :func:`summarise_window` turns the raw predictions of one monitoring
window into one row of the monitoring table (report section 5).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from .config import CATEGORICAL_FEATURES, NUMERIC_FEATURES, Config

EPS = 1e-4
BINS = 10
DISCRETE_MAX_LEVELS = 12   # numeric features with few values are treated as categories
KS_SAMPLE = 5000
OTHER = "__other__"
MISSING = "missing"

# Inputs of the degradation-risk classifier. Everything here is known at
# prediction time: current-window signals plus *lagged* realized performance.
RISK_FEATURES = [
    "missing_rate",
    "feature_drift_score",
    "max_psi",
    "share_drifted",
    "prediction_mean",
    "prediction_std",
    "psi_prediction",
    "low_confidence_rate",
    "pred_mean_shift",
    "metric_change",
    "f1_roll3",
    "drift_roll3",
    "drift_delta",
    "missing_delta",
    "psi_pred_roll3",
]

# Columns that describe the future or the current window's realized outcome.
# They may build the target, but must never be model inputs.
FORBIDDEN_AS_FEATURES = {
    "degradation_label", "future_min_f1", "degraded_now", "realized_f1", "realized_auc",
    "realized_precision", "realized_recall", "actual_cancel_rate", "realized_f1_valid",
}


def check_no_leakage(features: list[str]) -> None:
    bad = sorted(set(features) & FORBIDDEN_AS_FEATURES)
    if bad:
        raise ValueError(f"Leakage: future/outcome columns used as risk features: {bad}")


# ---------------------------------------------------------------------------
# Drift statistics
# ---------------------------------------------------------------------------
def psi(p_ref, p_cur) -> float:
    """Population Stability Index between two discrete distributions."""
    a = np.clip(np.asarray(p_ref, dtype=float), EPS, None)
    b = np.clip(np.asarray(p_cur, dtype=float), EPS, None)
    a, b = a / a.sum(), b / b.sum()
    return float(np.sum((b - a) * np.log(b / a)))


def _inner_edges(x: np.ndarray, bins: int = BINS) -> np.ndarray:
    return np.unique(np.quantile(x, np.linspace(0, 1, bins + 1)[1:-1]))


def _bin_counts(x: np.ndarray, inner: np.ndarray) -> np.ndarray:
    return np.bincount(np.searchsorted(inner, x, side="right"), minlength=len(inner) + 1).astype(float)


def _levels(series: pd.Series, numeric: bool) -> pd.Series:
    if numeric:
        s = pd.to_numeric(series, errors="coerce").astype(float).astype(object)
    else:
        s = series.astype(object)
    return s.where(series.notna() if not numeric else s.notna(), MISSING).astype(str)


def _level_counts(levels: pd.Series, ref_levels: list[str]) -> np.ndarray:
    vc = levels.value_counts()
    known = [float(vc.get(l, 0)) for l in ref_levels]
    other = float(vc[~vc.index.isin(ref_levels)].sum())
    return np.array(known + [other])


@dataclass
class ReferenceProfile:
    """Everything needed to compare a new window against the reference period."""

    features: list
    continuous: list
    discrete: list
    numeric_flags: dict
    inner_edges: dict = field(default_factory=dict)
    edge_props: dict = field(default_factory=dict)
    ks_sample: dict = field(default_factory=dict)
    levels: dict = field(default_factory=dict)
    level_counts: dict = field(default_factory=dict)
    pred_edges: np.ndarray = None
    pred_props: np.ndarray = None
    pred_mean: float = 0.0
    pred_std: float = 0.0
    low_conf_rate: float = 0.0
    missing_rate: float = 0.0
    f1_ref: float = 0.0
    auc_ref: float = float("nan")
    f1_window_std: float = float("nan")
    n_ref: int = 0


def build_reference_profile(df_ref: pd.DataFrame, proba_ref, y_ref, cfg: Config, f1_ref: float,
                            f1_window_std: float = float("nan")) -> ReferenceProfile:
    features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    numeric_flags = {f: f in NUMERIC_FEATURES for f in features}
    continuous, discrete = [], []
    for f in features:
        if f in NUMERIC_FEATURES and df_ref[f].nunique(dropna=True) > DISCRETE_MAX_LEVELS:
            continuous.append(f)
        else:
            discrete.append(f)
    prof = ReferenceProfile(features=features, continuous=continuous, discrete=discrete, numeric_flags=numeric_flags)
    rng = np.random.default_rng(cfg.seed)
    for f in continuous:
        x = pd.to_numeric(df_ref[f], errors="coerce").dropna().to_numpy(float)
        inner = _inner_edges(x)
        prof.inner_edges[f] = inner
        counts = _bin_counts(x, inner)
        prof.edge_props[f] = counts / counts.sum()
        prof.ks_sample[f] = x if len(x) <= KS_SAMPLE else rng.choice(x, KS_SAMPLE, replace=False)
    for f in discrete:
        lv = _levels(df_ref[f], numeric_flags[f])
        ref_levels = sorted(lv.unique().tolist())
        prof.levels[f] = ref_levels
        prof.level_counts[f] = _level_counts(lv, ref_levels)
    p = np.asarray(proba_ref, dtype=float)
    prof.pred_edges = _inner_edges(p)
    counts = _bin_counts(p, prof.pred_edges)
    prof.pred_props = counts / counts.sum()
    prof.pred_mean, prof.pred_std = float(p.mean()), float(p.std())
    prof.low_conf_rate = float(np.mean(np.abs(p - 0.5) < cfg.low_conf_band))
    prof.missing_rate = float(df_ref[features].isna().to_numpy().mean())
    prof.f1_ref = float(f1_ref)
    prof.f1_window_std = float(f1_window_std)
    y = np.asarray(y_ref)
    prof.auc_ref = float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan")
    prof.n_ref = int(len(p))
    return prof


# ---------------------------------------------------------------------------
# One window -> one row
# ---------------------------------------------------------------------------
def summarise_window(win: pd.DataFrame, proba, y, profile: ReferenceProfile, cfg: Config) -> dict:
    """Summarise one monitoring window. ``y`` may be ``None`` when labels are not yet known."""
    proba = np.asarray(proba, dtype=float)
    out: dict = {"n_predictions": int(len(win))}
    n_feat = len(profile.features)
    alpha = 0.05 / n_feat   # Bonferroni: many features are tested at once
    psis, drifted = [], 0

    for f in profile.continuous:
        x = pd.to_numeric(win[f], errors="coerce").dropna().to_numpy(float)
        if len(x) < 5:
            psis.append(0.0)
            continue
        counts = _bin_counts(x, profile.inner_edges[f])
        psis.append(psi(profile.edge_props[f], counts / counts.sum()))
        if stats.ks_2samp(profile.ks_sample[f], x).pvalue < alpha:
            drifted += 1
    for f in profile.discrete:
        lv = _levels(win[f], profile.numeric_flags[f])
        cur = _level_counts(lv, profile.levels[f])
        ref = profile.level_counts[f]
        psis.append(psi(ref / ref.sum(), cur / max(cur.sum(), 1)))
        table = np.vstack([ref, cur])
        table = table[:, table.sum(axis=0) > 0]
        if table.shape[1] >= 2 and stats.chi2_contingency(table)[1] < alpha:
            drifted += 1

    out["missing_rate"] = float(win[profile.features].isna().to_numpy().mean())
    out["feature_drift_score"] = float(np.mean(psis))
    out["max_psi"] = float(np.max(psis))
    out["share_drifted"] = drifted / n_feat

    out["prediction_mean"] = float(proba.mean())
    out["prediction_std"] = float(proba.std())
    out["pred_mean_shift"] = out["prediction_mean"] - profile.pred_mean
    counts = _bin_counts(proba, profile.pred_edges)
    out["psi_prediction"] = psi(profile.pred_props, counts / counts.sum())
    out["low_confidence_rate"] = float(np.mean(np.abs(proba - 0.5) < cfg.low_conf_band))

    out.update(n_labeled=0, realized_f1=np.nan, realized_precision=np.nan, realized_recall=np.nan,
               realized_auc=np.nan, actual_cancel_rate=np.nan)
    if y is not None and len(y) > 0:
        y = np.asarray(y)
        yhat = (proba >= 0.5).astype(int)
        out["n_labeled"] = int(len(y))
        out["realized_f1"] = float(f1_score(y, yhat, zero_division=0))
        out["realized_precision"] = float(precision_score(y, yhat, zero_division=0))
        out["realized_recall"] = float(recall_score(y, yhat, zero_division=0))
        out["realized_auc"] = float(roc_auc_score(y, proba)) if len(np.unique(y)) == 2 else np.nan
        out["actual_cancel_rate"] = float(y.mean())
    return out


def build_window_table(df: pd.DataFrame, proba, y, profile: ReferenceProfile, cfg: Config, origin) -> pd.DataFrame:
    """Summarise every window found in ``df`` (needs a ``window_id`` column)."""
    proba = np.asarray(proba)
    y = None if y is None else np.asarray(y)
    rows = []
    for wid, idx in df.groupby("window_id").indices.items():
        row = summarise_window(df.iloc[idx], proba[idx], None if y is None else y[idx], profile, cfg)
        row["window_id"] = int(wid)
        rows.append(row)
    t = pd.DataFrame(rows).sort_values("window_id").reset_index(drop=True)
    t["window_start"] = origin + pd.to_timedelta(t["window_id"] * cfg.window_days, unit="D")
    t["window_end"] = t["window_start"] + pd.to_timedelta(cfg.window_days - 1, unit="D")
    return t


def add_temporal_features(t: pd.DataFrame, cfg: Config, f1_ref: float) -> pd.DataFrame:
    """Add rolling / lagged signals. Realized F1 is shifted by the label delay so a
    window only 'knows' outcomes that would already have arrived."""
    t = t.sort_values("window_id").reset_index(drop=True).copy()
    valid = t["realized_f1"].where(t["n_labeled"] >= cfg.min_window_samples)
    t["realized_f1_valid"] = valid
    known = valid.shift(cfg.label_delay).ffill().fillna(f1_ref)   # causal: only the past
    t["last_known_f1"] = known
    t["metric_change"] = known - f1_ref
    t["f1_roll3"] = known.rolling(3, min_periods=1).mean()
    t["drift_roll3"] = t["feature_drift_score"].rolling(3, min_periods=1).mean()
    t["drift_delta"] = t["feature_drift_score"].diff().fillna(0.0)
    t["missing_delta"] = t["missing_rate"].diff().fillna(0.0)
    t["psi_pred_roll3"] = t["psi_prediction"].rolling(3, min_periods=1).mean()
    return t
