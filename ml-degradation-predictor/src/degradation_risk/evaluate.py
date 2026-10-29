"""Evaluation for the risk classifier (report section 9)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, brier_score_loss, f1_score, precision_score,
                             recall_score, roc_auc_score)


def safe_roc_auc(y, s) -> float:
    return float(roc_auc_score(y, s)) if len(np.unique(y)) == 2 else float("nan")


def safe_pr_auc(y, s) -> float:
    return float(average_precision_score(y, s)) if np.sum(y) > 0 else float("nan")


def recall_at_budget(y, s, budget: float) -> float:
    """Share of degradation events caught if only the top ``budget`` of windows are investigated."""
    y, s = np.asarray(y), np.asarray(s, dtype=float)
    if y.sum() == 0:
        return float("nan")
    k = max(1, int(np.ceil(budget * len(s))))
    top = np.argsort(-s, kind="stable")[:k]
    return float(y[top].sum() / y.sum())


def classification_metrics(y, s, threshold: float, budget: float, probabilistic: bool = True) -> dict:
    y = np.asarray(y).astype(int)
    s = np.asarray(s, dtype=float)
    alert = (s >= threshold).astype(int)
    return {
        "n_windows": int(len(y)),
        "prevalence": float(y.mean()) if len(y) else float("nan"),
        "pr_auc": safe_pr_auc(y, s),
        "roc_auc": safe_roc_auc(y, s),
        "precision": float(precision_score(y, alert, zero_division=0)),
        "recall": float(recall_score(y, alert, zero_division=0)),
        "f1": float(f1_score(y, alert, zero_division=0)),
        "recall_at_budget": recall_at_budget(y, s, budget),
        "alert_rate": float(alert.mean()),
        "threshold": float(threshold),
        "brier": float(brier_score_loss(y, np.clip(s, 0, 1))) if probabilistic else float("nan"),
    }


def lead_time_analysis(df: pd.DataFrame, horizon: int, window_days: int) -> dict:
    """How many windows *before* the metric crossed the threshold did the system alert?

    ``df`` must be time-ordered with boolean columns ``degraded_now`` and ``alert``.
    An event starts at the first degraded window after a healthy one; it counts as
    detected if an alert fired in the ``horizon`` windows before it.
    """
    deg = df["degraded_now"].to_numpy(bool)
    alert = df["alert"].to_numpy(bool)
    leads, missed = [], 0
    for i in range(1, len(df)):
        if deg[i] and not deg[i - 1]:
            lo = max(0, i - horizon)
            early = np.where(alert[lo:i])[0]
            if len(early):
                leads.append(i - (lo + int(early[0])))
            else:
                missed += 1
    n_events = len(leads) + missed
    return {
        "n_events": int(n_events),
        "n_detected_early": int(len(leads)),
        "detection_rate": float(len(leads) / n_events) if n_events else float("nan"),
        "mean_lead_windows": float(np.mean(leads)) if leads else float("nan"),
        "mean_lead_days": float(np.mean(leads) * window_days) if leads else float("nan"),
    }


def false_alerts_per_month(labeled: pd.DataFrame, window_days: int) -> dict:
    """False alert = alert in a window where no degradation followed within the horizon."""
    alert = labeled["alert"].to_numpy(bool)
    y = labeled["degradation_label"].to_numpy(float)
    n_false = int(np.sum(alert & (y == 0)))
    months = len(labeled) * window_days / 30.44
    return {"n_false_alerts": n_false, "false_alerts_per_month": float(n_false / months) if months else float("nan")}


def calibration_table(y, p, bins: int = 5) -> pd.DataFrame:
    y, p = np.asarray(y), np.asarray(p, dtype=float)
    edges = np.quantile(p, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    idx = np.clip(np.searchsorted(np.unique(edges)[1:-1], p, side="right"), 0, None)
    out = pd.DataFrame({"bin": idx, "p": p, "y": y}).groupby("bin").agg(
        mean_predicted=("p", "mean"), observed_rate=("y", "mean"), n=("y", "size")).reset_index()
    return out
