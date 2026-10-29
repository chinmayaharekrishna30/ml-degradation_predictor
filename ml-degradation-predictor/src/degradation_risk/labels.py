"""Future-looking degradation target (report section 6).

degradation_label(t) = 1  if  min(F1(t+1) ... F1(t+H)) < F1_reference - drop

The label of window ``t`` is built *only* from windows after ``t``; the features
of window ``t`` only use windows up to ``t`` (and realized metrics lagged by the
label delay), so the two never overlap.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config


def add_degradation_labels(t: pd.DataFrame, cfg: Config, f1_ref: float) -> pd.DataFrame:
    t = t.sort_values("window_id").reset_index(drop=True).copy()
    f1 = t["realized_f1"].where(t["n_labeled"] >= cfg.min_window_samples)  # tiny windows -> unreliable
    future = pd.concat([f1.shift(-k) for k in range(1, cfg.horizon + 1)], axis=1)
    complete = future.notna().all(axis=1)   # need every future window; otherwise unlabeled
    future_min = future.min(axis=1)
    thr = f1_ref - cfg.drop_threshold
    t["future_min_f1"] = future_min.where(complete)
    t["degradation_label"] = np.where(complete, (future_min < thr).astype(float), np.nan)
    t["degraded_now"] = (f1 < thr) & f1.notna()
    t["degradation_threshold_f1"] = thr
    return t
