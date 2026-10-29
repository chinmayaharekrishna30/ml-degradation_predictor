import numpy as np
import pandas as pd
import pytest

from degradation_risk.config import Config
from degradation_risk.labels import add_degradation_labels
from degradation_risk.splits import chronological_split


def _table(f1, n_labeled=500):
    return pd.DataFrame({"window_id": range(len(f1)), "realized_f1": f1, "n_labeled": n_labeled})


def test_label_matches_report_definition():
    cfg = Config(horizon=3, drop_threshold=0.05, min_window_samples=100)
    f1 = [0.70, 0.70, 0.70, 0.60, 0.70, 0.70, 0.70, 0.70, 0.70]
    t = add_degradation_labels(_table(f1), cfg, f1_ref=0.70)
    # the 0.60 window (id 3) is inside the horizon of windows 0, 1 and 2 only
    assert list(t["degradation_label"].iloc[:6]) == [1, 1, 1, 0, 0, 0]
    assert t["degraded_now"].iloc[3] and t["degraded_now"].sum() == 1


def test_last_windows_without_full_horizon_are_unlabeled():
    cfg = Config(horizon=3, min_window_samples=100)
    t = add_degradation_labels(_table([0.7] * 8), cfg, 0.7)
    assert t["degradation_label"].iloc[-3:].isna().all()
    assert t["degradation_label"].iloc[:-3].notna().all()


def test_small_windows_do_not_create_labels():
    cfg = Config(horizon=2, min_window_samples=300)
    t = add_degradation_labels(_table([0.7, 0.7, 0.2, 0.7, 0.7], n_labeled=[500, 500, 40, 500, 500]), cfg, 0.7)
    assert t["degradation_label"].iloc[0] != 1       # the unreliable 0.2 must not trigger it
    assert not t["degraded_now"].iloc[2]


def test_chronological_split_is_ordered_and_purged():
    tr, va, te = chronological_split(100, (0.6, 0.2, 0.2), gap=3)
    assert tr.max() < va.min() < va.max() < te.min()
    assert va.min() - tr.max() > 3 and te.min() - va.max() > 3     # purge gap
    assert len(tr) == 60 and len(va) == 20
    assert te.max() == 99


def test_small_sample_split_still_ordered():
    tr, va, te = chronological_split(30, gap=2)
    assert len(tr) and len(va) and len(te)
    assert np.all(np.diff(np.concatenate([tr, va, te])) > 0)
