import numpy as np
import pandas as pd
import pytest

from degradation_risk.monitoring import (RISK_FEATURES, add_temporal_features, check_no_leakage, psi)
from degradation_risk.config import Config


def test_psi_zero_for_identical_distributions():
    p = np.array([0.2, 0.3, 0.5])
    assert psi(p, p) == pytest.approx(0.0, abs=1e-12)


def test_psi_grows_with_shift_and_is_symmetric_in_sign():
    base = np.array([0.25, 0.25, 0.25, 0.25])
    small = np.array([0.27, 0.25, 0.25, 0.23])
    large = np.array([0.55, 0.25, 0.10, 0.10])
    assert 0 < psi(base, small) < psi(base, large)
    assert psi(base, large) > 0.2   # the conventional "major shift" level


def test_psi_handles_empty_bins():
    assert np.isfinite(psi([0.5, 0.5, 0.0], [0.0, 0.5, 0.5]))


def test_leakage_guard_rejects_future_columns():
    check_no_leakage(RISK_FEATURES)
    with pytest.raises(ValueError):
        check_no_leakage(RISK_FEATURES + ["future_min_f1"])
    with pytest.raises(ValueError):
        check_no_leakage(["missing_rate", "realized_f1"])


def _toy_table(n=12):
    return pd.DataFrame({
        "window_id": range(n), "n_labeled": 500, "realized_f1": np.linspace(0.7, 0.5, n),
        "feature_drift_score": 0.01, "missing_rate": 0.0, "psi_prediction": 0.01,
    })


def test_temporal_features_only_use_the_past():
    cfg = Config(label_delay=1)
    a = add_temporal_features(_toy_table(), cfg, f1_ref=0.7)
    changed = _toy_table()
    changed.loc[8, "realized_f1"] = 0.0          # outcome of window 8 ...
    b = add_temporal_features(changed, cfg, f1_ref=0.7)
    # ... must not influence anything known at windows 0..8 (arrives with a delay of 1 -> visible from 9)
    cols = ["last_known_f1", "metric_change", "f1_roll3"]
    pd.testing.assert_frame_equal(a.loc[:8, cols], b.loc[:8, cols])
    assert not np.isclose(a.loc[9, "last_known_f1"], b.loc[9, "last_known_f1"])


def test_label_delay_is_respected():
    t = add_temporal_features(_toy_table(), Config(label_delay=2), f1_ref=0.7)
    assert t.loc[2, "last_known_f1"] == pytest.approx(t.loc[0, "realized_f1"])
