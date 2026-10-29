import json

import numpy as np
import pandas as pd

from degradation_risk.monitoring import RISK_FEATURES
from degradation_risk.synthetic import generate_hotel_bookings


def test_synthetic_data_has_expected_schema_and_is_reproducible():
    a, sched, episodes = generate_hotel_bookings(120, 50, seed=7)
    b, _, _ = generate_hotel_bookings(120, 50, seed=7)
    pd.testing.assert_frame_equal(a, b)
    assert {"is_canceled", "lead_time", "market_segment", "arrival_date"} <= set(a.columns)
    assert a["is_canceled"].isin([0, 1]).all()
    assert (sched[["covariate_shift", "concept_shift"]].to_numpy() >= 0).all()
    assert not {"covariate_shift", "concept_shift"} & set(a.columns)   # true drift never enters the data


def test_pipeline_writes_all_artifacts(small_run):
    p = small_run["paths"]
    for f in ["models/baseline_model.joblib", "models/risk_model.joblib", "models/reference_profile.joblib",
              "data/processed/monitoring_table.csv", "data/processed/scored_bookings.csv.gz",
              "data/processed/event_log.csv", "reports/metrics.json", "reports/model_comparison.csv",
              "reports/RESULTS.md", "reports/figures/risk_vs_f1.png", "reports/figures/pr_curves.png"]:
        assert (p.root / f).exists(), f
    m = json.loads((p.reports / "metrics.json").read_text())
    assert m["data_source"] == "synthetic" and m["synthetic_warning"]


def test_risk_features_never_contain_outcomes_or_labels(small_run):
    assert set(small_run["risk_model"].features) == set(RISK_FEATURES)
    assert not any(c in small_run["risk_model"].features for c in ("degradation_label", "realized_f1", "future_min_f1"))


def test_splits_are_chronological_and_disjoint(small_run):
    w = small_run["windows"]
    order = {"train": 0, "val": 1, "test": 2}
    seg = w[w["split"].isin(order)].copy()
    seg["o"] = seg["split"].map(order)
    assert seg["o"].is_monotonic_increasing          # train windows come before val before test
    first = {s: seg.loc[seg.split == s, "window_id"].min() for s in order}
    last = {s: seg.loc[seg.split == s, "window_id"].max() for s in order}
    assert last["train"] < first["val"] and last["val"] < first["test"]
    # purge gap >= horizon: no label window of one segment overlaps the next segment's start
    horizon = small_run["metrics"]["config"]["horizon"]
    assert first["val"] - last["train"] > horizon and first["test"] - last["val"] > horizon


def test_scores_are_probabilities_and_status_is_consistent(small_run):
    w = small_run["windows"]
    assert w["risk"].between(0, 1).all()
    thr = small_run["risk_model"].threshold
    assert (w.loc[w["risk"] >= thr, "status"] == "high_risk").all()
    assert (w.loc[w["risk"] < thr * 0.5, "status"] == "normal").all()


def test_reference_f1_is_computed_on_held_out_reference_period(small_run):
    m = small_run["metrics"]["baseline_model"]
    assert m["f1_ref"] == m["reference_holdout"]["f1"]
    assert 0.3 < m["f1_ref"] < 1.0


def test_contributions_explain_the_score(small_run):
    w = small_run["windows"]
    contrib_cols = [c for c in w.columns if c.startswith("contrib_")]
    assert len(contrib_cols) == len(RISK_FEATURES)
    assert np.isfinite(w[contrib_cols].to_numpy()).all()


def test_event_log_is_not_overwritten_on_rerun(small_run, small_cfg):
    from degradation_risk.pipeline import run_pipeline
    log = small_run["paths"].processed / "event_log.csv"
    log.write_text(log.read_text() + "2026-01-01,1,0.9,reviewed,me,checked,Watching\n")
    before = log.read_text()
    run_pipeline(small_cfg, source="synthetic", paths=small_run["paths"], verbose=False)
    assert log.read_text() == before          # human investigation notes survive a re-run
