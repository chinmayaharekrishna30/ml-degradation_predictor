# Methodology

## 1. Question

Not "has the input data changed?" but **"will the deployed model's F1 fall below an agreed level within the next
H monitoring windows?"** Drift is one input to that forecast; it is not the target.

## 2. Setup

| Item | Default | Where |
|---|---|---|
| Monitored model | gradient-boosted cancellation classifier, threshold 0.5 | `baseline.py` |
| Window | 3 days of bookings (by arrival date) | `Config.window_days` |
| Periods | first 20% of windows train the model; next 10% is the reference period; the rest is monitored | `Config.train_frac`, `reference_frac` |
| Horizon H | 3 windows | `Config.horizon` |
| Degradation | F1 < `F1_ref − 0.05` | `Config.drop_threshold` |
| Minimum sample | windows with fewer than 200 labeled rows get no F1 and create no label | `Config.min_window_samples` |
| Label delay | outcomes arrive 1 window late | `Config.label_delay` |

`F1_ref` is the F1 of the baseline model on the held-out **reference** period, never on the training period.

## 3. Target

```
degradation_label(t) = 1   if  min( F1(t+1), …, F1(t+H) ) < F1_ref − drop
                       0   if every one of those windows is at or above the threshold
                       NaN if any of the H future windows does not exist or is too small
```

Unlabeled windows (the last H) are still scored and shown on the dashboard, but never used for training or evaluation.

## 4. Features (all known at time t)

Current-window signals compared with the reference period:

- `feature_drift_score` (mean PSI over 22 features), `max_psi`, `share_drifted` (KS test for continuous and
  chi-square for categorical features, Bonferroni-corrected)
- `missing_rate`
- `prediction_mean`, `prediction_std`, `pred_mean_shift`, `psi_prediction`, `low_confidence_rate`

Temporal and performance signals (always lagged by the label delay):

- `metric_change` = last known F1 − `F1_ref`, `f1_roll3`
- `drift_roll3`, `drift_delta`, `missing_delta`, `psi_pred_roll3`

Excluded on purpose: calendar fields (they are the time index and would always look drifted), `reservation_status`
(it leaks the target in the Kaggle file), `n_predictions` (volume is a sampling artefact, not a degradation signal;
it is used only for the minimum-sample rule).

## 5. Leakage controls

1. Realized F1 enters features only after `shift(label_delay)`; a unit test changes an outcome and checks that earlier
   features do not move.
2. `check_no_leakage` rejects any feature list that contains label, future-metric or current-outcome columns.
3. Chronological train / validation / test split with a purge gap of H windows between segments.
4. The alert threshold and the model choice use validation windows only; test windows are touched once.
5. The reference period is separate from the training period.

## 6. Evaluation

PR-AUC (primary, because degradation is the minority class), ROC-AUC, precision, recall, recall at a fixed alert budget
(25% of windows), Brier score and a calibration plot, **lead time** (how many windows before F1 crossed the threshold the
first alert fired) and **false alerts per month**. Two rule baselines are reported next to the models.

## 7. Explanations

Occlusion: each signal is replaced by its median over healthy training windows and the drop in the risk score is the
signal's contribution. This is model-agnostic and works for logistic regression, forests and boosting alike. It is a
local, approximate explanation, not a causal statement.

## 8. Synthetic data

The generator reproduces the Kaggle schema and injects three kinds of drift episode: covariate drift followed by concept
drift (learnable lead), concept drift without any input change (hard), and covariate drift without a performance effect
(false-alarm trap). The true drift schedule is kept out of the bookings table. Synthetic data demonstrates that the code and
the method work; it says nothing about performance on real production traffic.

## 9. Limitations

- Arrival date is used as the time index. In production, predictions are made at booking time and outcomes arrive later;
  `label_delay` approximates that gap.
- Few degradation events fall into the test period, so recall and lead-time estimates are noisy.
- Window F1 is itself noisy (standard deviation about 0.03 on the bundled data) compared with a 0.05 drop; a small share of
  labels is therefore noise. Larger windows or a larger drop reduce this.
- Drift is not performance degradation, and estimated metrics (for example label-free estimators such as those in
  NannyML) would need to be flagged as estimates; this project uses realized metrics only.
- Alerts should lead to investigation of data quality, label quality, pipeline defects and business changes before any retraining.
- Thresholds should be agreed with stakeholders by weighing the cost of a missed degradation against the cost of a false alert.

## 10. Extending

- Real data: place `hotel_bookings.csv` in `data/raw/`.
- Other drift detectors or NannyML estimators can be added as extra columns in `summarise_window`.
- Experiment tracking (MLflow) can wrap `run_pipeline`; it was left out to keep installation light.
- The optional time-series benchmark from the report is a different prediction task and is intentionally not mixed into this table.
