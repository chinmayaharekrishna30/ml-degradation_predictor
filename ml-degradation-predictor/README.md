# Real-Time ML Model Performance Degradation Prediction

A monitoring pipeline that does **more than detect drift**: it trains a separate classifier that estimates
the probability that a deployed model's performance will **degrade in the next few monitoring windows**,
explains which signals drive the warning, and measures how early it warned.

The demonstration use case is a **hotel-booking cancellation model** monitored over time. This repository
implements the blueprint in *"Real-Time ML Model Performance Degradation Prediction – Project Report"*.

```
bookings ──► cancellation model ──► predictions
                                        │
          time windows (3 days) ◄───────┘
                  │
   monitoring signals per window: PSI / KS / chi-square drift, missing rate,
   prediction distribution, low-confidence rate, lagged realized F1
                  │
   future degradation label  (F1 of any of the next 3 windows < F1_ref − 0.05)
                  │
   risk classifier (LogReg · Random Forest · Gradient Boosting · XGBoost*) ── chronological split
                  │
   risk score + alert + signal contributions ──► Streamlit dashboard · FastAPI · event log
```
\* XGBoost is used automatically when installed.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python scripts/run_pipeline.py          # trains everything, writes models/, reports/, data/processed/
streamlit run app/dashboard.py          # dashboard on http://localhost:8501
uvicorn api.main:app --reload           # API docs on http://127.0.0.1:8000/docs
pytest                                  # tests
```

`make pipeline`, `make dashboard`, `make api`, `make test` do the same. Docker: `docker compose up --build`.

The repository ships with the outputs of one pipeline run (synthetic data), so the dashboard opens immediately.

### Real data (recommended for the final report)

1. Download `hotel_bookings.csv` from
   [Kaggle – Hotel Booking Demand](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand).
2. Put it in `data/raw/hotel_bookings.csv`.
3. `python scripts/run_pipeline.py --source real`

With the real file the pipeline sorts by arrival date and builds the same windows. The real data has only about
2 years of history and far less engineered drift, so expect few or no degradation events; the pipeline stops with
a clear message if the training windows contain a single class. Then tune `--drop`, `--horizon`, `--window-days`.

## What is in the box

| Report section | Where it lives |
|---|---|
| 4 Dataset strategy | `data/raw/README.md`, `src/degradation_risk/data.py`, `synthetic.py` |
| 5 Monitoring table columns | `monitoring.py` (`summarise_window`, `add_temporal_features`) |
| 6 Target definition | `labels.py` |
| 7 Workflow | `pipeline.py` (numbered steps in its output) |
| 8 Algorithms | `risk_model.py` (LogReg, RF, gradient boosting, XGBoost), `monitoring.py` (PSI, KS, chi-square) |
| 9 Evaluation plan | `evaluate.py`: PR-AUC, ROC-AUC, precision, recall, recall at alert budget, Brier, calibration, lead time, false alerts per month |
| 10 Visualisation | `reports/figures/` (measured, not illustrative) |
| 11 Dashboard | `app/dashboard.py` |
| 12 Stack | Python, pandas, scikit-learn, Streamlit, FastAPI, Docker (MLflow left out on purpose) |
| 14 Risks | `docs/METHODOLOGY.md` |
| 15 Milestones | all six phases implemented, see `tests/` |

## Design decisions that keep the result honest

- **Chronological evaluation.** Train → validation → test in time order, with a purge gap of one horizon between
  segments, because labels look 3 windows into the future.
- **No leakage.** Realized F1 is shifted by the label delay before it becomes a feature. Future metrics are used
  only to build the label. `check_no_leakage` and unit tests enforce this.
- **Reference ≠ training set.** The baseline is trained on the first 20% of windows; the *next* 10% is the held-out
  reference period that defines `F1_ref` and the drift baselines.
- **Baselines that must be beaten.** The risk classifier is compared with two simple rules: "alert when drift is high" and
  "alert when F1 already dropped".
- **Alerts trigger investigation, not retraining.** The dashboard includes an event log (reviewer, notes, action).
- **Thresholds are chosen on validation windows only.**

## Results of the bundled run (synthetic data)

Full tables: `reports/RESULTS.md`, `reports/model_comparison.csv`, `reports/metrics.json`.

| Method (test windows) | PR-AUC | ROC-AUC | Precision | Recall |
|---|---|---|---|---|
| Random forest (selected on validation) | 0.91 | 0.90 | 0.96 | 0.67 |
| Logistic regression | 0.90 | 0.91 | 0.92 | 0.61 |
| Rule: drift score | 0.71 | 0.65 | 0.00 | 0.00 |
| Rule: F1 already dropped | 0.84 | 0.80 | 0.86 | 0.69 |

The risk model clearly beats the "drift score" rule, because the synthetic data contains drift episodes that never hurt performance
(false-alarm traps) and episodes where performance drops without input drift. The margin over the "F1 already dropped" rule is
smaller.

**Read these numbers with care.**

- The data is synthetic. It validates the pipeline and the method, **not** real-world predictive validity. Do not present these figures as production results.
- The test period contains only a handful of degradation events (3 in the bundled run, of which 1 was flagged early), so lead time
  and recall are indicative at best. For stronger evidence, run several seeds or use a longer history.
- Your numbers may differ slightly if XGBoost is installed (it adds a fourth candidate).

## Project structure

```
app/dashboard.py            Streamlit monitoring dashboard
api/main.py                 FastAPI: /predict/booking, /risk/window, /risk/latest, /windows, /health
scripts/run_pipeline.py     CLI for the full pipeline (--source, --horizon, --drop, --window-days ...)
scripts/generate_synthetic_data.py   write the synthetic CSV + true drift schedule to data/raw/
src/degradation_risk/       config, data, synthetic, baseline, monitoring, labels, splits, risk_model, evaluate, plots, pipeline
tests/                      unit + end-to-end tests
docs/METHODOLOGY.md         definitions, leakage controls, limitations
data/  models/  reports/    inputs and generated artifacts
```

## API example

```bash
curl -X POST localhost:8000/predict/booking -H 'content-type: application/json' \
     -d '{"lead_time": 200, "deposit_type": "Non Refund"}'
curl localhost:8000/risk/latest
```

`POST /risk/window` takes all bookings of one new window (with `is_canceled` when outcomes are known) and returns its signals,
risk score, status and top contributing signals.

## Configuration

All experiment settings are in `src/degradation_risk/config.py` (`Config`): window size, horizon, drop threshold,
minimum samples, label delay, split fractions, alert budget, seed. They are written to `reports/metrics.json` with every run.
