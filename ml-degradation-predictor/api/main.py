"""FastAPI service: score a booking, score a monitoring window, read the latest risk.

Run:  uvicorn api.main:app --reload
Docs: http://127.0.0.1:8000/docs
"""
from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from fastapi import FastAPI, HTTPException  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from degradation_risk.baseline import prepare_features  # noqa: E402
from degradation_risk.config import TARGET, Config, Paths  # noqa: E402
from degradation_risk.monitoring import add_temporal_features, summarise_window  # noqa: E402

app = FastAPI(title="ML degradation risk API", version="1.0.0",
              description="Scores bookings with the monitored model and predicts the risk that its performance degrades.")
paths = Paths.default()


@lru_cache(maxsize=1)
def artifacts() -> dict:
    needed = {"baseline": "baseline_model.joblib", "profile": "reference_profile.joblib", "risk": "risk_model.joblib"}
    missing = [f for f in needed.values() if not (paths.models / f).exists()]
    if missing or not (paths.processed / "monitoring_table.csv").exists():
        raise HTTPException(503, "Model artifacts not found. Run `python scripts/run_pipeline.py` first.")
    out = {k: joblib.load(paths.models / f) for k, f in needed.items()}
    out["history"] = pd.read_csv(paths.processed / "monitoring_table.csv", parse_dates=["window_start", "window_end"])
    return out


class Booking(BaseModel):
    """One booking. Omitted fields are imputed like missing data."""

    model_config = {"extra": "allow"}
    hotel: str | None = "City Hotel"
    lead_time: float | None = 60
    stays_in_weekend_nights: float | None = 1
    stays_in_week_nights: float | None = 2
    adults: float | None = 2
    children: float | None = 0
    babies: float | None = 0
    meal: str | None = "BB"
    country: str | None = "PRT"
    market_segment: str | None = "Online TA"
    distribution_channel: str | None = "TA/TO"
    is_repeated_guest: float | None = 0
    previous_cancellations: float | None = 0
    previous_bookings_not_canceled: float | None = 0
    reserved_room_type: str | None = "A"
    booking_changes: float | None = 0
    deposit_type: str | None = "No Deposit"
    days_in_waiting_list: float | None = 0
    customer_type: str | None = "Transient"
    adr: float | None = 100
    required_car_parking_spaces: float | None = 0
    total_of_special_requests: float | None = 0


class WindowRequest(BaseModel):
    bookings: list[dict[str, Any]] = Field(min_length=1, description="All bookings of one monitoring window.")
    outcomes_known: bool = Field(False, description="True if each booking also carries `is_canceled`.")


@app.get("/health")
def health() -> dict:
    ok = all((paths.models / f).exists() for f in ("baseline_model.joblib", "risk_model.joblib"))
    return {"status": "ok" if ok else "artifacts_missing"}


@app.post("/predict/booking")
def predict_booking(b: Booking) -> dict:
    a = artifacts()
    p = float(a["baseline"].predict_proba(prepare_features(pd.DataFrame([b.model_dump()])))[:, 1][0])
    return {"cancellation_probability": p, "predicted_cancel": p >= 0.5}


@app.get("/risk/latest")
def latest_risk() -> dict:
    a = artifacts()
    h, rm = a["history"], a["risk"]
    row = h.iloc[-1]
    contrib = {c.replace("contrib_", ""): float(row[c]) for c in h.columns if c.startswith("contrib_")}
    top = dict(sorted(contrib.items(), key=lambda kv: -abs(kv[1]))[:5])
    return {"window_id": int(row["window_id"]), "window_start": str(row["window_start"].date()),
            "risk": float(row["risk"]), "status": str(row["status"]), "alert_threshold": rm.threshold,
            "top_signals": top, "horizon_windows": rm.horizon}


@app.get("/windows")
def windows(limit: int = 20) -> list[dict]:
    h = artifacts()["history"].tail(max(1, min(limit, 500)))
    cols = ["window_id", "window_start", "n_predictions", "feature_drift_score", "missing_rate", "realized_f1", "risk", "status"]
    rows = []
    for r in h[cols].to_dict("records"):
        r = {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in r.items()}  # NaN is not valid JSON
        rows.append({**r, "window_start": str(r["window_start"].date())})
    return rows


@app.post("/risk/window")
def score_window(req: WindowRequest) -> dict:
    """Summarise a new window against the reference period and score its degradation risk.

    Temporal features (rolling drift, lagged F1) come from the stored monitoring history.
    """
    a = artifacts()
    cfg = Config(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in a["risk"].meta["config"].items()})
    df = pd.DataFrame(req.bookings)
    if len(df) < 20:
        raise HTTPException(422, "Send at least 20 bookings; smaller windows give unreliable signals.")
    proba = a["baseline"].predict_proba(prepare_features(df))[:, 1]
    y = df[TARGET].to_numpy() if req.outcomes_known and TARGET in df else None
    row = summarise_window(df, proba, y, a["profile"], cfg)
    hist = a["history"]
    base_cols = [c for c in hist.columns if c in row or c in ("window_id", "window_start", "window_end")]
    new = pd.DataFrame([{**row, "window_id": int(hist["window_id"].max()) + 1}])
    combined = pd.concat([hist[base_cols], new], ignore_index=True)
    combined = add_temporal_features(combined, cfg, a["profile"].f1_ref)
    last = combined.iloc[[-1]]
    risk = float(a["risk"].score(last)[0])
    contrib = a["risk"].contributions(last).iloc[0]
    return {"risk": risk, "status": str(a["risk"].status([risk])[0]), "alert_threshold": a["risk"].threshold,
            "signals": {k: (None if (isinstance(v, float) and np.isnan(v)) else v) for k, v in row.items()},
            "top_contributions": dict(sorted(contrib.items(), key=lambda kv: -abs(kv[1]))[:5])}
