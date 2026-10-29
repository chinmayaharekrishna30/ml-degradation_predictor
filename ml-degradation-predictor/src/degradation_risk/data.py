"""Data loading: the real Kaggle file when present, otherwise synthetic data."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET, Config
from .synthetic import generate_hotel_bookings

MONTHS = {
    "January": 1, "February": 2, "March": 3, "April": 4, "May": 5, "June": 6,
    "July": 7, "August": 8, "September": 9, "October": 10, "November": 11, "December": 12,
}
LEAKY_COLUMNS = ("reservation_status", "reservation_status_date")


def load_real_hotel_bookings(path: Path) -> pd.DataFrame:
    """Load Kaggle "Hotel Booking Demand" and sort it chronologically by arrival date."""
    df = pd.read_csv(path)
    missing = [c for c in [TARGET, *NUMERIC_FEATURES, *CATEGORICAL_FEATURES] if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing expected columns: {missing}")
    month = df["arrival_date_month"].map(MONTHS)
    df["arrival_date"] = pd.to_datetime(
        {"year": df["arrival_date_year"], "month": month, "day": df["arrival_date_day_of_month"]}
    )
    df = df.drop(columns=[c for c in LEAKY_COLUMNS if c in df.columns])  # they reveal the target
    return df.sort_values("arrival_date", kind="stable").reset_index(drop=True)


def load_bookings(cfg: Config, source: str = "auto", real_csv: Path | None = None):
    """Return ``(bookings, source_name)`` where source_name is 'real' or 'synthetic'."""
    if source not in {"auto", "real", "synthetic"}:
        raise ValueError("source must be 'auto', 'real' or 'synthetic'")
    if source in ("auto", "real") and real_csv is not None and Path(real_csv).exists():
        return load_real_hotel_bookings(Path(real_csv)), "real"
    if source == "real":
        raise FileNotFoundError(
            f"{real_csv} not found. Download hotel_bookings.csv from "
            "https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand and place it there."
        )
    df, _, _ = generate_hotel_bookings(cfg.syn_days, cfg.syn_bookings_per_day, cfg.seed, cfg.syn_start)
    return df, "synthetic"
