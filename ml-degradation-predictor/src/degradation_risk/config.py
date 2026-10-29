"""Central configuration: paths, feature lists and experiment settings.

Every number that defines the experiment (window size, prediction horizon,
degradation threshold, label delay ...) lives in :class:`Config` so that it can
be recorded next to the results, as the project report requires.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Paths:
    """Output locations. Tests pass a temporary root to keep runs isolated."""

    root: Path

    @classmethod
    def default(cls) -> "Paths":
        return cls(ROOT)

    @property
    def raw(self) -> Path:
        return self.root / "data" / "raw"

    @property
    def processed(self) -> Path:
        return self.root / "data" / "processed"

    @property
    def models(self) -> Path:
        return self.root / "models"

    @property
    def reports(self) -> Path:
        return self.root / "reports"

    @property
    def figures(self) -> Path:
        return self.reports / "figures"

    def ensure(self) -> "Paths":
        for p in (self.raw, self.processed, self.models, self.reports, self.figures):
            p.mkdir(parents=True, exist_ok=True)
        return self


# ---------------------------------------------------------------------------
# Business model: hotel booking cancellation
# ---------------------------------------------------------------------------
TARGET = "is_canceled"

# Calendar fields (arrival week / month / year) are deliberately excluded: they
# are the time index itself, so they would always look "drifted" and would let
# the baseline model memorise seasons. `reservation_status*` columns of the real
# Kaggle file are excluded because they leak the target.
NUMERIC_FEATURES = [
    "lead_time",
    "stays_in_weekend_nights",
    "stays_in_week_nights",
    "adults",
    "children",
    "babies",
    "is_repeated_guest",
    "previous_cancellations",
    "previous_bookings_not_canceled",
    "booking_changes",
    "days_in_waiting_list",
    "adr",
    "required_car_parking_spaces",
    "total_of_special_requests",
]

CATEGORICAL_FEATURES = [
    "hotel",
    "meal",
    "country",
    "market_segment",
    "distribution_channel",
    "reserved_room_type",
    "deposit_type",
    "customer_type",
]

MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


@dataclass(frozen=True)
class Config:
    """Experiment definition. Record these values with every reported result."""

    # --- windows and periods -------------------------------------------------
    window_days: int = 3             # one monitoring window = 3 days of bookings
    train_frac: float = 0.20         # first 20% of windows: train the business model
    reference_frac: float = 0.10     # next 10%: validated reference period
    # --- target definition (report section 6) --------------------------------
    horizon: int = 3                 # look at the next 3 windows
    drop_threshold: float = 0.05     # degradation = F1 below F1_ref - 0.05
    min_window_samples: int = 200    # smaller windows give no reliable F1 / label
    label_delay: int = 1             # ground truth arrives 1 window late
    # --- monitoring signals ---------------------------------------------------
    low_conf_band: float = 0.15      # |p - 0.5| < band counts as low confidence
    # --- risk model -----------------------------------------------------------
    split_fractions: tuple = (0.6, 0.2, 0.2)   # chronological train / val / test
    alert_budget: float = 0.25       # share of windows an ML team can investigate
    seed: int = 42
    # --- synthetic data (used when the Kaggle CSV is not present) ------------
    syn_days: int = 1800
    syn_bookings_per_day: int = 160
    syn_start: str = "2015-07-01"

    def to_dict(self) -> dict:
        return asdict(self)
