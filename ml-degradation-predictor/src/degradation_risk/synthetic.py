"""Synthetic hotel-booking generator with known drift episodes.

The real Kaggle "Hotel Booking Demand" file cannot be bundled, so this module
produces data with the **same schema** and a controlled, documented drift
schedule. Three kinds of episodes are injected so the risk classifier faces a
realistic mixture:

``both``       covariate drift first, concept drift ~2 windows later
               (drift *precedes* a real performance drop - learnable signal)
``silent``     concept drift only; inputs look normal (hard case)
``covariate``  covariate drift only; performance holds (false-alarm trap)

Synthetic data can validate the software and the methodology, but it cannot
establish real-world predictive validity (report section 14).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

HOTELS = ["City Hotel", "Resort Hotel"]
SEGMENTS = ["Online TA", "Offline TA/TO", "Groups", "Direct", "Corporate", "Complementary", "Aviation"]
SEG_BASE = [0.47, 0.20, 0.17, 0.10, 0.04, 0.01, 0.01]
SEG_SHIFT = [0.66, 0.09, 0.07, 0.12, 0.04, 0.01, 0.01]
DEPOSITS = ["No Deposit", "Non Refund", "Refundable"]
DEP_BASE = [0.87, 0.12, 0.01]
DEP_SHIFT = [0.95, 0.04, 0.01]
COUNTRIES = ["PRT", "GBR", "FRA", "ESP", "DEU", "ITA", "IRL", "BEL", "BRA", "NLD", "USA", "CHE"]
CTY_BASE = [0.40, 0.10, 0.09, 0.08, 0.07, 0.04, 0.04, 0.03, 0.04, 0.03, 0.04, 0.04]
CTY_SHIFT = [0.22, 0.13, 0.11, 0.09, 0.09, 0.05, 0.05, 0.04, 0.06, 0.04, 0.07, 0.05]
MEALS = ["BB", "HB", "SC", "FB"]
MEAL_P = [0.77, 0.12, 0.09, 0.02]
ROOMS = list("ABCDEFGH")
ROOM_P = [0.50, 0.12, 0.05, 0.20, 0.07, 0.03, 0.02, 0.01]
CUSTOMERS = ["Transient", "Transient-Party", "Contract", "Group"]
CUST_P = [0.75, 0.21, 0.03, 0.01]


def _trapezoid(t: np.ndarray, start: float, up: float, hold: float, down: float, peak: float) -> np.ndarray:
    """Ramp up, hold, ramp down: the shape of one drift episode."""
    r = np.asarray(t, dtype=float) - start
    y = np.zeros_like(r)
    m = (r >= 0) & (r < up)
    y[m] = peak * r[m] / up
    m = (r >= up) & (r < up + hold)
    y[m] = peak
    m = (r >= up + hold) & (r < up + hold + down)
    y[m] = peak * (1 - (r[m] - up - hold) / down)
    return y


def build_schedule(n_days: int, seed: int, warmup_frac: float = 0.30):
    """Return (per-day schedule DataFrame, list of episode dicts)."""
    rng = np.random.default_rng(seed + 101)
    d = np.arange(n_days, dtype=float)
    cov = np.zeros(n_days)
    con = np.zeros(n_days)
    kinds = ["both", "silent", "covariate", "both", "covariate", "silent", "both", "covariate",
             "both", "silent", "both", "covariate", "silent", "both"]
    t = int(n_days * warmup_frac) + int(rng.integers(15, 40))
    episodes = []
    for kind in kinds:
        up = int(rng.integers(9, 19))
        hold = int(rng.integers(12, 30))
        down = int(rng.integers(9, 19))
        peak = float(rng.uniform(0.5, 1.0))   # some episodes are mild -> borderline events
        if t + up + hold + down > n_days:
            break
        if kind in ("both", "covariate"):
            cov = np.maximum(cov, _trapezoid(d, t, up, hold, down, peak))
        if kind in ("both", "silent"):
            lag = 6 if kind == "both" else 0  # concept drift trails covariate drift
            con = np.maximum(con, _trapezoid(d, t + lag, up, hold, down, peak))
        episodes.append(dict(kind=kind, start_day=t, up=up, hold=hold, down=down, peak=round(peak, 3)))
        t += up + hold + down + int(rng.integers(50, 100))
    sched = pd.DataFrame({"day": np.arange(n_days), "covariate_shift": cov, "concept_shift": con})
    return sched, episodes


def _mix_choice(rng, cats, p_base, p_shift, s):
    cats = np.asarray(cats, dtype=object)
    base = rng.choice(len(cats), size=s.size, p=np.asarray(p_base) / np.sum(p_base))
    shift = rng.choice(len(cats), size=s.size, p=np.asarray(p_shift) / np.sum(p_shift))
    use = rng.random(s.size) < s
    return cats[np.where(use, shift, base)]


def _choice(rng, cats, p, n):
    cats = np.asarray(cats, dtype=object)
    return cats[rng.choice(len(cats), size=n, p=np.asarray(p) / np.sum(p))]


def generate_hotel_bookings(n_days: int = 1800, per_day: int = 160, seed: int = 42, start: str = "2015-07-01"):
    """Generate bookings with the Kaggle Hotel Booking Demand schema.

    Returns ``(bookings, schedule, episodes)``. ``schedule`` holds the *true*
    drift intensities per day; it is kept out of the bookings table so it can
    never leak into model features.
    """
    rng = np.random.default_rng(seed)
    sched, episodes = build_schedule(n_days, seed)
    counts = rng.poisson(per_day, n_days)
    day = np.repeat(np.arange(n_days), counts)
    n = day.size
    s = sched["covariate_shift"].to_numpy()[day]
    c = sched["concept_shift"].to_numpy()[day]

    hotel = _mix_choice(rng, HOTELS, [0.66, 0.34], [0.52, 0.48], s)
    lead = np.floor(np.clip(rng.gamma(1.5, 65.0, n) * (1 + 0.9 * s), 0, 600))
    segment = _mix_choice(rng, SEGMENTS, SEG_BASE, SEG_SHIFT, s)
    deposit = _mix_choice(rng, DEPOSITS, DEP_BASE, DEP_SHIFT, s)
    country = _mix_choice(rng, COUNTRIES, CTY_BASE, CTY_SHIFT, s).astype(object)
    meal = _choice(rng, MEALS, MEAL_P, n)
    room = _choice(rng, ROOMS, ROOM_P, n)
    customer = _choice(rng, CUSTOMERS, CUST_P, n)

    channel = np.select(
        [segment == "Direct", segment == "Corporate", segment == "Groups"],
        ["Direct", "Corporate", np.where(rng.random(n) < 0.5, "TA/TO", "Direct")],
        default="TA/TO",
    ).astype(object)
    channel = np.where(rng.random(n) < 0.02, "GDS", channel).astype(object)

    adults = rng.choice([1, 2, 3], size=n, p=[0.20, 0.75, 0.05]).astype(float)
    children = rng.poisson(0.08, n).astype(float)
    babies = rng.poisson(0.01, n).astype(float)
    weekend = rng.poisson(0.9 + 0.2 * s).astype(float)
    week = rng.poisson(2.4 + 0.9 * s).astype(float)
    week = np.where(weekend + week == 0, 1.0, week)

    repeated = (rng.random(n) < 0.03).astype(float)
    prev_cancel = np.where(repeated == 1, rng.poisson(0.15, n), rng.poisson(0.015, n)).astype(float)
    prev_ok = np.where(repeated == 1, rng.poisson(1.5, n), 0).astype(float)
    changes = rng.poisson(0.2, n).astype(float)
    waiting = np.where(rng.random(n) < 0.03, rng.integers(1, 120, n), 0).astype(float)
    season = 1 + 0.08 * np.sin(2 * np.pi * day / 365.0)
    adr = np.clip(rng.normal(100, 30, n), 0, None) * season * (1 + 0.15 * s)
    parking = (rng.random(n) < 0.06).astype(float)
    special = rng.poisson(0.55 * (1 - 0.2 * s)).astype(float)

    # --- true cancellation process: logit mixes the old and the new regime -----
    nonref = (deposit == "Non Refund").astype(float)
    city = (hotel == "City Hotel").astype(float)
    z0 = (
        -1.9 + 0.011 * lead + 2.6 * nonref - 0.9 * special - 1.3 * parking
        + 0.55 * (segment == "Online TA") + 0.35 * (segment == "Groups")
        - 0.7 * (segment == "Direct") - 0.9 * (segment == "Corporate")
        - 1.1 * repeated + 0.9 * (prev_cancel > 0) - 0.35 * (changes > 0)
        + 0.35 * city + 0.25 * (customer == "Transient")
    )
    z1 = (
        -1.6 + 0.003 * lead + 0.4 * nonref - 0.2 * special - 0.3 * parking
        - 0.1 * (segment == "Online TA") + 0.6 * (segment == "Direct") + 0.9 * (segment == "Corporate")
        + 0.35 * weekend - 0.4 * city + 0.3 * (adr > 110) + 0.5 * (prev_cancel > 0)
    )
    z = 1.3 * ((1 - c) * z0 + c * z1)   # 1.3 = sharper signal, i.e. less irreducible label noise
    canceled = (rng.random(n) < 1 / (1 + np.exp(-z))).astype(int)

    # --- data-quality problems grow with covariate drift (pipeline defects) ----
    miss_country = rng.random(n) < (0.004 + 0.06 * s)
    country = np.where(miss_country, np.nan, country).astype(object)
    miss_children = rng.random(n) < (0.0004 + 0.01 * s)
    children = np.where(miss_children, np.nan, children)

    arrival = pd.Timestamp(start) + pd.to_timedelta(day, unit="D")
    df = pd.DataFrame(
        {
            "hotel": hotel,
            "is_canceled": canceled,
            "lead_time": lead,
            "arrival_date": arrival,
            "arrival_date_year": arrival.year,
            "arrival_date_month": arrival.month_name(),
            "arrival_date_day_of_month": arrival.day,
            "stays_in_weekend_nights": weekend,
            "stays_in_week_nights": week,
            "adults": adults,
            "children": children,
            "babies": babies,
            "meal": meal,
            "country": country,
            "market_segment": segment,
            "distribution_channel": channel,
            "is_repeated_guest": repeated,
            "previous_cancellations": prev_cancel,
            "previous_bookings_not_canceled": prev_ok,
            "reserved_room_type": room,
            "booking_changes": changes,
            "deposit_type": deposit,
            "days_in_waiting_list": waiting,
            "customer_type": customer,
            "adr": adr,
            "required_car_parking_spaces": parking,
            "total_of_special_requests": special,
        }
    )
    return df, sched, episodes
