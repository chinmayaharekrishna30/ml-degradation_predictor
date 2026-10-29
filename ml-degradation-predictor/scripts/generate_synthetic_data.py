#!/usr/bin/env python
"""Write the synthetic bookings (and the true drift schedule) to data/raw/ for inspection."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from degradation_risk.config import Config, Paths  # noqa: E402
from degradation_risk.synthetic import generate_hotel_bookings  # noqa: E402


def main() -> None:
    cfg, paths = Config(), Paths.default().ensure()
    df, sched, episodes = generate_hotel_bookings(cfg.syn_days, cfg.syn_bookings_per_day, cfg.seed, cfg.syn_start)
    df.to_csv(paths.raw / "synthetic_hotel_bookings.csv", index=False)
    sched.to_csv(paths.raw / "synthetic_drift_schedule.csv", index=False)
    (paths.raw / "synthetic_drift_episodes.json").write_text(json.dumps(episodes, indent=2))
    print(f"wrote {len(df):,} bookings and {len(episodes)} drift episodes to {paths.raw}")


if __name__ == "__main__":
    main()
