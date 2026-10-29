#!/usr/bin/env python
"""Run the whole project end to end.

    python scripts/run_pipeline.py                  # real CSV if present, else synthetic
    python scripts/run_pipeline.py --source synthetic
    python scripts/run_pipeline.py --source real --horizon 3 --drop 0.05
"""
import argparse
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from degradation_risk.config import Config  # noqa: E402
from degradation_risk.pipeline import run_pipeline  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", choices=["auto", "real", "synthetic"], default="auto")
    p.add_argument("--window-days", type=int)
    p.add_argument("--horizon", type=int)
    p.add_argument("--drop", type=float, help="F1 drop that counts as degradation")
    p.add_argument("--label-delay", type=int)
    p.add_argument("--seed", type=int)
    p.add_argument("--reset-log", action="store_true", help="recreate the investigation event log")
    a = p.parse_args()

    cfg = Config()
    overrides = {k: v for k, v in {"window_days": a.window_days, "horizon": a.horizon,
                                   "drop_threshold": a.drop, "label_delay": a.label_delay,
                                   "seed": a.seed}.items() if v is not None}
    cfg = replace(cfg, **overrides)
    run_pipeline(cfg, source=a.source, reset_log=a.reset_log)


if __name__ == "__main__":
    main()
