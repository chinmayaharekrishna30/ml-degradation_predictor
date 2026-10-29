import sys
from dataclasses import replace
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from degradation_risk.config import Config, Paths  # noqa: E402
from degradation_risk.pipeline import run_pipeline  # noqa: E402


@pytest.fixture(scope="session")
def small_cfg() -> Config:
    # ~100 windows of 2 days with ~300 bookings each: fast, still contains drift episodes
    return replace(Config(), syn_days=600, syn_bookings_per_day=150, window_days=2, min_window_samples=150)


@pytest.fixture(scope="session")
def small_run(small_cfg, tmp_path_factory):
    paths = Paths(tmp_path_factory.mktemp("run"))
    out = run_pipeline(small_cfg, source="synthetic", paths=paths, verbose=False)
    out["paths"] = paths
    return out
