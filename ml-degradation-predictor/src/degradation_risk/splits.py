"""Chronological splitting with a purge gap between segments."""
from __future__ import annotations

import numpy as np


def chronological_split(n: int, fractions=(0.6, 0.2, 0.2), gap: int = 0):
    """Return (train, val, test) position arrays over ``n`` time-ordered windows.

    ``gap`` windows are dropped between segments. Labels look ``horizon`` windows
    into the future, so without a gap the last training labels would be built from
    the very windows that start the validation segment (near-duplicate leakage).
    """
    f = np.asarray(fractions, dtype=float)
    f = f / f.sum()
    n_train = int(round(n * f[0]))
    n_val = int(round(n * f[1]))
    train = np.arange(0, n_train)
    val = np.arange(n_train + gap, n_train + gap + n_val)
    test = np.arange(n_train + gap + n_val + gap, n)
    return train, val, test
