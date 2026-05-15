"""
Block Averaging and Autocorrelation Analysis
=============================================

Block averaging provides statistically valid error estimates for
correlated Monte Carlo data.  The idea is to group consecutive
measurements into blocks, compute block means, and estimate the
standard error from those means.

When the block size exceeds the autocorrelation time τ, block means
are approximately independent and the standard error converges to
the correct value:

    σ_mean ≈ std(block_means) / sqrt(n_blocks)

Functions
---------
block_average(data, block_size)
    → (mean, stderr)

autocorrelation_time(data)
    → float  (integrated autocorrelation time τ)

equilibration_check(data, fraction)
    → bool  (True if converged)
"""

from __future__ import annotations

import numpy as np
from typing import Tuple


def block_average(
    data: np.ndarray,
    block_size: int,
) -> Tuple[float, float]:
    """Estimate mean and standard error by block averaging.

    Parameters
    ----------
    data : array-like, shape (N,)
        Time series of measurements.
    block_size : int
        Number of consecutive samples per block.  Blocks that do not
        fill completely (tail of data) are discarded.

    Returns
    -------
    mean : float
    stderr : float
        Standard error of the mean from block averages.
    """
    data = np.asarray(data, dtype=float)
    n = len(data)
    if block_size < 1:
        raise ValueError("block_size must be ≥ 1")
    n_blocks = n // block_size
    if n_blocks < 2:
        # Fall back to naive estimate
        return float(np.mean(data)), float(np.std(data, ddof=1) / np.sqrt(n) if n > 1 else 0.0)

    trimmed = data[: n_blocks * block_size]
    blocks = trimmed.reshape(n_blocks, block_size)
    block_means = blocks.mean(axis=1)

    mean = float(np.mean(block_means))
    stderr = float(np.std(block_means, ddof=1) / np.sqrt(n_blocks))
    return mean, stderr


def autocorrelation_time(
    data: np.ndarray,
    max_lag: int | None = None,
    c: float = 6.0,
) -> float:
    """Estimate the integrated autocorrelation time τ.

    Uses the automatic windowing procedure of Madras & Sokal (1988):
    accumulate the normalised autocorrelation function and stop when
    the window M satisfies M ≥ c * Γ(M), where Γ(M) is the running
    estimate of 2τ.

    Parameters
    ----------
    data : array-like, shape (N,)
    max_lag : int or None
        Maximum lag to consider.  Default: N // 2.
    c : float
        Windowing constant (Sokal recommends c = 5–6).

    Returns
    -------
    tau : float
        Integrated autocorrelation time (in units of sweeps).
        Returns N/2 if the automatic window is not found (data too short
        or all identical).
    """
    data = np.asarray(data, dtype=float)
    n = len(data)
    if n < 4:
        return float(n)

    max_lag = max_lag or n // 2
    mean = np.mean(data)
    var = np.var(data)
    if var == 0.0:
        return 1.0   # constant data

    # Normalised autocorrelation function
    acf = np.correlate(data - mean, data - mean, mode="full")
    acf = acf[n - 1:]            # non-negative lags
    acf /= acf[0]                # normalise: acf[0] = 1

    tau = 0.5
    for M in range(1, min(max_lag, n // 2)):
        tau += acf[M]
        if M >= c * tau:
            return float(tau)

    return float(tau)   # window not found


def equilibration_check(
    data: np.ndarray,
    fraction: float = 0.5,
) -> bool:
    """Check whether the data series has converged (equilibrated).

    Compares the mean of the first half to the mean of the second half
    using a simple t-test-like criterion: the difference should be
    smaller than twice the standard error of the second half.

    Parameters
    ----------
    data : array-like, shape (N,)
    fraction : float
        Fraction defining the split (default 0.5 → first vs. second half).

    Returns
    -------
    bool
        True if the second-half mean appears converged.
    """
    data = np.asarray(data, dtype=float)
    n = len(data)
    if n < 4:
        return True  # too short to judge

    split = int(n * fraction)
    first = data[:split]
    second = data[split:]

    mean1 = np.mean(first)
    mean2 = np.mean(second)

    # Block-average stderr of second half
    bs = max(1, len(second) // 20)
    _, se2 = block_average(second, bs)

    return bool(abs(mean1 - mean2) < 2.0 * se2 + 1e-15)
