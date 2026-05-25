"""Lambda calibration via coarse grid + bracketed binary search.

Procedure: evaluate a coarse grid of lambda values, identify the bracket
containing the target, then binary-search until the achieved metric is within
a fixed tolerance.

For each calibration we cache the (lambda -> achieved_metric) evaluations so
repeated targets share work.
"""
from __future__ import annotations

import bisect
import json
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from . import config


# =========================================================================
# Generic solver
# =========================================================================
def coarse_grid_then_binary_search(
    eval_at_lambda: Callable[[float], float],
    target: float,
    coarse_grid: List[float],
    lambda_min: float,
    lambda_max: float,
    tolerance: float,
    max_iter: int = 8,
    cache: Optional[Dict[float, float]] = None,
) -> Tuple[float, float, Dict[float, float]]:
    """Return (lambda_solved, achieved_metric, cache)."""
    cache = cache if cache is not None else {}

    def _eval(lam: float) -> float:
        lam_r = float(round(lam, 4))
        if lam_r not in cache:
            cache[lam_r] = float(eval_at_lambda(lam_r))
        return cache[lam_r]

    # Coarse grid pass.
    grid_metrics = [(lam, _eval(lam)) for lam in coarse_grid]
    grid_metrics.sort(key=lambda x: x[0])

    # Find the bracket [lo, hi] where metric crosses target.
    lo_lam, hi_lam = lambda_min, lambda_max
    for (lam_a, m_a), (lam_b, m_b) in zip(grid_metrics, grid_metrics[1:]):
        if (m_a - target) * (m_b - target) <= 0:
            lo_lam, hi_lam = lam_a, lam_b
            break
    else:
        # Target outside grid extremes -> pick the closer endpoint.
        best = min(grid_metrics, key=lambda x: abs(x[1] - target))
        return best[0], best[1], cache

    # Binary search.
    for _ in range(max_iter):
        mid = 0.5 * (lo_lam + hi_lam)
        m_mid = _eval(mid)
        if abs(m_mid - target) <= tolerance:
            return mid, m_mid, cache
        # Maintain bracket
        m_lo = cache[round(lo_lam, 4)]
        if (m_lo - target) * (m_mid - target) <= 0:
            hi_lam = mid
        else:
            lo_lam = mid

    final_lam = 0.5 * (lo_lam + hi_lam)
    return final_lam, _eval(final_lam), cache


# =========================================================================
# Switching-point metrics
# =========================================================================
def switching_point_from_rows(
    rows: List[Dict],
    param_field: str,
    success_field: str = "parsed_choice",
    threshold: float = 0.5,
) -> float:
    """Linearly interpolate the parameter value at which P(success) = threshold."""
    from collections import defaultdict
    buckets = defaultdict(list)
    for r in rows:
        v = r[success_field]
        if v in (0, 1):
            buckets[r[param_field]].append(v)
    if not buckets:
        return float("nan")
    items = sorted(buckets.items())
    xs = np.array([k for k, _ in items], dtype=float)
    fracs = np.array([np.mean(v) for _, v in items], dtype=float)

    # Find first crossing of threshold (monotone-ish curve).
    cross = None
    for i in range(len(fracs) - 1):
        if (fracs[i] - threshold) * (fracs[i + 1] - threshold) <= 0 and fracs[i] != fracs[i + 1]:
            x0, x1 = xs[i], xs[i + 1]
            y0, y1 = fracs[i], fracs[i + 1]
            cross = x0 + (threshold - y0) * (x1 - x0) / (y1 - y0)
            break
    if cross is None:
        # Constant curve; return midpoint
        return float(xs[len(xs) // 2])
    return float(cross)


# =========================================================================
# Concrete calibrators
# =========================================================================
def calibrate_lottery_switching_point(
    target_tokens: float,
    eval_lottery_at_lambda: Callable[[float], float],
    cache: Optional[Dict[float, float]] = None,
) -> Tuple[float, float]:
    """Find lambda for which the lottery switching point equals `target_tokens`."""
    lam, achieved, _ = coarse_grid_then_binary_search(
        eval_at_lambda=eval_lottery_at_lambda,
        target=target_tokens,
        coarse_grid=config.CALIBRATION["coarse_grid"],
        lambda_min=config.CALIBRATION["lambda_min"],
        lambda_max=config.CALIBRATION["lambda_max"],
        tolerance=config.CALIBRATION["tolerance_switching_point_tokens"],
        max_iter=config.CALIBRATION["binary_search_max_iter"],
        cache=cache,
    )
    return float(lam), float(achieved)


def calibrate_ultimatum_acceptance_threshold(
    target_tokens: float,
    eval_ultimatum_at_lambda: Callable[[float], float],
    cache: Optional[Dict[float, float]] = None,
) -> Tuple[float, float]:
    lam, achieved, _ = coarse_grid_then_binary_search(
        eval_at_lambda=eval_ultimatum_at_lambda,
        target=target_tokens,
        coarse_grid=config.CALIBRATION["coarse_grid"],
        lambda_min=config.CALIBRATION["lambda_min"],
        lambda_max=config.CALIBRATION["lambda_max"],
        tolerance=config.CALIBRATION["tolerance_acceptance_threshold_tokens"],
        max_iter=config.CALIBRATION["binary_search_max_iter"],
        cache=cache,
    )
    return float(lam), float(achieved)


def calibrate_creativity_target(
    target_score: float,
    eval_creativity_at_lambda: Callable[[float], float],
    cache: Optional[Dict[float, float]] = None,
) -> Tuple[float, float]:
    lam, achieved, _ = coarse_grid_then_binary_search(
        eval_at_lambda=eval_creativity_at_lambda,
        target=target_score,
        coarse_grid=config.CALIBRATION["coarse_grid"],
        lambda_min=config.CALIBRATION["lambda_min"],
        lambda_max=config.CALIBRATION["lambda_max"],
        tolerance=config.CALIBRATION["tolerance_creativity_score"],
        max_iter=config.CALIBRATION["binary_search_max_iter"],
        cache=cache,
    )
    return float(lam), float(achieved)


# =========================================================================
# Lambda cache I/O
# =========================================================================
def save_cache(path: Path, cache: Dict[float, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump({f"{k:.4f}": v for k, v in cache.items()}, f, indent=2)


def load_cache(path: Path) -> Dict[float, float]:
    if not path.exists():
        return {}
    with open(path) as f:
        return {float(k): float(v) for k, v in json.load(f).items()}
