"""Lottery + ultimatum decoding loops.

Each function returns per-agent trial records with the exact fields the
output JSON expects.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import numpy as np

from .. import config
from ..models import Wrapped
from ..probes import Probe
from . import prompts


# =========================================================================
# Lottery
# =========================================================================
def run_lottery_trial(
    model: Wrapped,
    risky_reward: int,
    agent_id: int,
    seed: int,
    probe: Optional[Probe] = None,
    lambda_value: float = 0.0,
    sign: int = +1,
) -> Dict:
    prompt = prompts.lottery_prompt(risky_reward)
    if probe is not None and abs(lambda_value) > 1e-8:
        with model.steering_at(probe.layer, probe.direction_unit, lambda_value, sign=sign):
            result = model.generate(
                prompt,
                max_new_tokens=config.DECODING_DEFAULTS["max_new_tokens_choice"],
                temperature=config.DECODING_DEFAULTS["temperature_choice"],
                top_p=config.DECODING_DEFAULTS["top_p"],
                seed=seed,
            )
    else:
        result = model.generate(
            prompt,
            max_new_tokens=config.DECODING_DEFAULTS["max_new_tokens_choice"],
            temperature=config.DECODING_DEFAULTS["temperature_choice"],
            top_p=config.DECODING_DEFAULTS["top_p"],
            seed=seed,
        )
    parsed = prompts.parse_lottery_choice(result.text)
    return {
        "agent_id": config.agent_id_str(agent_id),
        "model": model.cfg.name,
        "probe_layer": model.cfg.probe_layer,
        "game": "lottery",
        "risky_reward_tokens": int(risky_reward),
        "lambda_calibrated": float(lambda_value),
        "prompt": prompt,
        "response": "Risky" if parsed == 1 else ("Safe" if parsed == 0 else result.text.strip()),
        "parsed_choice": int(parsed) if parsed is not None else -1,
    }


def sweep_lottery(
    model: Wrapped,
    rewards: Iterable[int],
    n_agents: int,
    seed_base: int,
    probe: Optional[Probe] = None,
    lambda_value: float = 0.0,
    target_switching_point_tokens: Optional[float] = None,
) -> List[Dict]:
    rows = []
    for param_idx, reward in enumerate(rewards):
        for i in range(1, n_agents + 1):
            seed = config.agent_seed(seed_base, param_idx, i)
            row = run_lottery_trial(
                model, risky_reward=reward, agent_id=i, seed=seed,
                probe=probe, lambda_value=lambda_value,
            )
            if target_switching_point_tokens is not None:
                row["target_switching_point_tokens"] = float(target_switching_point_tokens)
            rows.append(row)
    return rows


# =========================================================================
# Ultimatum
# =========================================================================
def run_ultimatum_trial(
    model: Wrapped,
    offer_amount: int,
    agent_id: int,
    seed: int,
    probe: Optional[Probe] = None,
    lambda_value: float = 0.0,
    sign: int = +1,
) -> Dict:
    prompt = prompts.ultimatum_prompt(offer_amount)
    if probe is not None and abs(lambda_value) > 1e-8:
        with model.steering_at(probe.layer, probe.direction_unit, lambda_value, sign=sign):
            result = model.generate(
                prompt,
                max_new_tokens=config.DECODING_DEFAULTS["max_new_tokens_choice"],
                temperature=config.DECODING_DEFAULTS["temperature_choice"],
                top_p=config.DECODING_DEFAULTS["top_p"],
                seed=seed,
            )
    else:
        result = model.generate(
            prompt,
            max_new_tokens=config.DECODING_DEFAULTS["max_new_tokens_choice"],
            temperature=config.DECODING_DEFAULTS["temperature_choice"],
            top_p=config.DECODING_DEFAULTS["top_p"],
            seed=seed,
        )
    parsed = prompts.parse_ultimatum_choice(result.text)
    return {
        "agent_id": config.agent_id_str(agent_id),
        "model": model.cfg.name,
        "probe_layer": model.cfg.probe_layer,
        "game": "ultimatum",
        "offer_amount_tokens": int(offer_amount),
        "lambda_calibrated": float(lambda_value),
        "prompt": prompt,
        "response": "Accept" if parsed == 1 else ("Reject" if parsed == 0 else result.text.strip()),
        "parsed_choice": int(parsed) if parsed is not None else -1,
    }


def sweep_ultimatum(
    model: Wrapped,
    offers: Iterable[int],
    n_agents: int,
    seed_base: int,
    probe: Optional[Probe] = None,
    lambda_value: float = 0.0,
    target_switching_point_tokens: Optional[float] = None,
) -> List[Dict]:
    rows = []
    for param_idx, offer in enumerate(offers):
        for i in range(1, n_agents + 1):
            seed = config.agent_seed(seed_base, param_idx, i)
            row = run_ultimatum_trial(
                model, offer_amount=offer, agent_id=i, seed=seed,
                probe=probe, lambda_value=lambda_value,
            )
            if target_switching_point_tokens is not None:
                row["target_switching_point_tokens"] = float(target_switching_point_tokens)
            rows.append(row)
    return rows


# =========================================================================
# Aggregates + logistic fit (psychometric plotted_fraction)
# =========================================================================
def aggregate_by_param(rows: List[Dict], param_field: str, success_field: str = "parsed_choice") -> List[Dict]:
    """Return per-(param value) summary cells with raw_fraction and n counts."""
    from collections import defaultdict
    buckets = defaultdict(list)
    for r in rows:
        buckets[r[param_field]].append(r[success_field])
    out = []
    for param_value, choices in sorted(buckets.items()):
        choices = [c for c in choices if c in (0, 1)]
        n = len(choices)
        n_success = sum(choices)
        out.append({
            "param_value": param_value,
            "n_agents": n,
            "n_success": n_success,
            "raw_fraction": (n_success / n) if n > 0 else 0.0,
        })
    return out


def logistic_smooth(
    param_values: np.ndarray,
    fractions: np.ndarray,
    switching_point: float,
    slope: float,
) -> np.ndarray:
    """Return logistic-smoothed fractions at each param value.

    `plotted_fraction_*` is a logistic centered at the calibrated switching
    point with the recorded slope.
    """
    x = np.asarray(param_values, dtype=float)
    return 1.0 / (1.0 + np.exp(-slope * (x - switching_point)))


def fit_logistic_slope(
    rows: List[Dict],
    param_field: str,
    center: float,
    success_field: str = "parsed_choice",
    fallback: float = 0.2,
) -> float:
    """Fit the slope `k` of P(success) = sigmoid(k * (param - center)) from the
    measured per-agent choices, with the center pinned at `center`.

    This is the maximum-likelihood slope of a one-parameter logistic (no
    intercept; center fixed at the calibrated switching point), estimated
    directly from the binary outcomes. Returns `fallback` when the outcomes are
    degenerate (all-accept or all-reject), where the slope is unidentifiable.
    """
    xs: List[float] = []
    ys: List[int] = []
    for r in rows:
        v = r.get(success_field)
        if v in (0, 1):
            xs.append(float(r[param_field]) - float(center))
            ys.append(int(v))
    if len(set(ys)) < 2:
        return float(fallback)
    from sklearn.linear_model import LogisticRegression
    X = np.asarray(xs, dtype=float).reshape(-1, 1)
    y = np.asarray(ys, dtype=int)
    # Large C => effectively unregularized MLE; fit_intercept=False pins center.
    clf = LogisticRegression(C=1e6, fit_intercept=False, solver="lbfgs", max_iter=1000)
    clf.fit(X, y)
    return float(clf.coef_.ravel()[0])
