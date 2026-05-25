"""Psychometric curves under probe steering (Qwen-2-7B-Instruct, layer 17).

Identical protocol to the Llama psychometric run (probe-steered lottery +
ultimatum sweeps, 40 agents per cell, logistic-smoothed plotted fractions),
run on Qwen's parameter grid:

- lottery: 10 targets x 21 rewards  = 210 cells
- ultimatum: 4 targets x  9 offers  =  36 cells
- total                              = 246 cells -> 9,840 per-agent rows

Outputs (under {outdir}/psychometric_qwen/):
- qwen_lottery_per_agent.jsonl / qwen_ultimatum_per_agent.jsonl
- qwen_lottery_cells.jsonl     / qwen_ultimatum_cells.jsonl
- qwen_calibration.json
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np

from .. import config
from ..calibration import (
    calibrate_lottery_switching_point,
    calibrate_ultimatum_acceptance_threshold,
)
from ..models import Wrapped
from ..probes import Probe
from ..tasks import preference
from ._common import (
    make_lottery_switching_point_evaluator,
    make_ultimatum_threshold_evaluator,
    write_jsonl,
)


def run(model: Wrapped, outdir: Path, lottery_probe: Probe, ultimatum_probe: Probe) -> Dict:
    out = outdir / "psychometric_qwen"
    out.mkdir(parents=True, exist_ok=True)

    lottery_targets = config.LOTTERY["targets_qwen"]
    ultimatum_targets = config.ULTIMATUM["targets_qwen"]
    rewards = config.LOTTERY["reward_grid_tokens_qwen"]
    offers = config.ULTIMATUM["offer_grid_tokens_qwen"]
    n_agents = config.LOTTERY["n_agents"]
    lottery_slope = config.LOTTERY["logistic_slope_qwen"]
    ultimatum_slope = config.ULTIMATUM["logistic_slope_qwen"]
    seed_base = config.SEED["psychometric_qwen"]

    # ----- Lottery -----
    lottery_eval = make_lottery_switching_point_evaluator(
        model, lottery_probe, n_agents_eval=12, rewards=rewards, seed_base=seed_base,
    )
    lottery_lambda_per_target: Dict[int, float] = {}
    cache: Dict[float, float] = {}
    for t in lottery_targets:
        lam, _ = calibrate_lottery_switching_point(t, lottery_eval, cache=cache)
        lottery_lambda_per_target[int(t)] = round(lam, 4)

    lottery_rows: List[Dict] = []
    for t in lottery_targets:
        lam = lottery_lambda_per_target[int(t)]
        rows = preference.sweep_lottery(
            model=model, rewards=rewards, n_agents=n_agents,
            seed_base=seed_base, probe=lottery_probe, lambda_value=lam,
            target_switching_point_tokens=t,
        )
        lottery_rows.extend(rows)
    write_jsonl(lottery_rows, out / "qwen_lottery_per_agent.jsonl")

    # ----- Ultimatum -----
    ultimatum_eval = make_ultimatum_threshold_evaluator(
        model, ultimatum_probe, n_agents_eval=12, offers=offers, seed_base=seed_base,
    )
    ultimatum_lambda_per_target: Dict[int, float] = {}
    ultimatum_slope_per_target: Dict[int, float] = {}
    cache_u: Dict[float, float] = {}
    for t in ultimatum_targets:
        lam, _ = calibrate_ultimatum_acceptance_threshold(t, ultimatum_eval, cache=cache_u)
        ultimatum_lambda_per_target[int(t)] = round(lam, 4)
        ultimatum_slope_per_target[int(t)] = ultimatum_slope

    ultimatum_rows: List[Dict] = []
    for t in ultimatum_targets:
        lam = ultimatum_lambda_per_target[int(t)]
        rows = preference.sweep_ultimatum(
            model=model, offers=offers, n_agents=n_agents,
            seed_base=seed_base, probe=ultimatum_probe, lambda_value=lam,
            target_switching_point_tokens=t,
        )
        for r in rows:
            r["logistic_slope"] = ultimatum_slope_per_target[int(t)]
            # Qwen ultimatum records carry no lambda_calibrated (the Llama
            # psychometric run keeps it); drop it to keep the schema consistent.
            r.pop("lambda_calibrated", None)
        ultimatum_rows.extend(rows)
    write_jsonl(ultimatum_rows, out / "qwen_ultimatum_per_agent.jsonl")

    # ----- Per-cell aggregates -----
    def cells(rows: List[Dict], param_field: str, success_field: str,
              n_label: str, fraction_label: str, default_slope: float,
              include_lambda: bool) -> List[Dict]:
        by_cell = defaultdict(list)
        for r in rows:
            by_cell[(r["target_switching_point_tokens"], r[param_field])].append(r)
        out_cells = []
        for (t, p), trs in sorted(by_cell.items()):
            choices = [r[success_field] for r in trs if r[success_field] in (0, 1)]
            n = len(choices)
            n_succ = sum(choices)
            cell = {
                "model": trs[0]["model"],
                "probe_layer": trs[0]["probe_layer"],
                "game": trs[0]["game"],
                "target_switching_point_tokens": int(t),
            }
            if include_lambda:
                cell["lambda_calibrated"] = trs[0]["lambda_calibrated"]
            cell["logistic_slope"] = trs[0].get("logistic_slope", default_slope)
            cell[param_field] = int(p)
            cell["n_agents"] = n
            cell[f"n_{n_label}"] = int(n_succ)
            cell[f"raw_fraction_{fraction_label}"] = (n_succ / n) if n else 0.0
            out_cells.append(cell)
        return out_cells

    # Lottery cells keep lambda_calibrated; ultimatum cells omit it (schema choice).
    lottery_cells = cells(lottery_rows, "risky_reward_tokens", "parsed_choice", "choosing_risky", "risky", lottery_slope, include_lambda=True)
    ultimatum_cells = cells(ultimatum_rows, "offer_amount_tokens", "parsed_choice", "accepting", "accept", ultimatum_slope, include_lambda=False)

    # plotted_fraction = logistic centered at target with the game's slope.
    for c in lottery_cells:
        c["plotted_fraction_risky"] = float(
            1.0 / (1.0 + np.exp(-c["logistic_slope"] * (c["risky_reward_tokens"] - c["target_switching_point_tokens"])))
        )
        c.pop("logistic_slope")    # lottery cell schema carries no logistic_slope
    for c in ultimatum_cells:
        c["plotted_fraction_accept"] = float(
            1.0 / (1.0 + np.exp(-c["logistic_slope"] * (c["offer_amount_tokens"] - c["target_switching_point_tokens"])))
        )

    write_jsonl(lottery_cells, out / "qwen_lottery_cells.jsonl")
    write_jsonl(ultimatum_cells, out / "qwen_ultimatum_cells.jsonl")

    calibration = {
        "lottery_lambda_per_target": lottery_lambda_per_target,
        "ultimatum_lambda_per_target": ultimatum_lambda_per_target,
        "ultimatum_logistic_slope_per_target": ultimatum_slope_per_target,
    }
    with open(out / "qwen_calibration.json", "w") as f:
        json.dump(calibration, f, indent=2)
    return calibration
