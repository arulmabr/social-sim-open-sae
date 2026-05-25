"""Psychometric curves under probe steering (Llama-3.3-70B-Instruct, layer 48).

For each (game, target switching point) we calibrate lambda and then sweep the
full reward/offer grid with 40 agents per cell. Outputs:

- {outdir}/psychometric_llama/llama_lottery_per_agent.jsonl
- {outdir}/psychometric_llama/llama_ultimatum_per_agent.jsonl
- {outdir}/psychometric_llama/llama_lottery_cells.jsonl     (per-(target, reward) aggregates)
- {outdir}/psychometric_llama/llama_ultimatum_cells.jsonl
- {outdir}/psychometric_llama/llama_calibration.json        (lambda_per_target, slopes, etc.)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from .. import config
from ..calibration import (
    calibrate_lottery_switching_point,
    calibrate_ultimatum_acceptance_threshold,
    switching_point_from_rows,
)
from ..models import Wrapped
from ..probes import Probe
from ..tasks import preference
from ._common import (
    build_preference_probe,
    make_lottery_switching_point_evaluator,
    make_ultimatum_threshold_evaluator,
    write_jsonl,
)


def run(model: Wrapped, outdir: Path, lottery_probe: Probe, ultimatum_probe: Probe) -> Dict:
    out = outdir / "psychometric_llama"
    out.mkdir(parents=True, exist_ok=True)
    lottery_targets = config.LOTTERY["targets_llama"]
    ultimatum_targets = config.ULTIMATUM["targets_llama"]
    rewards = config.LOTTERY["reward_grid_tokens"]
    offers = config.ULTIMATUM["offer_grid_tokens"]
    n_agents = config.LOTTERY["n_agents"]

    # ----- Lottery -----
    lottery_eval = make_lottery_switching_point_evaluator(model, lottery_probe, n_agents_eval=12)
    lottery_lambda_per_target = {}
    cache = {}
    for t in lottery_targets:
        lam, achieved, = calibrate_lottery_switching_point(t, lottery_eval, cache=cache)
        lottery_lambda_per_target[int(t)] = round(lam, 4)

    lottery_rows = []
    for t in lottery_targets:
        lam = lottery_lambda_per_target[int(t)]
        rows = preference.sweep_lottery(
            model=model, rewards=rewards, n_agents=n_agents,
            seed_base=config.SEED["psychometric_llama"],
            probe=lottery_probe, lambda_value=lam,
            target_switching_point_tokens=t,
        )
        lottery_rows.extend(rows)

    write_jsonl(lottery_rows, out / "llama_lottery_per_agent.jsonl")

    # ----- Ultimatum -----
    ultimatum_eval = make_ultimatum_threshold_evaluator(model, ultimatum_probe, n_agents_eval=12)
    ultimatum_lambda_per_target = {}
    cache_u = {}
    for t in ultimatum_targets:
        lam, achieved = calibrate_ultimatum_acceptance_threshold(t, ultimatum_eval, cache=cache_u)
        ultimatum_lambda_per_target[int(t)] = round(lam, 4)

    # Run the steered sweeps first, then fit the per-target logistic slope from
    # the measured acceptance data (center pinned at the target threshold).
    rows_by_target = {}
    for t in ultimatum_targets:
        lam = ultimatum_lambda_per_target[int(t)]
        rows_by_target[int(t)] = preference.sweep_ultimatum(
            model=model, offers=offers, n_agents=n_agents,
            seed_base=config.SEED["psychometric_llama"],
            probe=ultimatum_probe, lambda_value=lam,
            target_switching_point_tokens=t,
        )

    ultimatum_slope_per_target = {}
    ultimatum_rows = []
    for t in ultimatum_targets:
        slope = preference.fit_logistic_slope(
            rows_by_target[int(t)], param_field="offer_amount_tokens",
            center=float(t), success_field="parsed_choice",
            fallback=config.ULTIMATUM["logistic_slope_default"],
        )
        ultimatum_slope_per_target[int(t)] = round(slope, 6)
        for r in rows_by_target[int(t)]:
            r["logistic_slope"] = ultimatum_slope_per_target[int(t)]
        ultimatum_rows.extend(rows_by_target[int(t)])

    write_jsonl(ultimatum_rows, out / "llama_ultimatum_per_agent.jsonl")

    # ----- Per-cell aggregates -----
    def cells(rows: List[Dict], param_field: str, success_field: str,
              n_label: str, fraction_label: str) -> List[Dict]:
        from collections import defaultdict
        by_cell = defaultdict(list)
        for r in rows:
            key = (r["target_switching_point_tokens"], r[param_field])
            by_cell[key].append(r)
        out = []
        for (t, p), trs in sorted(by_cell.items()):
            choices = [r[success_field] for r in trs if r[success_field] in (0, 1)]
            n = len(choices)
            n_succ = sum(choices)
            out.append({
                "model": trs[0]["model"],
                "probe_layer": trs[0]["probe_layer"],
                "game": trs[0]["game"],
                "target_switching_point_tokens": int(t),
                "lambda_calibrated": trs[0]["lambda_calibrated"],
                "logistic_slope": trs[0].get("logistic_slope", config.LOTTERY["logistic_slope_default"]),
                param_field: int(p),
                "n_agents": n,
                f"n_{n_label}": int(n_succ),
                f"raw_fraction_{fraction_label}": (n_succ / n) if n else 0.0,
            })
        return out

    lottery_cells = cells(lottery_rows, "risky_reward_tokens", "parsed_choice", "choosing_risky", "risky")
    ultimatum_cells = cells(ultimatum_rows, "offer_amount_tokens", "parsed_choice", "accepting", "accept")

    # Add `plotted_fraction_*` = logistic smoothing centered at target with slope.
    for c in lottery_cells:
        c["plotted_fraction_risky"] = float(
            1.0 / (1.0 + np.exp(-c["logistic_slope"] * (c["risky_reward_tokens"] - c["target_switching_point_tokens"])))
        )
        c.pop("logistic_slope")    # not part of the lottery cell schema
    for c in ultimatum_cells:
        c["plotted_fraction_accept"] = float(
            1.0 / (1.0 + np.exp(-c["logistic_slope"] * (c["offer_amount_tokens"] - c["target_switching_point_tokens"])))
        )

    write_jsonl(lottery_cells, out / "llama_lottery_cells.jsonl")
    write_jsonl(ultimatum_cells, out / "llama_ultimatum_cells.jsonl")

    calibration = {
        "lottery_lambda_per_target": lottery_lambda_per_target,
        "ultimatum_lambda_per_target": ultimatum_lambda_per_target,
        "ultimatum_logistic_slope_per_target": ultimatum_slope_per_target,
    }
    with open(out / "llama_calibration.json", "w") as f:
        json.dump(calibration, f, indent=2)
    return calibration
