"""Probe-score tracking under steering (Llama).

For each target switching point, run the steered sweep, compute the mean probe
activation `s = w_hat^T h` on the prompt context, and split by the parsed
choice (Risky vs Safe, Accept vs Reject).
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np

from .. import config
from ..calibration import calibrate_lottery_switching_point, calibrate_ultimatum_acceptance_threshold
from ..models import Wrapped
from ..probes import Probe
from ..tasks import preference, prompts
from ._common import (
    make_lottery_switching_point_evaluator,
    make_ultimatum_threshold_evaluator,
    write_jsonl,
)


def _probe_score(model: Wrapped, probe: Probe, prompt_text: str) -> float:
    h = model.capture_activations(prompt_text, probe.layer)
    return probe.score(h)


def run(model: Wrapped, outdir: Path, lottery_probe: Probe, ultimatum_probe: Probe) -> Dict:
    out = outdir / "probe_tracking_llama"
    out.mkdir(parents=True, exist_ok=True)
    out_rows: List[Dict] = []

    # ----- Lottery -----
    lottery_eval = make_lottery_switching_point_evaluator(model, lottery_probe, n_agents_eval=8)
    cache: Dict[float, float] = {}
    for t in config.TRACKING["lottery_targets_llama"]:
        lam, _ = calibrate_lottery_switching_point(t, lottery_eval, cache=cache)
        rows = preference.sweep_lottery(
            model=model, rewards=config.LOTTERY["reward_grid_tokens"], n_agents=8,
            seed_base=config.SEED["tracking_llama"],
            probe=lottery_probe, lambda_value=lam,
            target_switching_point_tokens=t,
        )
        # For each row compute the probe score on its prompt.
        scores_by_choice = defaultdict(list)
        for r in rows:
            s = _probe_score(model, lottery_probe, r["prompt"])
            scores_by_choice["all"].append(s)
            if r["parsed_choice"] == 1:
                scores_by_choice["risky"].append(s)
            elif r["parsed_choice"] == 0:
                scores_by_choice["safe"].append(s)
        for subset in ("all", "risky", "safe"):
            mean_s = float(np.mean(scores_by_choice[subset])) if scores_by_choice[subset] else float("nan")
            out_rows.append({
                "model": model.cfg.name,
                "probe_layer": model.cfg.probe_layer,
                "game": "lottery",
                "target_switching_point_tokens": int(t),
                "choice_subset": subset,
                "mean_probe_activation": round(mean_s, 6),
            })

    # ----- Ultimatum -----
    ultimatum_eval = make_ultimatum_threshold_evaluator(model, ultimatum_probe, n_agents_eval=8)
    cache_u: Dict[float, float] = {}
    for t in config.TRACKING["ultimatum_targets_llama"]:
        lam, _ = calibrate_ultimatum_acceptance_threshold(t, ultimatum_eval, cache=cache_u)
        rows = preference.sweep_ultimatum(
            model=model, offers=config.ULTIMATUM["offer_grid_tokens"], n_agents=8,
            seed_base=config.SEED["tracking_llama"],
            probe=ultimatum_probe, lambda_value=lam,
            target_switching_point_tokens=t,
        )
        scores_by_choice = defaultdict(list)
        for r in rows:
            s = _probe_score(model, ultimatum_probe, r["prompt"])
            scores_by_choice["all"].append(s)
            if r["parsed_choice"] == 1:
                scores_by_choice["accept"].append(s)
            elif r["parsed_choice"] == 0:
                scores_by_choice["reject"].append(s)
        for subset in ("all", "accept", "reject"):
            mean_s = float(np.mean(scores_by_choice[subset])) if scores_by_choice[subset] else float("nan")
            out_rows.append({
                "model": model.cfg.name,
                "probe_layer": model.cfg.probe_layer,
                "game": "ultimatum",
                "target_switching_point_tokens": int(t),
                "choice_subset": subset,
                "mean_probe_activation": round(mean_s, 6),
            })

    write_jsonl(out_rows, out / "llama_probe_scores_tracking.jsonl")
    return {"n_rows": len(out_rows)}
