"""Dose-response on Qwen.

Reuses the Llama dose-response logic with the Qwen target ranges.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from .. import config
from ..calibration import (
    calibrate_lottery_switching_point,
    calibrate_ultimatum_acceptance_threshold,
)
from ..models import Wrapped
from ..probes import Probe
from ._common import (
    make_lottery_switching_point_evaluator,
    make_ultimatum_threshold_evaluator,
    write_jsonl,
)


def run(model: Wrapped, outdir: Path, lottery_probe: Probe, ultimatum_probe: Probe) -> Dict:
    out = outdir / "dose_response_qwen"
    out.mkdir(parents=True, exist_ok=True)

    seed_base = config.SEED["dose_response_qwen"]
    lottery_eval = make_lottery_switching_point_evaluator(
        model, lottery_probe, n_agents_eval=8,
        rewards=config.LOTTERY["reward_grid_tokens_qwen"],
        seed_base=seed_base,
    )
    ultimatum_eval = make_ultimatum_threshold_evaluator(
        model, ultimatum_probe, n_agents_eval=8,
        offers=config.ULTIMATUM["offer_grid_tokens_qwen"],
        seed_base=seed_base,
    )
    rows: List[Dict] = []
    cache: Dict[float, float] = {}
    for t in config.DOSE_RESPONSE["lottery_targets_qwen"]:
        lam, _ = calibrate_lottery_switching_point(t, lottery_eval, cache=cache)
        rows.append({
            "model": model.cfg.name,
            "probe_layer": model.cfg.probe_layer,
            "game": "lottery",
            "target_switching_point_tokens": int(t),
            "lambda_required": round(lam, 4),
            "calibration_procedure": "coarse_grid_then_bracketed_binary_search",
        })
    cache_u: Dict[float, float] = {}
    for t in config.DOSE_RESPONSE["ultimatum_targets_qwen"]:
        lam, _ = calibrate_ultimatum_acceptance_threshold(t, ultimatum_eval, cache=cache_u)
        rows.append({
            "model": model.cfg.name,
            "probe_layer": model.cfg.probe_layer,
            "game": "ultimatum",
            "target_switching_point_tokens": int(t),
            "lambda_required": round(lam, 4),
            "calibration_procedure": "coarse_grid_then_bracketed_binary_search",
        })

    write_jsonl(rows, out / "qwen_dose_response.jsonl")
    return {"n_rows": len(rows)}
