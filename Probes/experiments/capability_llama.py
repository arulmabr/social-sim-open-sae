"""In-distribution capability control (Llama).

Two task-specific probes:
- brick: divergent_creativity probe trained on brick, evaluated on brick.
- stapler: product_innovation probe trained on stapler enhancements, evaluated on stapler.

Pipeline per probe:
1. Train task-specific probe (`build_capability_probe`).
2. Calibrate lambda per target creativity score in [3, 5, 7, 9].
3. Run 40 agents per target; judge each response.
4. Save per-agent JSONL.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from .. import config
from ..calibration import calibrate_creativity_target
from ..judge import score_response
from ..models import Wrapped
from ..probes import Probe
from ..tasks import capability as cap_task
from ..tasks import prompts
from ._common import build_capability_probe, make_creativity_evaluator, write_jsonl


def _judge_fn(text: str, task: str) -> float:
    r = score_response(text, task=task, judge_name=config.CAPABILITY["judge_hf_id_or_provider"])
    return r["creativity_score"] if r else 5.0


def _calibrate_and_score(
    model: Wrapped, probe: Probe, task: str, obj: str, n_agents: int, seed_base: int,
) -> Tuple[Dict[float, float], List[Dict]]:
    judge_score = lambda text: _judge_fn(text, task)

    eval_at = make_creativity_evaluator(model, probe, task, obj, judge_score, n_agents_eval=8)

    target_to_lambda: Dict[float, float] = {}
    cache: Dict[float, float] = {}
    for target in config.CAPABILITY["targets"]:
        lam, _ = calibrate_creativity_target(target, eval_at, cache=cache)
        target_to_lambda[float(target)] = round(lam, 4)

    rows = cap_task.sweep_capability_targets(
        model=model, probe=probe, task=task, obj=obj,
        target_to_lambda=target_to_lambda,
        n_agents=n_agents, seed_base=seed_base,
    )
    # Score each row.
    scored = []
    for r in rows:
        s = score_response(r["response"], task=task, judge_name=config.CAPABILITY["judge_hf_id_or_provider"])
        if s is None:
            r2 = dict(r, scores=None, creativity_score=None, judge=config.CAPABILITY["judge_name"])
        else:
            r2 = dict(
                r,
                judge=config.CAPABILITY["judge_name"],
                scores={k: int(s[k]) for k in ("fluency", "flexibility", "originality", "elaboration")},
                creativity_score=float(s["creativity_score"]),
            )
        scored.append(r2)
    return target_to_lambda, scored


def run_brick(model: Wrapped, outdir: Path) -> Dict:
    out = outdir / "capability_llama"
    out.mkdir(parents=True, exist_ok=True)
    probe = build_capability_probe(
        model=model, task="divergent_creativity", obj="brick",
        judge_score_fn=lambda t: _judge_fn(t, "divergent_creativity"),
    )
    target_to_lambda, scored = _calibrate_and_score(
        model=model, probe=probe, task="divergent_creativity", obj="brick",
        n_agents=config.CAPABILITY["n_agents"],
        seed_base=config.SEED["capability_llama"],
    )
    write_jsonl(scored, out / "llama_brick_per_agent.jsonl")
    return {"target_to_lambda": target_to_lambda, "n_rows": len(scored), "probe_layer": probe.layer}


def run_stapler_product_innovation(model: Wrapped, outdir: Path) -> Dict:
    out = outdir / "capability_llama"
    out.mkdir(parents=True, exist_ok=True)
    probe = build_capability_probe(
        model=model, task="product_innovation", obj="stapler",
        judge_score_fn=lambda t: _judge_fn(t, "product_innovation"),
    )
    target_to_lambda, scored = _calibrate_and_score(
        model=model, probe=probe, task="product_innovation", obj="stapler",
        n_agents=config.CAPABILITY["n_agents"],
        seed_base=config.SEED["capability_llama"],
    )
    write_jsonl(scored, out / "llama_stapler_product_innovation_per_agent.jsonl")
    return {"target_to_lambda": target_to_lambda, "n_rows": len(scored), "probe_layer": probe.layer}
