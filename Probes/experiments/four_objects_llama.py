"""Probe transfer across four objects (Llama).

Probe trained on brick alternative-uses data, then applied as a fixed
steering direction to all four objects (brick, stapler, paperclip, bowl).
Lambdas are re-calibrated per (target, object) so this shows
controllability holds out-of-sample even when lambda is tuned per object.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from .. import config
from ..calibration import calibrate_creativity_target
from ..judge import score_response
from ..models import Wrapped
from ..probes import Probe
from ..tasks import capability as cap_task
from ._common import build_capability_probe, make_creativity_evaluator, write_jsonl


def run(model: Wrapped, outdir: Path) -> Dict:
    out = outdir / "four_objects_llama"
    out.mkdir(parents=True, exist_ok=True)
    task = "divergent_creativity"
    # Train probe on brick (in-distribution).
    probe = build_capability_probe(
        model=model, task=task, obj="brick",
        judge_score_fn=lambda t: (score_response(t, task=task) or {"creativity_score": 5.0})["creativity_score"],
    )

    out_rows: List[Dict] = []
    summary_per_object: Dict[str, Dict] = {}

    for obj in config.CAPABILITY["objects"]:
        eval_at = make_creativity_evaluator(
            model, probe, task, obj,
            lambda t: (score_response(t, task=task) or {"creativity_score": 5.0})["creativity_score"],
            n_agents_eval=8,
        )
        target_to_lambda: Dict[float, float] = {}
        cache: Dict[float, float] = {}
        for target in config.CAPABILITY["targets"]:
            lam, _ = calibrate_creativity_target(target, eval_at, cache=cache)
            target_to_lambda[float(target)] = round(lam, 4)

        rows = cap_task.sweep_capability_targets(
            model=model, probe=probe, task=task, obj=obj,
            target_to_lambda=target_to_lambda,
            n_agents=config.CAPABILITY["n_agents"],
            seed_base=config.SEED["capability_llama"],
        )
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

        out_rows.extend(scored)
        summary_per_object[obj] = {"target_to_lambda": target_to_lambda, "n_rows": len(scored)}

    write_jsonl(out_rows, out / "llama_four_objects_per_agent.jsonl")
    return {"summary_per_object": summary_per_object, "probe_layer": probe.layer}
