"""Capability (creativity / product-innovation) decoding loop.

Each call generates the model's full text response under probe steering at a
calibrated lambda. Judge scoring happens in `judge.py` afterwards.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from .. import config
from ..models import Wrapped
from ..probes import Probe
from . import prompts


def run_capability_trial(
    model: Wrapped,
    task: str,
    obj: str,
    agent_id: int,
    seed: int,
    probe: Optional[Probe] = None,
    lambda_value: float = 0.0,
    sign: int = +1,
) -> Dict:
    prompt_text, prompt_id = prompts.get_capability_prompt(task, obj)
    if probe is not None and abs(lambda_value) > 1e-8:
        with model.steering_at(probe.layer, probe.direction_unit, lambda_value, sign=sign):
            result = model.generate(
                prompt_text,
                max_new_tokens=config.DECODING_DEFAULTS["max_new_tokens_capability"],
                temperature=config.DECODING_DEFAULTS["temperature_capability"],
                top_p=config.DECODING_DEFAULTS["top_p"],
                seed=seed,
            )
    else:
        result = model.generate(
            prompt_text,
            max_new_tokens=config.DECODING_DEFAULTS["max_new_tokens_capability"],
            temperature=config.DECODING_DEFAULTS["temperature_capability"],
            top_p=config.DECODING_DEFAULTS["top_p"],
            seed=seed,
        )
    return {
        "agent_id": config.agent_id_str(agent_id),
        "model": model.cfg.name,
        "probe_layer": model.cfg.probe_layer,
        "task": task,
        "object": obj,
        "prompt_id": prompt_id,
        "prompt": prompt_text,
        "response": result.text,
        "lambda_calibrated": float(lambda_value),
    }


def sweep_capability_targets(
    model: Wrapped,
    probe: Probe,
    task: str,
    obj: str,
    target_to_lambda: Dict[float, float],
    n_agents: int,
    seed_base: int,
) -> List[Dict]:
    """Run all (target, agent) trials for a single (task, object).

    `target_to_lambda` maps each target creativity score (e.g. 3.0, 5.0, 7.0, 9.0)
    to the lambda that was calibrated to hit that target.
    """
    rows = []
    for param_idx, (target, lam) in enumerate(sorted(target_to_lambda.items())):
        for i in range(1, n_agents + 1):
            seed = config.agent_seed(seed_base, param_idx, i)
            row = run_capability_trial(
                model, task=task, obj=obj, agent_id=i, seed=seed,
                probe=probe, lambda_value=lam,
            )
            row["target_creativity_score"] = float(target)
            rows.append(row)
    return rows
