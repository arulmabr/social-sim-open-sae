"""Aggregate per-experiment JSONL logs into the 13-key probe-results JSON.

Top-level keys (each maps to one probe figure):
- figure_7_psychometric_curves_llama:               metadata + data + per_agent_rows + row_count
- figure_9_capability_brick_target_vs_achieved:     metadata + data
- figure_9_capability_stapler_product_innovation_*: metadata + data
- figure_10_capability_four_objects_*:              metadata + data
- figure_11_dose_response_lottery_ultimatum_llama:  metadata + data
- figure_12_probe_scores_track_target_*:            metadata + data
- figure_13_cross_object_generalization_llama:      metadata + data
- figure_14-19: Qwen variants of the above
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np

from . import config
from .experiments._common import read_jsonl


# =========================================================================
# Field re-ordering for a stable output schema
# =========================================================================
LOTTERY_CELL_ORDER = [
    "model", "probe_layer", "game", "target_switching_point_tokens",
    "lambda_calibrated", "risky_reward_tokens",
    "n_agents", "n_choosing_risky", "raw_fraction_risky", "plotted_fraction_risky",
]
ULTIMATUM_CELL_ORDER = [
    "model", "probe_layer", "game", "target_switching_point_tokens",
    "lambda_calibrated", "logistic_slope", "offer_amount_tokens",
    "n_agents", "n_accepting", "raw_fraction_accept", "plotted_fraction_accept",
]

LOTTERY_PER_AGENT_ORDER = [
    "agent_id", "model", "probe_layer", "game",
    "target_switching_point_tokens", "lambda_calibrated",
    "risky_reward_tokens", "prompt", "response", "parsed_choice",
]
ULTIMATUM_PER_AGENT_ORDER = [
    "agent_id", "model", "probe_layer", "game",
    "target_switching_point_tokens",
    "offer_amount_tokens", "prompt", "response", "parsed_choice",
    "lambda_calibrated", "logistic_slope",
]

CAPABILITY_ORDER = [
    "model", "judge", "probe_layer", "task", "object",
    "prompt_id", "target_creativity_score", "lambda_calibrated",
    "agent_id", "scores", "creativity_score", "response", "prompt",
]


def _ordered(d: Dict, order: List[str]) -> Dict:
    """Return dict with keys in `order` first, then any remaining keys."""
    out = {k: d[k] for k in order if k in d}
    for k, v in d.items():
        if k not in out:
            out[k] = v
    return out


# Canonical display order for probe-tracking choice subsets.
_SUBSET_ORDER = ["all", "risky", "safe", "accept", "reject"]


def _subsets_from_rows(rows: List[Dict]) -> List[str]:
    """Derive the subset list from the choice_subset values actually present in
    the tracking rows, in canonical order. Computed from data, never assumed."""
    present = {r["choice_subset"] for r in rows if "choice_subset" in r}
    ordered = [s for s in _SUBSET_ORDER if s in present]
    # Append any non-canonical subsets deterministically so nothing is dropped.
    ordered += sorted(present - set(ordered))
    return ordered


# =========================================================================
# Figure builders
# =========================================================================
def build_psychometric_llama(run_dir: Path) -> Dict:
    lottery_cells = read_jsonl(run_dir / "psychometric_llama" / "llama_lottery_cells.jsonl")
    ultimatum_cells = read_jsonl(run_dir / "psychometric_llama" / "llama_ultimatum_cells.jsonl")
    lottery_per_agent = read_jsonl(run_dir / "psychometric_llama" / "llama_lottery_per_agent.jsonl")
    ultimatum_per_agent = read_jsonl(run_dir / "psychometric_llama" / "llama_ultimatum_per_agent.jsonl")
    with open(run_dir / "psychometric_llama" / "llama_calibration.json") as f:
        calib = json.load(f)

    # Order fields
    data = [_ordered(c, LOTTERY_CELL_ORDER) for c in lottery_cells] + \
           [_ordered(c, ULTIMATUM_CELL_ORDER) for c in ultimatum_cells]
    per_agent = [_ordered(r, LOTTERY_PER_AGENT_ORDER) for r in lottery_per_agent] + \
                [_ordered(r, ULTIMATUM_PER_AGENT_ORDER) for r in ultimatum_per_agent]

    metadata = {
        "model": config.LLAMA.name,
        "probe_layer": config.LLAMA.probe_layer,
        "games": ["lottery", "ultimatum"],
        "n_agents_per_condition": config.LOTTERY["n_agents"],
        "lottery_reward_grid_tokens": config.LOTTERY["reward_grid_tokens"],
        "lottery_targets_tokens": config.LOTTERY["targets_llama"],
        "ultimatum_offers_tokens": config.ULTIMATUM["offer_grid_tokens"],
        "ultimatum_targets_tokens": config.ULTIMATUM["targets_llama"],
        "lottery_logistic_slope": config.LOTTERY["logistic_slope_default"],
        "ultimatum_logistic_slope": config.ULTIMATUM["logistic_slope_default"],
        "ultimatum_logistic_slope_per_target": {
            str(k): v for k, v in calib["ultimatum_logistic_slope_per_target"].items()
        },
        "method": "probe_steering",
        "random_seed": config.SEED["psychometric_llama"],
    }
    return {
        "figure_name": "Performance of using probes to steer generative agents in preference tasks",
        "experiment_type": "probe_steering_psychometric",
        "metadata": metadata,
        "data": data,
        "per_agent_rows": per_agent,
        "row_count": len(per_agent),
    }


def _capability_summary(rows: List[Dict]) -> Dict:
    """Compute lambda_per_target, achieved_per_target, yerr_per_target."""
    by_target = defaultdict(list)
    lam_by_target: Dict[float, float] = {}
    for r in rows:
        t = r["target_creativity_score"]
        if r.get("creativity_score") is not None:
            by_target[t].append(r["creativity_score"])
        lam_by_target[t] = r["lambda_calibrated"]
    lambda_per_target = {str(int(t)): lam_by_target[t] for t in sorted(lam_by_target)}
    achieved_per_target = {}
    yerr_per_target = {}
    for t, vals in by_target.items():
        arr = np.array(vals)
        achieved_per_target[str(int(t))] = round(float(arr.mean()), 4)
        yerr_per_target[str(int(t))] = round(float(arr.std(ddof=1) / np.sqrt(len(arr))), 4)
    return lambda_per_target, achieved_per_target, yerr_per_target


def build_capability_brick_llama(run_dir: Path) -> Dict:
    rows = read_jsonl(run_dir / "capability_llama" / "llama_brick_per_agent.jsonl")
    data = [_ordered(r, CAPABILITY_ORDER) for r in rows]
    lam, achieved, yerr = _capability_summary(rows)
    metadata = {
        "model": config.LLAMA.name,
        "judge": config.CAPABILITY["judge_name"],
        "probe_layer": config.LLAMA.probe_layer,
        "task": "divergent_creativity",
        "object": "brick",
        "targets": config.CAPABILITY["targets"],
        "scoring_dimensions": list(config.CAPABILITY["scoring_dimensions"]),
        "score_range": list(config.CAPABILITY["score_range"]),
        "creativity_score_definition": "mean of the four GPT-5 sub-scores (fluency, flexibility, originality, elaboration)",
        "method": "probe_steering",
        "lambda_per_target": lam,
        "achieved_per_target": achieved,
        "yerr_per_target": yerr,
        "n_agents_per_point": config.CAPABILITY["n_agents"],
        "yerr_definition": f"Standard error of the mean across {config.CAPABILITY['n_agents']} per-agent creativity scores (SEM = SD / sqrt({config.CAPABILITY['n_agents']}))",
    }
    return {
        "figure_name": "Achieved creativity scores as a function of target scores for the brick prompt using probe-based steering on Llama-3.3-70B-Instruct (layer 48)",
        "experiment_type": "probe_steering_capability_target_vs_achieved",
        "metadata": metadata,
        "data": data,
    }


def build_capability_stapler_llama(run_dir: Path) -> Dict:
    rows = read_jsonl(run_dir / "capability_llama" / "llama_stapler_product_innovation_per_agent.jsonl")
    data = [_ordered(r, CAPABILITY_ORDER) for r in rows]
    lam, achieved, yerr = _capability_summary(rows)
    metadata = {
        "model": config.LLAMA.name,
        "judge": config.CAPABILITY["judge_name"],
        "probe_layer": config.LLAMA.probe_layer,
        "task": "product_innovation",
        "object": "stapler",
        "targets": config.CAPABILITY["targets"],
        "scoring_dimensions": list(config.CAPABILITY["scoring_dimensions"]),
        "score_range": list(config.CAPABILITY["score_range"]),
        "creativity_score_definition": "mean of the four GPT-5 sub-scores (fluency, flexibility, originality, elaboration)",
        "method": "probe_steering",
        "lambda_per_target": lam,
        "achieved_per_target": achieved,
        "yerr_per_target": yerr,
        "n_agents_per_point": config.CAPABILITY["n_agents"],
        "yerr_definition": f"Standard error of the mean across {config.CAPABILITY['n_agents']} per-agent creativity scores (SEM = SD / sqrt({config.CAPABILITY['n_agents']}))",
    }
    return {
        "figure_name": "In-distribution capability control for product-innovation probe on stapler enhancements",
        "experiment_type": "probe_steering_capability_target_vs_achieved",
        "metadata": metadata,
        "data": data,
    }


def build_four_objects_llama(run_dir: Path) -> Dict:
    rows = read_jsonl(run_dir / "four_objects_llama" / "llama_four_objects_per_agent.jsonl")
    data = [_ordered(r, CAPABILITY_ORDER) for r in rows]
    # Per-object summary
    summary_per_object: Dict[str, Dict] = {}
    by_obj = defaultdict(list)
    for r in rows:
        by_obj[r["object"]].append(r)
    for obj, obj_rows in by_obj.items():
        lam, achieved, yerr = _capability_summary(obj_rows)
        summary_per_object[obj] = {
            "lambda_per_target": lam,
            "achieved_per_target": achieved,
            "yerr_per_target": yerr,
        }
    metadata = {
        "model": config.LLAMA.name,
        "judge": config.CAPABILITY["judge_name"],
        "probe_layer": config.LLAMA.probe_layer,
        "task": "divergent_creativity",
        "objects": list(config.CAPABILITY["objects"]),
        "targets": config.CAPABILITY["targets"],
        "scoring_dimensions": list(config.CAPABILITY["scoring_dimensions"]),
        "score_range": list(config.CAPABILITY["score_range"]),
        "method": "probe_steering",
        "summary_per_object": summary_per_object,
        "n_agents_per_point": config.CAPABILITY["n_agents"],
        "yerr_definition": f"Standard error of the mean across {config.CAPABILITY['n_agents']} per-agent creativity scores (SEM = SD / sqrt({config.CAPABILITY['n_agents']}))",
    }
    return {
        "figure_name": "Achieved creativity scores across all four object prompts (brick, stapler, paperclip, bowl) using probe-based steering on Llama-3.3-70B-Instruct (layer 48)",
        "experiment_type": "probe_steering_capability_target_vs_achieved",
        "metadata": metadata,
        "data": data,
    }


def build_dose_response_llama(run_dir: Path) -> Dict:
    rows = read_jsonl(run_dir / "dose_response_llama" / "llama_dose_response.jsonl")
    metadata = {
        "model": config.LLAMA.name,
        "probe_layer": config.LLAMA.probe_layer,
        "games": ["lottery", "ultimatum"],
        "lottery_target_range_tokens": [
            min(config.DOSE_RESPONSE["lottery_targets_llama"]),
            max(config.DOSE_RESPONSE["lottery_targets_llama"]),
        ],
        "ultimatum_target_range_tokens": [
            min(config.DOSE_RESPONSE["ultimatum_targets_llama"]),
            max(config.DOSE_RESPONSE["ultimatum_targets_llama"]),
        ],
        "lambda_range": [config.CALIBRATION["lambda_min"], config.CALIBRATION["lambda_max"]],
        "method": "probe_steering",
        "calibration_procedure": "coarse_grid_then_bracketed_binary_search",
        "random_seed": config.SEED["dose_response_llama"],
    }
    return {
        "figure_name": "Dose-response showing lambda required to reach target switching points for lottery (left) and ultimatum (right) games on Llama-3.3-70B-Instruct (layer 48)",
        "experiment_type": "probe_steering_dose_response_calibration",
        "metadata": metadata,
        "data": rows,
    }


def build_probe_tracking_llama(run_dir: Path) -> Dict:
    rows = read_jsonl(run_dir / "probe_tracking_llama" / "llama_probe_scores_tracking.jsonl")
    metadata = {
        "model": config.LLAMA.name,
        "probe_layer": config.LLAMA.probe_layer,
        "games": ["lottery", "ultimatum"],
        "method": "probe_steering",
        "probe_activation_definition": "s = w_hat^T h, signed projection of layer-48 hidden state onto unit-normalized probe direction",
        "subsets": _subsets_from_rows(rows),
        "random_seed": config.SEED["tracking_llama"],
    }
    return {
        "figure_name": "Probe scores tracking target behavior for lottery (left) and ultimatum (right) games on Llama-3.3-70B-Instruct (layer 48)",
        "experiment_type": "probe_activation_tracking",
        "metadata": metadata,
        "data": rows,
    }


def build_cross_object_llama(run_dir: Path) -> Dict:
    rows = read_jsonl(run_dir / "cross_object_llama" / "llama_cross_object.jsonl")
    metadata = {
        "model": config.LLAMA.name,
        "probe_layer": config.LLAMA.probe_layer,
        "construct": "divergent_creativity_probe",
        "objects": list(config.CROSS_OBJECT["objects"]),
        "metric": "accuracy",
        "method": "linear_probe_evaluation",
        "splits": ["in_distribution", "cross_object"],
        "random_seed": config.SEED["cross_object_llama"],
    }
    return {
        "figure_name": "Cross-object generalization analysis on Llama-3.3-70B-Instruct (layer 48)",
        "experiment_type": "probe_cross_object_generalization",
        "metadata": metadata,
        "data": rows,
    }


# ----- Qwen mirrors -----
def build_psychometric_qwen(run_dir: Path) -> Dict:
    lottery_cells = read_jsonl(run_dir / "psychometric_qwen" / "qwen_lottery_cells.jsonl")
    ultimatum_cells = read_jsonl(run_dir / "psychometric_qwen" / "qwen_ultimatum_cells.jsonl")
    lottery_per_agent = read_jsonl(run_dir / "psychometric_qwen" / "qwen_lottery_per_agent.jsonl")
    ultimatum_per_agent = read_jsonl(run_dir / "psychometric_qwen" / "qwen_ultimatum_per_agent.jsonl")

    data = [_ordered(c, LOTTERY_CELL_ORDER) for c in lottery_cells] + \
           [_ordered(c, ULTIMATUM_CELL_ORDER) for c in ultimatum_cells]
    per_agent = [_ordered(r, LOTTERY_PER_AGENT_ORDER) for r in lottery_per_agent] + \
                [_ordered(r, ULTIMATUM_PER_AGENT_ORDER) for r in ultimatum_per_agent]
    metadata = {
        "model": config.QWEN.name,
        "probe_layer": config.QWEN.probe_layer,
        "games": ["lottery", "ultimatum"],
        "n_agents_per_condition": config.LOTTERY["n_agents"],
        "lottery_reward_grid_tokens": config.LOTTERY["reward_grid_tokens_qwen"],
        "lottery_targets_tokens": config.LOTTERY["targets_qwen"],
        "ultimatum_offers_tokens": config.ULTIMATUM["offer_grid_tokens_qwen"],
        "ultimatum_targets_tokens": config.ULTIMATUM["targets_qwen"],
        "lottery_logistic_slope": config.LOTTERY["logistic_slope_qwen"],
        "ultimatum_logistic_slope": config.ULTIMATUM["logistic_slope_qwen"],
        "method": "probe_steering",
        "random_seed": config.SEED["psychometric_qwen"],
    }
    return {
        "figure_name": "Psychometric curves across calibrated lambda values for lottery and ultimatum games using probe-based steering on Qwen-2-7B-Instruct (layer 17)",
        "experiment_type": "probe_steering_psychometric_curves",
        "metadata": metadata,
        "data": data,
        "per_agent_rows": per_agent,
        "row_count": len(per_agent),
    }


def build_brick_capability_qwen(run_dir: Path) -> Dict:
    rows = read_jsonl(run_dir / "qwen" / "capability_llama" / "llama_brick_per_agent.jsonl")
    data = [_ordered(r, CAPABILITY_ORDER) for r in rows]
    lam, achieved, yerr = _capability_summary(rows)
    metadata = {
        "model": config.QWEN.name,
        "judge": config.CAPABILITY["judge_name"],
        "probe_layer": config.QWEN.probe_layer,
        "task": "divergent_creativity",
        "object": "brick",
        "targets": config.CAPABILITY["targets"],
        "scoring_dimensions": list(config.CAPABILITY["scoring_dimensions"]),
        "score_range": list(config.CAPABILITY["score_range"]),
        "method": "probe_steering",
        "lambda_per_target": lam,
        "achieved_per_target": achieved,
        "yerr_per_target": yerr,
        "n_agents_per_point": config.CAPABILITY["n_agents"],
        "random_seed": config.SEED["capability_qwen"],
        "yerr_definition": f"Standard error of the mean across {config.CAPABILITY['n_agents']} per-agent creativity scores (SEM = SD / sqrt({config.CAPABILITY['n_agents']}))",
    }
    return {
        "figure_name": "Achieved creativity scores as a function of target scores for the brick prompt using probe-based steering on Qwen-2-7B-Instruct (layer 17)",
        "experiment_type": "probe_steering_capability_target_vs_achieved",
        "metadata": metadata,
        "data": data,
    }


def build_four_objects_qwen(run_dir: Path) -> Dict:
    rows = read_jsonl(run_dir / "qwen" / "four_objects_llama" / "llama_four_objects_per_agent.jsonl")
    data = [_ordered(r, CAPABILITY_ORDER) for r in rows]
    by_obj = defaultdict(list)
    for r in rows:
        by_obj[r["object"]].append(r)

    # This figure carries TWO summary blocks with different shapes:
    #   * summary_per_object_computed_from_original_records:
    #         {obj: {"achieved_per_target": {target: mean}}}  -- recomputed straight
    #         from the per-agent records (achieved only, full precision).
    #   * summary_per_object:
    #         {obj: {"achieved": [...], "yerr": [...], "lambdas": [...]}}  -- list-form
    #         plotting summary ordered by ascending target.
    # Both blocks are computed from the same single source (these per-agent rows).
    records_summary: Dict[str, Dict] = {}
    plot_summary: Dict[str, Dict] = {}
    for obj in config.CAPABILITY["objects"]:
        obj_rows = by_obj.get(obj)
        if not obj_rows:
            continue
        lam, achieved, yerr = _capability_summary(obj_rows)
        targets = sorted(int(t) for t in achieved)

        achieved_full = defaultdict(list)
        for r in obj_rows:
            if r.get("creativity_score") is not None:
                achieved_full[int(r["target_creativity_score"])].append(r["creativity_score"])
        records_summary[obj] = {
            "achieved_per_target": {
                str(t): round(float(np.mean(achieved_full[t])), 6) for t in targets
            }
        }

        plot_summary[obj] = {
            "achieved": [achieved[str(t)] for t in targets],
            "yerr": [yerr[str(t)] for t in targets],
            "lambdas": [lam[str(t)] for t in targets],
        }
    metadata = {
        "model": config.QWEN.name,
        "judge": config.CAPABILITY["judge_name"],
        "probe_layer": config.QWEN.probe_layer,
        "task": "divergent_creativity",
        "objects": list(config.CAPABILITY["objects"]),
        "targets": config.CAPABILITY["targets"],
        "scoring_dimensions": list(config.CAPABILITY["scoring_dimensions"]),
        "score_range": list(config.CAPABILITY["score_range"]),
        "method": "probe_steering",
        "summary_per_object_computed_from_original_records": records_summary,
        "n_agents_per_point": config.CAPABILITY["n_agents"],
        "summary_per_object": plot_summary,
        "yerr_definition": f"Standard error of the mean across {config.CAPABILITY['n_agents']} per-agent creativity scores (SEM = SD / sqrt({config.CAPABILITY['n_agents']}))",
    }
    return {
        "figure_name": "Achieved creativity scores across all four object prompts (brick, stapler, paperclip, bowl) using probe-based steering on Qwen-2-7B-Instruct (layer 17)",
        "experiment_type": "probe_steering_capability_target_vs_achieved",
        "metadata": metadata,
        "data": data,
    }


def build_dose_response_qwen(run_dir: Path) -> Dict:
    rows = read_jsonl(run_dir / "dose_response_qwen" / "qwen_dose_response.jsonl")
    metadata = {
        "model": config.QWEN.name,
        "probe_layer": config.QWEN.probe_layer,
        "games": ["lottery", "ultimatum"],
        "lottery_target_range_tokens": [
            min(config.DOSE_RESPONSE["lottery_targets_qwen"]),
            max(config.DOSE_RESPONSE["lottery_targets_qwen"]),
        ],
        "ultimatum_target_range_tokens": [
            min(config.DOSE_RESPONSE["ultimatum_targets_qwen"]),
            max(config.DOSE_RESPONSE["ultimatum_targets_qwen"]),
        ],
        "lambda_range": [config.CALIBRATION["lambda_min"], config.CALIBRATION["lambda_max"]],
        "method": "probe_steering",
        "calibration_procedure": "coarse_grid_then_bracketed_binary_search",
        "random_seed": config.SEED["dose_response_qwen"],
    }
    return {
        "figure_name": "Dose-response showing lambda required to reach target switching points for lottery and ultimatum games on Qwen-2-7B-Instruct (layer 17)",
        "experiment_type": "probe_steering_dose_response_calibration",
        "metadata": metadata,
        "data": rows,
    }


def build_probe_tracking_qwen(run_dir: Path) -> Dict:
    rows = read_jsonl(run_dir / "probe_tracking_qwen" / "qwen_probe_scores_tracking.jsonl")
    metadata = {
        "model": config.QWEN.name,
        "probe_layer": config.QWEN.probe_layer,
        "games": ["lottery", "ultimatum"],
        # Derive subsets from the rows actually present (canonical order), exactly
        # like the Llama tracking builder. Both games are logged here, so this
        # yields all five subsets (all/risky/safe + accept/reject) rather than the
        # lottery-only triple a hardcoded list would give.
        "subsets": _subsets_from_rows(rows),
        "method": "probe_steering",
        "probe_activation_definition": "s = w_hat^T h, signed projection of layer-17 hidden state onto unit-normalized probe direction",
        "random_seed": config.SEED["tracking_qwen"],
    }
    return {
        "figure_name": "Probe activation score distributions for risky versus safe choices across different steering strengths on Qwen-2-7B-Instruct (layer 17)",
        "experiment_type": "probe_activation_tracking",
        "metadata": metadata,
        "data": rows,
    }


def build_cross_object_qwen(run_dir: Path) -> Dict:
    rows = read_jsonl(run_dir / "qwen" / "cross_object_llama" / "qwen_cross_object.jsonl")
    if not rows:
        # Fallback: the cross-object runner names the file by model prefix.
        rows = read_jsonl(run_dir / "qwen" / "cross_object_llama" / "llama_cross_object.jsonl")
    metadata = {
        "model": config.QWEN.name,
        "probe_layer": config.QWEN.probe_layer,
        "construct": "divergent_creativity_probe",
        "objects": list(config.CROSS_OBJECT["objects"]),
        "metric": "accuracy",
        "method": "linear_probe_evaluation",
        "splits": ["in_distribution", "cross_object"],
        "random_seed": config.SEED["cross_object_qwen"],
    }
    return {
        "figure_name": "Cross-object generalization analysis on Qwen-2-7B-Instruct (layer 17)",
        "experiment_type": "probe_cross_object_generalization",
        "metadata": metadata,
        "data": rows,
    }


# =========================================================================
# Top-level
# =========================================================================
FIGURE_BUILDERS = {
    "figure_7_psychometric_curves_llama": build_psychometric_llama,
    "figure_9_capability_brick_target_vs_achieved": build_capability_brick_llama,
    "figure_9_capability_stapler_product_innovation_target_vs_achieved": build_capability_stapler_llama,
    "figure_10_capability_four_objects_target_vs_achieved": build_four_objects_llama,
    "figure_11_dose_response_lottery_ultimatum_llama": build_dose_response_llama,
    "figure_12_probe_scores_track_target_lottery_ultimatum_llama": build_probe_tracking_llama,
    "figure_13_cross_object_generalization_llama": build_cross_object_llama,
    "figure_14_psychometric_curves_qwen": build_psychometric_qwen,
    "figure_15_dose_response_lottery_ultimatum_qwen": build_dose_response_qwen,
    "figure_16_probe_scores_track_target_qwen": build_probe_tracking_qwen,
    "figure_17_capability_brick_target_vs_achieved_qwen": build_brick_capability_qwen,
    "figure_18_capability_four_objects_target_vs_achieved_qwen": build_four_objects_qwen,
    "figure_19_cross_object_generalization_qwen": build_cross_object_qwen,
}


def assert_invariants(top: Dict) -> None:
    """Verify the output invariants documented in the README."""
    assert set(top.keys()) == set(FIGURE_BUILDERS.keys()), (
        f"Top-level keys mismatch. Got: {sorted(top.keys())}"
    )
    # Capability rows: creativity_score == mean(scores).
    for k in [
        "figure_9_capability_brick_target_vs_achieved",
        "figure_9_capability_stapler_product_innovation_target_vs_achieved",
        "figure_10_capability_four_objects_target_vs_achieved",
        "figure_17_capability_brick_target_vs_achieved_qwen",
        "figure_18_capability_four_objects_target_vs_achieved_qwen",
    ]:
        for r in top[k]["data"]:
            if r.get("scores") is None or r.get("creativity_score") is None:
                continue
            mean = sum(r["scores"].values()) / 4.0
            assert abs(mean - r["creativity_score"]) < 1e-6, (
                f"{k}: creativity_score mismatch ({mean} vs {r['creativity_score']})"
            )
    # Psychometric figures: per-cell n_choosing_risky / n_accepting matches per_agent expansion
    for k in [
        "figure_7_psychometric_curves_llama",
        "figure_14_psychometric_curves_qwen",
    ]:
        fig = top[k]
        per_agent = fig["per_agent_rows"]
        lot = defaultdict(int)
        ult = defaultdict(int)
        for r in per_agent:
            if r["game"] == "lottery":
                key = (r["target_switching_point_tokens"], r["risky_reward_tokens"])
                lot[key] += int(r["parsed_choice"] == 1)
            else:
                key = (r["target_switching_point_tokens"], r["offer_amount_tokens"])
                ult[key] += int(r["parsed_choice"] == 1)
        for cell in fig["data"]:
            if cell["game"] == "lottery":
                key = (cell["target_switching_point_tokens"], cell["risky_reward_tokens"])
                assert cell["n_choosing_risky"] == lot[key], f"{k} lottery mismatch at {key}"
            else:
                key = (cell["target_switching_point_tokens"], cell["offer_amount_tokens"])
                assert cell["n_accepting"] == ult[key], f"{k} ultimatum mismatch at {key}"
    # Probe-tracking figures: metadata.subsets must equal the subsets actually
    # present in the data (canonical order). Both tracking builders derive this
    # the same way, so Llama and Qwen stay consistent and the metadata can never
    # drift from the rows it describes.
    for k in [
        "figure_12_probe_scores_track_target_lottery_ultimatum_llama",
        "figure_16_probe_scores_track_target_qwen",
    ]:
        fig = top[k]
        expected = _subsets_from_rows(fig["data"])
        assert fig["metadata"]["subsets"] == expected, (
            f"{k}: metadata.subsets {fig['metadata']['subsets']} != subsets present in data {expected}"
        )


def build_all(llama_run: Path, qwen_run: Path, out_path: Path) -> Dict:
    """Combine Llama + Qwen run directories into one canonical JSON."""
    top: Dict[str, Dict] = {}
    for key, builder in FIGURE_BUILDERS.items():
        run_dir = qwen_run if "_qwen" in key else llama_run
        top[key] = builder(run_dir)
    assert_invariants(top)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(top, f, indent=2)
    return top


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--llama-run", required=True, type=Path)
    parser.add_argument("--qwen-run", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    top = build_all(args.llama_run, args.qwen_run, args.out)
    print(f"Wrote {args.out}: {len(top)} figures, "
          f"{sum(len(v['data']) for v in top.values())} total data rows")
