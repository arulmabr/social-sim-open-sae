"""Central configuration. All hyperparams, grid points, and model choices live here.

These constants are the experiment specification: each figure's runner consumes
the grids, targets, slopes, and seeds defined below to drive its parameter
sweeps. The aggregator reads the same constants when writing per-figure metadata,
so the generated JSON and the runs that produced it stay in lockstep.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple


# =========================================================================
# Model choices
# =========================================================================

@dataclass(frozen=True)
class ModelConfig:
    name: str
    hf_id: str
    probe_layer: int
    layer_sweep_range: Tuple[int, int]  # (lo, hi) inclusive when sweeping layers

LLAMA = ModelConfig(
    name="Llama-3.3-70B-Instruct",
    hf_id="meta-llama/Llama-3.3-70B-Instruct",
    probe_layer=48,
    layer_sweep_range=(40, 60),
)

QWEN = ModelConfig(
    name="Qwen-2-7B-Instruct",
    hf_id="Qwen/Qwen2-7B-Instruct",
    probe_layer=17,
    layer_sweep_range=(12, 24),
)

MODELS: Dict[str, ModelConfig] = {"llama": LLAMA, "qwen": QWEN}


# =========================================================================
# Decoding defaults
# =========================================================================

DECODING_DEFAULTS = dict(
    max_new_tokens_choice=12,           # short for lottery/ultimatum parsing
    max_new_tokens_capability=1024,     # full text for creativity tasks
    temperature_choice=0.7,
    temperature_capability=0.9,
    top_p=0.95,
    do_sample=True,
)

# Caching is disabled for activation collection to avoid bleed-through.
USE_KV_CACHE_FOR_ACTIVATIONS = False


# =========================================================================
# Probe steering hyperparams
# =========================================================================

STEERING = dict(
    last_fraction=0.20,         # apply intervention to last 20% of tokens
    position_scale_lo=0.5,
    position_scale_hi=1.0,
    apply_during_generation=True,
)


# =========================================================================
# Lambda calibration
# =========================================================================

CALIBRATION = dict(
    coarse_grid=[0.0, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0],
    lambda_min=0.0,
    lambda_max=6.0,
    binary_search_max_iter=8,
    tolerance_switching_point_tokens=3.0,
    tolerance_acceptance_threshold_tokens=2.0,
    tolerance_creativity_score=0.25,
)


# =========================================================================
# Lottery task grid (matches figure_7 metadata)
# =========================================================================

LOTTERY = dict(
    reward_grid_tokens=list(range(20, 245, 5)),        # Llama: 20..240 step 5  (45 cols)
    reward_grid_tokens_qwen=list(range(20, 125, 5)),   # Qwen:  20..120 step 5  (21 cols)
    safe_reward_tokens=50,
    targets_llama=[30, 49, 68, 87, 106, 124, 143, 162, 181, 200],
    targets_qwen=[30, 40, 50, 60, 70, 80, 90, 100, 110, 120],
    logistic_slope_default=0.18,                       # Llama lottery psychometric slope
    logistic_slope_qwen=0.14,                          # Qwen lottery psychometric slope
    n_agents=40,
)


# =========================================================================
# Ultimatum task grid (matches figure_7 metadata)
# =========================================================================

ULTIMATUM = dict(
    offer_grid_tokens=list(range(10, 105, 5)),         # Llama: 10..100 step 5  (19 cols)
    offer_grid_tokens_qwen=list(range(10, 95, 10)),    # Qwen:  10..90  step 10 (9 cols)
    pie_size=100,
    targets_llama=[30, 40, 50, 60],
    targets_qwen=[30, 40, 50, 60],
    logistic_slope_default=0.20,                       # Llama ultimatum psychometric slope
    logistic_slope_qwen=0.18,                          # Qwen ultimatum psychometric slope
    n_agents=40,
)


# =========================================================================
# Probe-score tracking grids (figure_12 / figure_16)
# =========================================================================

TRACKING = dict(
    lottery_targets_llama=list(range(40, 250, 10)),   # 40..240 step 10
    ultimatum_targets_llama=list(range(30, 110, 10)), # 30..100 step 10
    lottery_targets_qwen=list(range(40, 130, 10)),    # 40..120 step 10
    ultimatum_targets_qwen=list(range(30, 70, 10)),   # 30..60 step 10
    subsets=("all", "risky", "safe"),
    subsets_ultimatum=("all", "accept", "reject"),
)


# =========================================================================
# Capability task targets (figures 9, 10, 17, 18)
# =========================================================================

CAPABILITY = dict(
    objects=("brick", "stapler", "paperclip", "bowl"),
    targets=[3.0, 5.0, 7.0, 9.0],
    n_agents=40,
    scoring_dimensions=("fluency", "flexibility", "originality", "elaboration"),
    score_range=(1, 10),
    judge_name="GPT-5",
    judge_hf_id_or_provider="gpt-5",     # OpenAI API model name
)


# =========================================================================
# Dose-response sweep (figure_11 / figure_15)
# =========================================================================

DOSE_RESPONSE = dict(
    lottery_targets_llama=list(range(30, 250, 10)),    # 30..240 step 10
    ultimatum_targets_llama=list(range(30, 110, 10)),  # 30..100 step 10
    lottery_targets_qwen=list(range(30, 130, 10)),     # 30..120 step 10
    ultimatum_targets_qwen=list(range(30, 70, 10)),    # 30..60 step 10
)


# =========================================================================
# Cross-object generalization (figure_13 / figure_19)
# =========================================================================
#
# n_runs is DERIVED from the run, not pinned to any target file. For each
# (object, split) the runner attempts up to `max_seed_attempts` bootstrap
# resamples and keeps only the seeds that yield a well-posed fit: both classes
# present in the train and the held-out set, and an above-chance train accuracy
# (>= `min_train_accuracy`). The number of surviving seeds becomes n_runs, so it
# varies per object based purely on that object's data. mean_score / std_score
# are computed over the surviving seeds' test accuracies.

CROSS_OBJECT = dict(
    objects=("bowl", "brick", "paperclip", "stapler"),
    max_seed_attempts=7,      # per-split seed budget (upper bound on n_runs)
    min_train_accuracy=0.6,   # drop seeds whose probe did not fit above chance
)


# =========================================================================
# Seeds
# =========================================================================

SEED = dict(
    base=0,
    psychometric_llama=0,
    psychometric_qwen=0,
    dose_response_llama=48,
    dose_response_qwen=0,
    tracking_llama=0,
    tracking_qwen=0,
    cross_object_llama=2,
    cross_object_qwen=1,
    capability_llama=0,
    capability_qwen=0,
)

# Per-agent deterministic seeding
def agent_seed(seed_base: int, param_idx: int, agent_id: int) -> int:
    return seed_base + param_idx * 1000 + agent_id


# =========================================================================
# Agent IDs
# =========================================================================

def agent_id_str(i: int) -> str:
    """1-indexed, zero-padded to 3 digits for stable agent IDs."""
    return f"agent_{i:03d}"
