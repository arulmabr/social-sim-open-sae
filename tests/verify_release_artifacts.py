#!/usr/bin/env python3
"""Lightweight checks for the public release artifact set."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def require(path: str) -> Path:
    full = ROOT / path
    if not full.exists():
        raise AssertionError(f"Missing required artifact: {path}")
    return full


def check_creativity_torrance() -> None:
    evals = pd.read_csv(require("data/processed/creativity/torrance_gpt5_eval/torrance_gpt_evals.csv"))
    summary = pd.read_csv(require("data/processed/creativity/torrance_gpt5_eval/torrance_eval_summary.csv"))
    if len(evals) != 320:
        raise AssertionError(f"Expected 320 Torrance rows, found {len(evals)}")
    if len(summary) != 8:
        raise AssertionError(f"Expected 8 Torrance summary rows, found {len(summary)}")
    score_cols = ["fluency", "flexibility", "originality", "elaboration"]
    for col in score_cols:
        if not evals[col].between(1, 10).all():
            raise AssertionError(f"Score column out of range: {col}")
    expected = evals[score_cols].mean(axis=1)
    if not (abs(evals["final_score"] - expected) < 1e-12).all():
        raise AssertionError("final_score is not the mean of the four Torrance dimensions")


def check_creativity_open_sae() -> None:
    base = "data/processed/creativity/open_sae_response_only_frequency"
    acts = pd.read_csv(require(f"{base}/open_sae_feature_activations.csv"))
    top = pd.read_csv(require(f"{base}/open_sae_condition_top_features.csv"))
    meta = json.loads(require(f"{base}/open_sae_metadata.json").read_text())
    if len(acts) != 3200:
        raise AssertionError(f"Expected 3,200 creativity Open-SAE rows, found {len(acts)}")
    if meta.get("processed_response_task_units") != 320:
        raise AssertionError("Creativity Open-SAE unit count mismatch")
    if meta.get("special_or_control_token_topk_hits") != 0:
        raise AssertionError("Creativity Open-SAE has special/control-token top-k hits")
    if top.groupby(["task", "condition"]).ngroups != 8:
        raise AssertionError("Creativity Open-SAE top-feature cells should be 8")


def check_safe_risky() -> None:
    base = "data/processed/games/safe_risky/open_sae_calibration"
    acts = pd.read_csv(require(f"{base}/open_sae_feature_activations.csv"))
    behavior = pd.read_csv(require(f"{base}/safe_risky_behavior_summary.csv"))
    meta = json.loads(require(f"{base}/open_sae_metadata.json").read_text())
    if len(acts) != 42000:
        raise AssertionError(f"Expected 42,000 safe-risk Open-SAE rows, found {len(acts)}")
    if meta.get("processed_response_task_units") != 4200:
        raise AssertionError("Safe-risk Open-SAE unit count mismatch")
    if meta.get("special_or_control_token_topk_hits") != 0:
        raise AssertionError("Safe-risk Open-SAE has special/control-token top-k hits")
    if len(behavior) != 105:
        raise AssertionError(f"Expected 105 safe-risk behavior rows, found {len(behavior)}")
    if int(behavior["comment_nonempty_count"].sum()) != int(behavior["total_responses"].sum()):
        raise AssertionError("Safe-risk comments are not complete")


def main() -> None:
    check_creativity_torrance()
    check_creativity_open_sae()
    check_safe_risky()
    print("release artifact verification passed")


if __name__ == "__main__":
    main()
