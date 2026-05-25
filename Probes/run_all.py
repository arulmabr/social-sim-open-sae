"""Top-level orchestrator: run every figure for one model.

Usage:
    python -m probe_pipeline_final.run_all --model llama --outdir runs/llama_final
    python -m probe_pipeline_final.run_all --model qwen  --outdir runs/qwen_final

Then aggregate with:
    python -m probe_pipeline_final.aggregator \
        --llama-run runs/llama_final --qwen-run runs/qwen_final \
        --out probe_results_final.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

from . import config
from .experiments import (
    psychometric_llama,
    capability_llama,
    four_objects_llama,
    dose_response_llama,
    probe_tracking_llama,
    cross_object_llama,
    psychometric_qwen,
    dose_response_qwen,
    probe_tracking_qwen,
    brick_capability_qwen,
    four_objects_qwen,
    cross_object_qwen,
)
from .experiments._common import build_preference_probe
from .models import load


def run_llama(outdir: Path) -> Dict:
    outdir.mkdir(parents=True, exist_ok=True)
    model = load("llama")
    summary: Dict = {}

    # Train the preference probes once; reuse across figures 7/11/12.
    lottery_probe = build_preference_probe(model, game="lottery")
    ultimatum_probe = build_preference_probe(model, game="ultimatum")
    lottery_probe.save(outdir / "lottery_probe.pkl")
    ultimatum_probe.save(outdir / "ultimatum_probe.pkl")

    summary["psychometric"] = psychometric_llama.run(model, outdir, lottery_probe, ultimatum_probe)
    summary["dose_response"] = dose_response_llama.run(model, outdir, lottery_probe, ultimatum_probe)
    summary["probe_tracking"] = probe_tracking_llama.run(model, outdir, lottery_probe, ultimatum_probe)

    summary["capability_brick"] = capability_llama.run_brick(model, outdir)
    summary["capability_stapler"] = capability_llama.run_stapler_product_innovation(model, outdir)
    summary["four_objects"] = four_objects_llama.run(model, outdir)
    summary["cross_object"] = cross_object_llama.run(model, outdir)

    with open(outdir / "run_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    return summary


def run_qwen(outdir: Path) -> Dict:
    outdir.mkdir(parents=True, exist_ok=True)
    model = load("qwen")
    summary: Dict = {}

    lottery_probe = build_preference_probe(model, game="lottery")
    ultimatum_probe = build_preference_probe(model, game="ultimatum")
    lottery_probe.save(outdir / "lottery_probe.pkl")
    ultimatum_probe.save(outdir / "ultimatum_probe.pkl")

    summary["psychometric"] = psychometric_qwen.run(model, outdir, lottery_probe, ultimatum_probe)
    summary["dose_response"] = dose_response_qwen.run(model, outdir, lottery_probe, ultimatum_probe)
    summary["probe_tracking"] = probe_tracking_qwen.run(model, outdir, lottery_probe, ultimatum_probe)

    summary["brick_capability"] = brick_capability_qwen.run(model, outdir)
    summary["four_objects"] = four_objects_qwen.run(model, outdir)
    summary["cross_object"] = cross_object_qwen.run(model, outdir)

    with open(outdir / "run_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["llama", "qwen"], required=True)
    parser.add_argument("--outdir", required=True, type=Path)
    args = parser.parse_args()
    if args.model == "llama":
        run_llama(args.outdir)
    else:
        run_qwen(args.outdir)


if __name__ == "__main__":
    main()
