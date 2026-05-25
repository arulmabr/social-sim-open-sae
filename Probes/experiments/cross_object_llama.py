"""Cross-object generalization (Llama; reused for Qwen via cross_object_qwen).

For each test object:
- In-distribution: train + test on same object's labeled data, multiple seeds
- Cross-object: train on other 3 objects, test on held-out, multiple seeds

Output: accuracy mean / std / n_runs per (train_object, test_object) record.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from .. import config
from ..judge import score_response
from ..models import Wrapped
from ..probes import median_split, fit_and_score
from ..tasks import prompts
from ._common import write_jsonl


def _generate_capability_labeled_pool(
    model: Wrapped, task: str, obj: str, n_samples: int = 80,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate (activations, labels) for object `obj` using median-split judge scores."""
    prompt_text, _ = prompts.get_capability_prompt(task, obj)
    activations: List[np.ndarray] = []
    scores: List[float] = []
    for i in range(n_samples):
        # Mix baseline + persona-prefixed prompts for label variance.
        if i % 2 == 0:
            full_prompt = prompt_text
            seed = config.agent_seed(0, 0, i + 1)
        else:
            full_prompt = (
                "You are a highly creative and unconventional thinker. " + prompt_text
            )
            seed = config.agent_seed(0, 1, i + 1)
        gen = model.generate(
            full_prompt,
            max_new_tokens=config.DECODING_DEFAULTS["max_new_tokens_capability"],
            temperature=config.DECODING_DEFAULTS["temperature_capability"],
            top_p=config.DECODING_DEFAULTS["top_p"],
            seed=seed,
        )
        h = model.capture_activations(full_prompt, model.cfg.probe_layer)
        activations.append(h)
        s = score_response(gen.text, task=task, judge_name=config.CAPABILITY["judge_hf_id_or_provider"])
        scores.append(s["creativity_score"] if s else 5.0)
    X = np.stack(activations, axis=0)
    y = median_split(scores)
    return X, y


def run(model: Wrapped, outdir: Path) -> Dict:
    out = outdir / "cross_object_llama"
    out.mkdir(parents=True, exist_ok=True)
    # Cross-object seed is model-specific (Llama and Qwen splits differ).
    is_qwen = model.cfg is config.QWEN
    seed_base = (
        config.SEED["cross_object_qwen"] if is_qwen
        else config.SEED["cross_object_llama"]
    )
    # n_runs is derived per object: attempt a fixed seed budget, keep only the
    # seeds that produce a well-posed fit (see config.CROSS_OBJECT).
    max_attempts = config.CROSS_OBJECT["max_seed_attempts"]
    min_train_acc = config.CROSS_OBJECT["min_train_accuracy"]
    task = "divergent_creativity"
    pools: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for obj in config.CROSS_OBJECT["objects"]:
        pools[obj] = _generate_capability_labeled_pool(model, task, obj)

    rows: List[Dict] = []

    def _both_classes(y: np.ndarray) -> bool:
        return int(np.unique(y).size) >= 2

    for obj in config.CROSS_OBJECT["objects"]:
        # In-distribution: bootstrap-resample the object's pool and evaluate on
        # its out-of-bag rows. Keep only well-posed seeds (both classes present
        # in train and OOB, above-chance train fit). n_runs = surviving seeds.
        Xo, yo = pools[obj]
        n = len(yo)
        accs_in: List[float] = []
        rng = np.random.RandomState(seed_base)
        for attempt in range(max_attempts):
            boot = rng.randint(0, n, size=n)
            oob = np.setdiff1d(np.arange(n), np.unique(boot))
            if oob.size == 0 or not _both_classes(yo[boot]) or not _both_classes(yo[oob]):
                continue
            train_acc, test_acc = fit_and_score(
                Xo[boot], yo[boot], Xo[oob], yo[oob], random_state=attempt
            )
            if train_acc < min_train_acc:
                continue
            accs_in.append(test_acc)
        rows.append({
            "model": model.cfg.name,
            "probe_layer": model.cfg.probe_layer,
            "construct": "divergent_creativity_probe",
            "train_object": obj,
            "test_object": obj,
            "split_type": "in_distribution",
            "metric": "accuracy",
            "mean_score": round(float(np.mean(accs_in)), 4) if accs_in else None,
            "std_score": round(float(np.std(accs_in)), 6) if accs_in else None,
            "n_runs": int(len(accs_in)),
        })

        # Cross-object: train on union of other 3 objects, test on `obj`. The
        # train pool and test set are fixed, so we bootstrap-resample the train
        # pool per seed; same sanity filter, same derived n_runs.
        Xs = []
        ys = []
        for o in config.CROSS_OBJECT["objects"]:
            if o == obj:
                continue
            Xs.append(pools[o][0])
            ys.append(pools[o][1])
        Xtr = np.concatenate(Xs, axis=0)
        ytr = np.concatenate(ys, axis=0)
        m = len(ytr)
        accs_cross: List[float] = []
        rng_cross = np.random.RandomState(seed_base + 1)
        for attempt in range(max_attempts):
            boot = rng_cross.randint(0, m, size=m)
            if not _both_classes(ytr[boot]):
                continue
            train_acc, test_acc = fit_and_score(
                Xtr[boot], ytr[boot], Xo, yo, random_state=attempt
            )
            if train_acc < min_train_acc:
                continue
            accs_cross.append(test_acc)
        rows.append({
            "model": model.cfg.name,
            "probe_layer": model.cfg.probe_layer,
            "construct": "divergent_creativity_probe",
            "train_object": "others",
            "test_object": obj,
            "split_type": "cross_object",
            "metric": "accuracy",
            "mean_score": round(float(np.mean(accs_cross)), 4) if accs_cross else None,
            "std_score": round(float(np.std(accs_cross)), 6) if accs_cross else None,
            "n_runs": int(len(accs_cross)),
        })

    write_jsonl(rows, out / f"{model.cfg.name.lower().split('-')[0]}_cross_object.jsonl")
    return {"n_rows": len(rows)}
