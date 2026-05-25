"""Re-score existing capability outputs with multiple judges (parallel, resumable).

Reads `probe_results_final.json` (or any of the per-experiment capability
JSONLs produced by `run_all.py`) and adds per-judge scores for every
(response, judge) pair using a thread pool. Results are
written to a JSONL incrementally; if the script is killed mid-run, restarting
it skips (row_id, judge) pairs that already have a result on disk.

Outputs in the `--out` directory:
1. `multi_judge_scores.jsonl`   per (response, judge) row with 4 Torrance
                                sub-scores, mean creativity_score, response
                                length stats, optional blind-order index
2. `inter_rater_agreement.json` pairwise Spearman + mean pairwise rho
3. `length_regression.json`     per-judge slope of score on log(length_chars)
4. `errors.jsonl`               one line per failed (row, judge) call

Usage:
    python -m Probes.multi_judge_rescore \
        --in probe_results_final.json \
        --judges gpt-5 claude-sonnet-4-6 gemini-3.1-pro kimi-k2.6 deepseek-v4-pro \
        --out runs/multi_judge \
        --max-workers 24 \
        --blind --length-controlled
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import random
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from . import judge as judge_mod
from .judge import JUDGES


# =========================================================================
# Input loading
# =========================================================================
CAPABILITY_FIGURE_KEYS = (
    "figure_9_capability_brick_target_vs_achieved",
    "figure_9_capability_stapler_product_innovation_target_vs_achieved",
    "figure_10_capability_four_objects_target_vs_achieved",
    "figure_17_capability_brick_target_vs_achieved_qwen",
    "figure_18_capability_four_objects_target_vs_achieved_qwen",
)


def load_capability_rows(path: Path) -> List[Dict]:
    if path.suffix == ".json":
        with open(path) as f:
            data = json.load(f)
        rows: List[Dict] = []
        for k in CAPABILITY_FIGURE_KEYS:
            if k in data:
                for r in data[k]["data"]:
                    if r.get("response"):
                        rows.append({**r, "_source_figure": k})
        return rows
    elif path.suffix == ".jsonl":
        rows = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    else:
        raise ValueError(f"Unsupported input file type: {path}")


def row_id(r: Dict) -> str:
    """Stable identifier for a capability response row across runs."""
    parts = (
        r.get("_source_figure") or "",
        r.get("model") or "",
        r.get("task") or "",
        r.get("object") or "",
        str(r.get("target_creativity_score") or ""),
        r.get("agent_id") or "",
    )
    return "|".join(parts)


# =========================================================================
# Length stats
# =========================================================================
def response_length_stats(response_text: str) -> Dict[str, int]:
    return {
        "response_length_chars": len(response_text),
        "response_length_words": len(response_text.split()),
        "response_length_tokens_estimate": max(1, int(len(response_text) / 4)),
    }


# =========================================================================
# Parallel scoring loop with resumability
# =========================================================================
@dataclass
class WorkItem:
    row: Dict
    judge_name: str
    blind_order_idx: int


def _do_one(item: WorkItem, length_controlled: bool) -> Tuple[WorkItem, Optional[Dict], Optional[str]]:
    """Score one (row, judge) pair. Returns (item, output_row_or_None, error_or_None)."""
    r = item.row
    text = r.get("response", "")
    task = r.get("task")
    try:
        scored = judge_mod.score_response_with_judge(
            text, task, judge_name=item.judge_name, length_controlled=length_controlled,
        )
    except Exception as e:
        return item, None, f"{type(e).__name__}: {e}"
    sub = {k: scored[k] for k in ("fluency", "flexibility", "originality", "elaboration")} if scored else None
    out = {
        "row_id": row_id(r),
        "model": r.get("model"),
        "probe_layer": r.get("probe_layer"),
        "task": task,
        "object": r.get("object"),
        "prompt_id": r.get("prompt_id"),
        "target_creativity_score": r.get("target_creativity_score"),
        "lambda_calibrated": r.get("lambda_calibrated"),
        "agent_id": r.get("agent_id"),
        "_source_figure": r.get("_source_figure"),
        "judge_name": item.judge_name,
        "judge": JUDGES[item.judge_name].display_name,
        "judge_provider": JUDGES[item.judge_name].provider,
        "judge_model_id": JUDGES[item.judge_name].model_id,
        "scores": sub,
        "creativity_score": float(scored["creativity_score"]) if scored else None,
        **response_length_stats(text),
        "length_controlled_rubric": bool(length_controlled),
        "blind_order_idx": int(item.blind_order_idx),
        "original_judge": r.get("judge"),
        "original_creativity_score": r.get("creativity_score"),
    }
    return item, out, (None if scored is not None else "score_returned_none")


def _read_done_pairs(path: Path) -> set:
    done = set()
    if not path.exists():
        return done
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            done.add((r.get("row_id"), r.get("judge_name")))
    return done


def rescore_parallel(
    rows: List[Dict],
    judge_names: List[str],
    out_path: Path,
    errors_path: Path,
    max_workers: int = 16,
    length_controlled: bool = False,
    blind: bool = False,
    blind_seed: int = 0,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    errors_path.parent.mkdir(parents=True, exist_ok=True)

    # Determine blind order
    if blind:
        rng = random.Random(blind_seed)
        order = list(range(len(rows)))
        rng.shuffle(order)
        rows = [rows[i] for i in order]
        idx_map = {id(r): i for i, r in enumerate(rows)}
    else:
        idx_map = {id(r): -1 for r in rows}

    # Build the work queue, skipping pairs already on disk
    done = _read_done_pairs(out_path)
    work: List[WorkItem] = []
    for r in rows:
        rid = row_id(r)
        for jn in judge_names:
            if (rid, jn) in done:
                continue
            work.append(WorkItem(row=r, judge_name=jn, blind_order_idx=idx_map[id(r)]))

    total = len(work)
    already_done = len(rows) * len(judge_names) - total
    print(f"Total (row, judge) pairs: {len(rows) * len(judge_names)} "
          f"({already_done} already done, {total} to do)", file=sys.stderr)
    if total == 0:
        return

    write_lock = threading.Lock()
    err_lock = threading.Lock()
    completed = 0
    failed = 0
    t0 = time.time()

    with open(out_path, "a") as out_f, open(errors_path, "a") as err_f, \
         ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_do_one, item, length_controlled) for item in work]
        for fut in as_completed(futures):
            item, out_row, err = fut.result()
            if out_row is not None and out_row.get("creativity_score") is not None:
                with write_lock:
                    out_f.write(json.dumps(out_row) + "\n")
                    out_f.flush()
                completed += 1
            else:
                with err_lock:
                    err_f.write(json.dumps({
                        "row_id": row_id(item.row),
                        "judge_name": item.judge_name,
                        "error": err or "no_score",
                        "ts": time.time(),
                    }) + "\n")
                    err_f.flush()
                failed += 1
            n_done = completed + failed
            if n_done % 25 == 0 or n_done == total:
                elapsed = time.time() - t0
                rate = n_done / max(elapsed, 1e-6)
                eta_sec = (total - n_done) / max(rate, 1e-6)
                print(f"  [{n_done}/{total}] ok={completed} fail={failed} "
                      f"rate={rate:.1f}/s eta={eta_sec/60:.1f}min", file=sys.stderr)

    print(f"Done. ok={completed} fail={failed} elapsed={(time.time()-t0)/60:.1f}min",
          file=sys.stderr)


# =========================================================================
# Inter-rater agreement
# =========================================================================
def _spearman(xs: List[float], ys: List[float]) -> Optional[float]:
    if len(xs) != len(ys) or len(xs) < 3:
        return None
    def rank(arr):
        order = sorted(range(len(arr)), key=lambda i: arr[i])
        ranks = [0.0] * len(arr)
        i = 0
        while i < len(arr):
            j = i
            while j + 1 < len(arr) and arr[order[j + 1]] == arr[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                ranks[order[k]] = avg
            i = j + 1
        return ranks
    rx, ry = rank(xs), rank(ys)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    if den == 0:
        return None
    return num / den


def compute_inter_rater_agreement(scored_path: Path) -> Dict:
    rows = []
    with open(scored_path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    by_trial: Dict[Tuple, Dict[str, float]] = defaultdict(dict)
    for r in rows:
        if r.get("creativity_score") is None:
            continue
        key = (r["row_id"],)
        by_trial[key][r["judge_name"]] = float(r["creativity_score"])

    judges_seen = sorted({jn for js in by_trial.values() for jn in js})
    pairwise: Dict[str, Optional[float]] = {}
    pairwise_n: Dict[str, int] = {}
    for a, b in itertools.combinations(judges_seen, 2):
        xs, ys = [], []
        for js in by_trial.values():
            if a in js and b in js:
                xs.append(js[a])
                ys.append(js[b])
        rho = _spearman(xs, ys)
        pairwise[f"{a}__vs__{b}"] = rho
        pairwise_n[f"{a}__vs__{b}"] = len(xs)
    valid = [v for v in pairwise.values() if v is not None]
    mean_pairwise = sum(valid) / len(valid) if valid else None
    return {
        "judges_seen": judges_seen,
        "pairwise_spearman": pairwise,
        "pairwise_n": pairwise_n,
        "mean_pairwise_spearman": mean_pairwise,
        "n_trials": len(by_trial),
    }


# =========================================================================
# Length confound regression
# =========================================================================
def compute_length_regression(scored_path: Path) -> Dict:
    rows = []
    with open(scored_path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    by_judge: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
    for r in rows:
        if r.get("creativity_score") is None or r.get("response_length_chars", 0) <= 0:
            continue
        by_judge[r["judge_name"]].append((math.log(r["response_length_chars"]), float(r["creativity_score"])))
    out = {}
    for j, pts in by_judge.items():
        n = len(pts)
        if n < 3:
            out[j] = {"n": n, "slope": None, "intercept": None, "r_squared": None}
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        mx, my = sum(xs) / n, sum(ys) / n
        sxy = sum((x - mx) * (y - my) for x, y in pts)
        sxx = sum((x - mx) ** 2 for x in xs)
        syy = sum((y - my) ** 2 for y in ys)
        slope = sxy / sxx if sxx else 0.0
        intercept = my - slope * mx
        ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in pts)
        r2 = 1 - (ss_res / syy) if syy else None
        out[j] = {"n": n, "slope": slope, "intercept": intercept, "r_squared": r2}
    return out


# =========================================================================
# CLI
# =========================================================================
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="in_path", type=Path, required=True)
    parser.add_argument(
        "--judges", nargs="+",
        default=["gpt-5", "claude-sonnet-4-6", "gemini-3.1-pro", "kimi-k2.6", "deepseek-v4-pro"],
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-workers", type=int, default=16)
    parser.add_argument("--blind", action="store_true")
    parser.add_argument("--blind-seed", type=int, default=0)
    parser.add_argument("--length-controlled", action="store_true")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--skip-rescore", action="store_true",
                        help="Skip scoring; only compute agreement + length regression from existing output")
    args = parser.parse_args()

    for jn in args.judges:
        if jn not in JUDGES:
            raise SystemExit(f"Unknown judge {jn!r}. Known: {list(JUDGES.keys())}")

    args.out.mkdir(parents=True, exist_ok=True)
    scores_path = args.out / "multi_judge_scores.jsonl"
    errors_path = args.out / "errors.jsonl"

    if not args.skip_rescore:
        rows = load_capability_rows(args.in_path)
        if args.max_rows is not None:
            rows = rows[: args.max_rows]
        print(f"Loaded {len(rows)} capability rows from {args.in_path}", file=sys.stderr)
        rescore_parallel(
            rows, args.judges, scores_path, errors_path,
            max_workers=args.max_workers,
            length_controlled=args.length_controlled,
            blind=args.blind,
            blind_seed=args.blind_seed,
        )

    agreement = compute_inter_rater_agreement(scores_path)
    with open(args.out / "inter_rater_agreement.json", "w") as f:
        json.dump(agreement, f, indent=2)
    print(f"Mean pairwise Spearman: {agreement['mean_pairwise_spearman']}", file=sys.stderr)

    length_reg = compute_length_regression(scores_path)
    with open(args.out / "length_regression.json", "w") as f:
        json.dump(length_reg, f, indent=2)
    for j, row in length_reg.items():
        print(f"  length-reg {j}: slope={row['slope']!r} R^2={row['r_squared']!r}", file=sys.stderr)


if __name__ == "__main__":
    main()
