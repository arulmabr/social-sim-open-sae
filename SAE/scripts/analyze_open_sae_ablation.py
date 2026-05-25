"""Analyze Open-SAE token-pooling ablations from saved output folders.

This script is intentionally lightweight: it does not load the base model or
SAE. It compares already-produced Open-SAE output folders and writes a compact
audit table, report, and diagnostic plot that make token-pooling artifacts
visible.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class OutputSpec:
    """One Open-SAE output folder to include in the ablation comparison."""

    name: str
    path: Path
    expected_units: int | None = None
    expected_top_rows: int | None = None


def parse_output_spec(raw: str) -> OutputSpec:
    """Parse NAME=PATH[:EXPECTED_UNITS:EXPECTED_ROWS]."""

    if "=" not in raw:
        raise argparse.ArgumentTypeError(
            "Output specs must have form NAME=PATH or NAME=PATH:EXPECTED_UNITS:EXPECTED_ROWS"
        )
    name, rest = raw.split("=", 1)
    parts = rest.rsplit(":", 2)
    if len(parts) == 3 and parts[1].isdigit() and parts[2].isdigit():
        path = Path(parts[0])
        expected_units = int(parts[1])
        expected_top_rows = int(parts[2])
    else:
        path = Path(rest)
        expected_units = None
        expected_top_rows = None
    if not name:
        raise argparse.ArgumentTypeError("Output spec name cannot be empty")
    return OutputSpec(name=name, path=path, expected_units=expected_units, expected_top_rows=expected_top_rows)


def import_pandas() -> Any:
    import pandas as pd

    return pd


def import_matplotlib() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize_output(spec: OutputSpec) -> dict[str, Any]:
    pd = import_pandas()
    metadata = read_json(spec.path / "open_sae_metadata.json")
    activations_path = spec.path / "open_sae_feature_activations.csv"
    if not activations_path.exists():
        return {
            "ablation": spec.name,
            "path": str(spec.path),
            "status": "missing_open_sae_feature_activations_csv",
        }

    activations = pd.read_csv(activations_path)
    rank1 = activations[activations["feature_rank"] == 1].copy()
    has_special_column = "max_token_is_special_or_control" in rank1.columns

    top_token = ""
    top_feature = ""
    if "max_token_text" in rank1.columns and not rank1.empty:
        top_token = str(rank1["max_token_text"].mode(dropna=False).iloc[0])
    if "feature_index" in rank1.columns and not rank1.empty:
        top_feature = str(rank1["feature_index"].mode(dropna=False).iloc[0])

    processed_units = metadata.get("processed_response_task_units")
    if processed_units is None:
        processed_units = len(rank1)

    actual_top_rows = metadata.get("actual_top_feature_rows")
    if actual_top_rows is None:
        actual_top_rows = len(activations)

    expected_units = spec.expected_units
    if expected_units is None:
        expected_units = processed_units
    expected_top_rows = spec.expected_top_rows
    if expected_top_rows is None:
        expected_top_rows = metadata.get("expected_top_feature_rows")

    special_hits = metadata.get("special_or_control_token_topk_hits")
    if special_hits is None and has_special_column:
        special_hits = int(activations["max_token_is_special_or_control"].fillna(False).sum())

    rank1_special_hits = None
    if has_special_column:
        rank1_special_hits = int(rank1["max_token_is_special_or_control"].fillna(False).sum())
    elif "max_token_text" in rank1.columns:
        rank1_special_hits = int(rank1["max_token_text"].astype(str).str.startswith("<|").sum())

    artifact_flag = False
    artifact_reason = ""
    if len(rank1) > 0:
        rank1_std = float(rank1["activation"].std(ddof=1)) if len(rank1) > 1 else 0.0
        rank1_max_activation_std = (
            float(rank1["max_activation"].std(ddof=1))
            if "max_activation" in rank1.columns and len(rank1) > 1
            else None
        )
        rank1_unique_features = int(rank1["feature_index"].nunique()) if "feature_index" in rank1.columns else 0
        rank1_unique_tokens = int(rank1["max_token_text"].nunique()) if "max_token_text" in rank1.columns else 0
        if rank1_std == 0.0 and rank1_unique_features == 1 and rank1_unique_tokens == 1:
            artifact_flag = True
            artifact_reason = "rank1 activations collapse to one feature/token with zero variance"
        if top_token.startswith("<|"):
            artifact_flag = True
            artifact_reason = "rank1 mode token is a chat-template/control token"
    else:
        rank1_std = 0.0
        rank1_max_activation_std = None
        rank1_unique_features = 0
        rank1_unique_tokens = 0
        artifact_flag = True
        artifact_reason = "no rank1 rows found"

    feature_aggregation = metadata.get("feature_aggregation")
    if feature_aggregation is None and metadata.get("activation_scope"):
        feature_aggregation = "max_legacy"

    return {
        "ablation": spec.name,
        "path": str(spec.path),
        "status": "ok",
        "dataset_kind": metadata.get("dataset_kind"),
        "activation_scope": metadata.get("activation_scope"),
        "feature_aggregation": feature_aggregation,
        "activation_threshold": metadata.get("activation_threshold"),
        "include_system_message": metadata.get("include_system_message"),
        "processed_response_task_units": processed_units,
        "expected_response_task_units": expected_units,
        "unit_count_ok": processed_units == expected_units,
        "top_feature_rows": actual_top_rows,
        "expected_top_feature_rows": expected_top_rows,
        "top_feature_row_count_ok": actual_top_rows == expected_top_rows if expected_top_rows is not None else None,
        "special_or_control_token_topk_hits": special_hits,
        "rank1_special_or_control_token_hits": rank1_special_hits,
        "rank1_activation_std": rank1_std,
        "rank1_max_activation_std": rank1_max_activation_std,
        "rank1_activation_min": float(rank1["activation"].min()) if len(rank1) else None,
        "rank1_activation_max": float(rank1["activation"].max()) if len(rank1) else None,
        "rank1_unique_features": rank1_unique_features,
        "rank1_unique_tokens": rank1_unique_tokens,
        "rank1_mode_feature_index": top_feature,
        "rank1_mode_token_text": top_token,
        "artifact_flag": artifact_flag,
        "artifact_reason": artifact_reason,
        "plots": ";".join(str(item) for item in metadata.get("plots", [])),
    }


def write_report(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Open-SAE Ablation Analysis",
        "",
        "This report compares saved Open-SAE output folders without re-running the model.",
        "The main failure mode under test is max-pooling over chat-template/control tokens,",
        "which appears as rank-1 activations collapsing to one feature and one token.",
        "",
        "## Summary",
        "",
    ]
    for row in rows:
        lines.extend(
            [
                f"### {row['ablation']}",
                "",
                f"- Path: `{row['path']}`",
                f"- Status: `{row['status']}`",
                f"- Dataset: `{row.get('dataset_kind')}`",
                f"- Activation scope: `{row.get('activation_scope')}`",
                f"- Feature aggregation: `{row.get('feature_aggregation')}`",
                f"- Activation threshold: `{row.get('activation_threshold')}`",
                f"- Units: `{row.get('processed_response_task_units')}` / expected `{row.get('expected_response_task_units')}`",
                f"- Top-k rows: `{row.get('top_feature_rows')}` / expected `{row.get('expected_top_feature_rows')}`",
                f"- Special/control top-k hits: `{row.get('special_or_control_token_topk_hits')}`",
                f"- Rank-1 std: `{row.get('rank1_activation_std')}`",
                f"- Rank-1 max-activation std: `{row.get('rank1_max_activation_std')}`",
                f"- Rank-1 unique features: `{row.get('rank1_unique_features')}`",
                f"- Rank-1 unique tokens: `{row.get('rank1_unique_tokens')}`",
                f"- Rank-1 mode token: `{row.get('rank1_mode_token_text')}`",
                f"- Artifact flag: `{row.get('artifact_flag')}`",
                f"- Artifact reason: `{row.get('artifact_reason')}`",
                "",
            ]
        )

    lines.extend(
        [
            "## Interpretation",
            "",
            "A valid response-level decomposition should have zero special/control-token top-k hits",
            "and non-trivial variation in rank-1 feature and token identities. A collapsed rank-1",
            "distribution is evidence that the pooling scope is dominated by a template token rather",
            "than task content or generated response content.",
            "",
            "## Completed Matrix",
            "",
            "All supplied outputs were read from disk and summarized here. Treat any row with",
            "`artifact_flag=True` as a diagnostic/failure control, not as a paper-style feature",
            "decomposition.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_plot(path: Path, rows: list[dict[str, Any]]) -> None:
    plt = import_matplotlib()
    ok_rows = [row for row in rows if row["status"] == "ok"]
    if not ok_rows:
        return

    names = [row["ablation"] for row in ok_rows]
    std_values = [float(row["rank1_activation_std"]) for row in ok_rows]
    unique_features = [int(row["rank1_unique_features"]) for row in ok_rows]
    unique_tokens = [int(row["rank1_unique_tokens"]) for row in ok_rows]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, values, title, ylabel in [
        (axes[0], std_values, "Rank-1 Activation Spread", "Std dev"),
        (axes[1], unique_features, "Rank-1 Feature Diversity", "Unique features"),
        (axes[2], unique_tokens, "Rank-1 Token Diversity", "Unique tokens"),
    ]:
        colors = ["#B85450" if row["artifact_flag"] else "#4C78A8" for row in ok_rows]
        ax.bar(names, values, color=colors)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for ablation summary outputs.",
    )
    parser.add_argument(
        "--output",
        dest="outputs",
        action="append",
        type=parse_output_spec,
        required=True,
        help="Ablation output spec, NAME=PATH or NAME=PATH:EXPECTED_UNITS:EXPECTED_ROWS.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = [summarize_output(spec) for spec in args.outputs]
    write_csv(args.output_dir / "open_sae_ablation_summary.csv", rows)
    write_report(args.output_dir / "open_sae_ablation_report.md", rows)
    write_plot(args.output_dir / "open_sae_ablation_rank1_diagnostics.png", rows)
    print(
        json.dumps(
            {
                "status": "complete",
                "output_dir": str(args.output_dir),
                "ablations": len(rows),
                "artifact_flags": sum(bool(row.get("artifact_flag")) for row in rows),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
