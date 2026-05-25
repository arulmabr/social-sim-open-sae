"""Cross-object generalization on Qwen (reuses the Llama cross-object runner)."""
from __future__ import annotations

from pathlib import Path
from typing import Dict

from ..models import Wrapped
from .cross_object_llama import run as run_cross


def run(model: Wrapped, outdir: Path) -> Dict:
    return run_cross(model, outdir / "qwen")
