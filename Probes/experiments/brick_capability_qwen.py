"""Brick capability control on Qwen (reuses the Llama capability runner)."""
from __future__ import annotations

from pathlib import Path
from typing import Dict

from ..models import Wrapped
from .capability_llama import run_brick


def run(model: Wrapped, outdir: Path) -> Dict:
    """Delegate to the brick capability runner; Qwen-specific seeds/layer come from model.cfg."""
    return run_brick(model, outdir / "qwen")
