"""Four-object capability control on Qwen (reuses the Llama four-objects runner)."""
from __future__ import annotations

from pathlib import Path
from typing import Dict

from ..models import Wrapped
from .four_objects_llama import run as run_four_objects


def run(model: Wrapped, outdir: Path) -> Dict:
    return run_four_objects(model, outdir / "qwen")
