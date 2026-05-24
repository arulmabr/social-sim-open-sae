#!/usr/bin/env python3
"""Conservative secret scan for the public export."""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SKIP_PARTS = {".git", "__pycache__", ".pytest_cache"}
PATTERNS = {
    "openai_api_key": re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_-]{20,}\b"),
    "goodfire_api_key": re.compile(r"\bsk-goodfire-[A-Za-z0-9_-]{20,}\b"),
    "huggingface_token": re.compile(r"\bhf_[A-Za-z0-9]{20,}\b"),
    "assigned_openai_key": re.compile(r"OPENAI_API_KEY\s*=\s*['\"][^'\"]+['\"]"),
    "assigned_hf_token": re.compile(r"HF_TOKEN\s*=\s*['\"][^'\"]+['\"]"),
    "assigned_goodfire_key": re.compile(r"GOODFIRE_API_KEY\s*=\s*['\"][^'\"]+['\"]"),
}


def iter_text_files() -> list[Path]:
    paths: list[Path] = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or any(part in SKIP_PARTS for part in path.parts):
            continue
        try:
            path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        paths.append(path)
    return paths


def main() -> None:
    hits: list[str] = []
    for path in iter_text_files():
        rel = path.relative_to(ROOT)
        text = path.read_text(encoding="utf-8")
        for name, pattern in PATTERNS.items():
            for match in pattern.finditer(text):
                line = text.count("\n", 0, match.start()) + 1
                hits.append(f"{rel}:{line}:{name}")
    if hits:
        raise AssertionError("Potential secrets found:\n" + "\n".join(hits[:50]))
    print("secret scan passed")


if __name__ == "__main__":
    main()
