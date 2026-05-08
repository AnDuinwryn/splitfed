"""Make local source-tree script execution work without `pip install -e`."""

from __future__ import annotations

import sys
from pathlib import Path


def ensure_project_package_on_path() -> Path:
    for anc in Path(__file__).resolve().parents:
        package_root = anc / "voice_disorder_torch"
        if (package_root / "config.py").is_file():
            s = str(anc)
            if s not in sys.path:
                sys.path.insert(0, s)
            return anc
    raise RuntimeError("Could not find local voice_disorder_torch package root.")


def find_project_root() -> Path:
    """Find the standalone project root from a split-learning script path."""

    for anc in Path(__file__).resolve().parents:
        if (anc / "pyproject.toml").is_file() and (anc / "voice_disorder_torch" / "config.py").is_file():
            return anc
    raise RuntimeError("Could not find standalone project root.")
