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

