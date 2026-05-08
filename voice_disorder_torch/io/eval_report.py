from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def eval_json_sanitize(obj: Any) -> Any:
    """Make evaluation trees JSON-safe (numpy, inf, Path, sklearn report numbers)."""
    if obj is None or isinstance(obj, str):
        return obj
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        x = float(obj)
        return x if math.isfinite(x) else None
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): eval_json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [eval_json_sanitize(x) for x in obj]
    if isinstance(obj, np.ndarray):
        return eval_json_sanitize(obj.tolist())
    if isinstance(obj, np.generic):
        return eval_json_sanitize(obj.item())
    return str(obj)


def save_eval_json(path: Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(eval_json_sanitize(payload), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
