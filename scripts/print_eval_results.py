#!/usr/bin/env python3
"""Load evaluation JSON (--results-json from evaluate.py) and print a CLI summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from voice_disorder_torch.ui.eval_cli import print_eval

def main() -> None:
    p = argparse.ArgumentParser(description="Print summary from evaluate.py results JSON.")
    p.add_argument("json_path", type=Path, help="Path to eval JSON (e.g. saved_models/eval_results.json).")
    p.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="ROC array lengths + per-class classification_report.",
    )
    args = p.parse_args()

    path = Path(args.json_path)
    if not path.is_file():
        raise SystemExit(f"File not found: {path.resolve()}")

    data = json.loads(path.read_text(encoding="utf-8"))
    ev = data.get("evaluation") or {}
    if not isinstance(ev, dict):
        raise SystemExit("Invalid eval JSON: missing evaluation dict.")

    # Match eval_launcher console output.
    print_eval(ev, verbose=bool(args.verbose))
    print(f"Wrote evaluation JSON: {path.resolve()}")


if __name__ == "__main__":
    main()
