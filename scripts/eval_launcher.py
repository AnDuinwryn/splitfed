#!/usr/bin/env python3
"""Interactive launcher for evaluating /a/+ /i/ model pairs.

Select a run once; the launcher derives both stems (a+i) automatically.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from _path_setup import ensure_project_package_on_path

ensure_project_package_on_path()


_CENTRAL_RE = re.compile(r"^(?P<kind>.+)_(?P<vowel>[ai])_d(?P<d>\d+)_t(?P<t>\d+)_i(?P<i>\d+)$")
_SPLIT_CNN_RE = re.compile(r"^split_cnn_(?P<vowel>[ai])_d(?P<d>\d+)_t(?P<t>\d+)_i(?P<i>\d+)$")
_SPLIT_SSAST_RE = re.compile(
    r"^split_ssast_(?P<vowel>[ai])_d(?P<d>\d+)_t(?P<t>\d+)_i(?P<i>\d+)_b(?P<b>\d+)$"
)


@dataclass(frozen=True)
class PairChoice:
    key: str
    group: str
    script: str
    model_type: str
    stem_a: str
    stem_i: str


def _supports_ansi() -> bool:
    return sys.stdout.isatty() and os.environ.get("TERM", "") not in {"", "dumb"}


def _discover_pairs(save_dir: Path) -> list[PairChoice]:
    # Centralized checkpoints: {stem}.pt (exclude split-learning parts)
    central_stems = {
        p.stem
        for p in save_dir.glob("*.pt")
        if not (p.stem.endswith("_split_client") or p.stem.endswith("_split_server"))
    }
    # Split-learning checkpoints: {stem}_split_client.pt / {stem}_split_server.pt
    split_stems = {p.stem.removesuffix("_split_client") for p in save_dir.glob("*_split_client.pt")}

    stems = sorted(central_stems | split_stems)
    central: dict[tuple[str, str, str, str], dict[str, str]] = {}
    split_cnn: dict[tuple[str, str, str], dict[str, str]] = {}
    split_ssast: dict[tuple[str, str, str, str], dict[str, str]] = {}

    for s in stems:
        m = _SPLIT_CNN_RE.match(s)
        if m:
            vowel, d, t, i = m.group("vowel", "d", "t", "i")
            key = (d, t, i)
            split_cnn.setdefault(key, {})[vowel] = s
            continue
        m = _SPLIT_SSAST_RE.match(s)
        if m:
            vowel, d, t, i, b = m.group("vowel", "d", "t", "i", "b")
            key = (d, t, i, b)
            split_ssast.setdefault(key, {})[vowel] = s
            continue
        m = _CENTRAL_RE.match(s)
        if m:
            kind, vowel, d, t, i = m.group("kind", "vowel", "d", "t", "i")
            key = (kind, d, t, i)
            central.setdefault(key, {})[vowel] = s
            continue

    out: list[PairChoice] = []
    for (kind, d, t, i), have in sorted(central.items()):
        if "a" in have and "i" in have:
            out.append(
                PairChoice(
                    key=f"central_d{d}_t{t}_i{i}",
                    group=f"central/{kind}",
                    script="scripts/evaluate.py",
                    model_type=kind,
                    stem_a=have["a"],
                    stem_i=have["i"],
                )
            )
    for (d, t, i), have in sorted(split_cnn.items()):
        if "a" in have and "i" in have:
            out.append(
                PairChoice(
                    key=f"split_cnn_d{d}_t{t}_i{i}",
                    group="split/cnn",
                    script="scripts/split_learning/evaluate_split.py",
                    model_type="cnn",
                    stem_a=have["a"],
                    stem_i=have["i"],
                )
            )
    for (d, t, i, b), have in sorted(split_ssast.items()):
        if "a" in have and "i" in have:
            out.append(
                PairChoice(
                    key=f"split_ssast_d{d}_t{t}_i{i}_b{b}",
                    group="split/ssast",
                    script="scripts/split_learning/evaluate_split.py",
                    model_type="ssast",
                    stem_a=have["a"],
                    stem_i=have["i"],
                )
            )
    return out


def _run_eval(choice: PairChoice, *, save_dir: Path, results_json: Path | None) -> int:
    cmd = [
        sys.executable,
        choice.script,
        "--save-dir",
        str(save_dir),
        "--model-a",
        choice.stem_a,
        "--model-i",
        choice.stem_i,
        "--model-type",
        choice.model_type,
        "--patient-eval-strategy",
        "fixed",
        "--patient-prob-threshold",
        "0.5",
    ]
    if os.environ.get("VD_EVAL_VERBOSE") == "1":
        cmd.append("--verbose")
    if results_json is not None:
        cmd += ["--results-json", str(results_json)]
    return subprocess.call(cmd)


def _select_curses(choices: list[PairChoice]) -> PairChoice | None:
    import curses  # stdlib

    def ui(stdscr):
        curses.curs_set(0)
        curses.start_color()
        curses.use_default_colors()
        cyan = curses.COLOR_CYAN
        curses.init_pair(1, cyan, -1)
        idx = 0
        while True:
            stdscr.erase()
            h, w = stdscr.getmaxyx()
            title = "Select evaluation key (↑/↓, Enter, q)"
            stdscr.addnstr(0, 0, title, w - 1)
            for j, ch in enumerate(choices[: max(0, h - 2)]):
                prefix = "> " if j == idx else "  "
                line = prefix + ch.key
                if j == idx:
                    stdscr.attron(curses.color_pair(1) | curses.A_BOLD)
                    stdscr.addnstr(j + 1, 0, line, w - 1)
                    stdscr.attroff(curses.color_pair(1) | curses.A_BOLD)
                else:
                    stdscr.addnstr(j + 1, 0, line, w - 1)
            stdscr.refresh()

            key = stdscr.getch()
            if key in (ord("q"), 27):
                return None
            if key in (curses.KEY_UP, ord("k")):
                idx = (idx - 1) % len(choices)
            elif key in (curses.KEY_DOWN, ord("j")):
                idx = (idx + 1) % len(choices)
            elif key in (curses.KEY_ENTER, 10, 13):
                return choices[idx]

    return curses.wrapper(ui)


def main() -> int:
    ap = argparse.ArgumentParser(description="Interactive evaluate launcher (auto-pairs /a/+ /i/).")
    ap.add_argument("--save-dir", type=Path, default=Path("./saved_models"))
    ap.add_argument("--results-json", type=Path, default=None)
    ap.add_argument("--verbose", action="store_true", help="Print single-vowel blocks too.")
    args = ap.parse_args()

    choices = _discover_pairs(Path(args.save_dir))
    if not choices:
        print(f"No model pairs found under {Path(args.save_dir).resolve()}")
        return 2

    if sys.stdout.isatty():
        picked = _select_curses(choices)
        if picked is None:
            return 1
    else:
        # Non-interactive fallback
        for i, ch in enumerate(choices):
            print(f"{i:>2}: {ch.key}")
        return 2

    # Pass through verbose to underlying scripts.
    if args.verbose:
        os.environ["VD_EVAL_VERBOSE"] = "1"
    return _run_eval(picked, save_dir=Path(args.save_dir), results_json=args.results_json)


if __name__ == "__main__":
    raise SystemExit(main())

