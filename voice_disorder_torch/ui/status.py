from __future__ import annotations

import itertools
import os
import sys
import threading
import time
from contextlib import contextmanager


def _supports_ansi(stream) -> bool:
    if not hasattr(stream, "isatty") or not stream.isatty():
        return False
    term = os.environ.get("TERM", "")
    return term not in {"", "dumb"}


@contextmanager
def status(label: str, *, stream=None, interval_s: float = 0.24):
    """Minimal terminal status spinner (uv-like dots) with green check on success."""

    stream = stream or sys.stderr
    ansi = _supports_ansi(stream)
    stop = threading.Event()
    frames = itertools.cycle([".   ", "..  ", "... ", "...."])
    prefix_ok = "\x1b[32m✓\x1b[0m" if ansi else "OK"
    prefix_fail = "\x1b[31m✗\x1b[0m" if ansi else "FAIL"
    max_len = len(f"{label} ....")

    def run() -> None:
        while not stop.is_set():
            frame = next(frames)
            text = f"{label} {frame}"
            stream.write("\r" + text)
            stream.flush()
            time.sleep(interval_s)

    t = threading.Thread(target=run, daemon=True)
    t.start()
    try:
        yield
    except Exception:
        stop.set()
        t.join(timeout=1)
        stream.write("\r" + (" " * max_len) + "\r" + f"{prefix_fail} {label}\n")
        stream.flush()
        raise
    else:
        stop.set()
        t.join(timeout=1)
        stream.write("\r" + (" " * max_len) + "\r" + f"{prefix_ok} {label}\n")
        stream.flush()

