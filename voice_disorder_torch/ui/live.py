from __future__ import annotations

import os
import sys


def supports_ansi(stream=None) -> bool:
    stream = stream or sys.stdout
    return hasattr(stream, "isatty") and stream.isatty() and os.environ.get("TERM", "") not in {"", "dumb"}


class LiveBlock:
    """Redraw a fixed-height block in-place (ANSI terminals only)."""

    def __init__(self, *, height: int, stream=None) -> None:
        self.stream = stream or sys.stdout
        self.height = int(height)
        self.enabled = supports_ansi(self.stream)
        self._printed_once = False

    def redraw(self, lines: list[str]) -> None:
        if not self.enabled:
            for ln in lines:
                print(ln)
            return
        if len(lines) != self.height:
            raise ValueError(f"LiveBlock height mismatch: expected {self.height}, got {len(lines)}")
        if self._printed_once:
            self.stream.write(f"\x1b[{self.height}A")  # cursor up
        for ln in lines:
            self.stream.write("\r\x1b[2K" + ln + "\n")  # clear line + write
        self.stream.flush()
        self._printed_once = True

