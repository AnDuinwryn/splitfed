#!/usr/bin/env bash
set -euo pipefail

uv run --no-sync python paper2601_make_domain_shift_report.py "$@"
