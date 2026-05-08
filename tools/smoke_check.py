"""Quick project health checks for the voice disorder training code.

This script intentionally avoids loading dataset pickle contents or running
training. It verifies import paths, core input files, and minimal CNN forward
passes.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str


def find_project_root(start: Path) -> Path:
    """Find the standalone project root from either root or nested cwd."""

    for candidate in (start.resolve(), *start.resolve().parents):
        if (candidate / "pyproject.toml").is_file() and (
            candidate / "voice_disorder_torch" / "config.py"
        ).is_file() and (
            candidate / "voice_disorder_torch" / "split_learning" / "training.py"
        ).is_file():
            return candidate
    raise RuntimeError("Could not locate project root.")


def add_source_paths(project_root: Path) -> None:
    for path in (project_root,):
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)


def run_check(name: str, fn: Callable[[], str]) -> CheckResult:
    try:
        return CheckResult(name=name, ok=True, detail=fn())
    except Exception as exc:  # noqa: BLE001 - report all smoke-check failures.
        return CheckResult(name=name, ok=False, detail=f"{type(exc).__name__}: {exc}")


def require_dir(path: Path) -> str:
    if not path.is_dir():
        raise FileNotFoundError(path)
    return str(path)


def require_file(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    return str(path)


def require_pickle_dir(path: Path) -> str:
    require_dir(path)
    if not any(path.glob("*.pkl")):
        raise FileNotFoundError(f"No .pkl files found in {path}")
    return str(path)


def check_imports() -> str:
    import voice_disorder_torch  # noqa: F401
    from voice_disorder_torch.config import TrainConfig  # noqa: F401
    from voice_disorder_torch.models.cnn2d import build_cnn2d_original  # noqa: F401
    from voice_disorder_torch.split_learning.models.cnn import build_split_cnn_parts  # noqa: F401

    return "voice_disorder_torch imports succeeded"


def check_cnn_forward() -> str:
    import torch
    from voice_disorder_torch.config import TrainConfig
    from voice_disorder_torch.models.cnn2d import build_cnn2d_original

    sample = torch.zeros(2, 1, 128, 259)
    model = build_cnn2d_original(sample, TrainConfig())
    model.eval()
    with torch.no_grad():
        output = model(sample)
    if tuple(output.shape) != (2, 1):
        raise RuntimeError(f"Unexpected CNN output shape: {tuple(output.shape)}")
    return f"output shape {tuple(output.shape)}"


def check_ssast_default_checkpoint(project_root: Path) -> str:
    from voice_disorder_torch.config import DEFAULT_SSAST_MODEL_SIZE, default_ssast_checkpoint_path

    return require_file(default_ssast_checkpoint_path(DEFAULT_SSAST_MODEL_SIZE, project_root))


def check_split_cnn_forward() -> str:
    import torch
    from voice_disorder_torch.config import TrainConfig
    from voice_disorder_torch.split_learning.models.cnn import build_split_cnn_parts

    sample = torch.zeros(2, 1, 128, 259)
    client, server = build_split_cnn_parts(sample[:1], TrainConfig(), init_seed=2718)
    client.eval()
    server.eval()
    with torch.no_grad():
        output = server(client(sample))
    if tuple(output.shape) != (2, 1):
        raise RuntimeError(f"Unexpected split CNN output shape: {tuple(output.shape)}")
    return f"output shape {tuple(output.shape)}"


def build_checks(project_root: Path) -> list[tuple[str, Callable[[], str]]]:
    return [
        (
            "EENT pickle directory",
            lambda: require_pickle_dir(project_root / "Data" / "EENT_processed" / "pickle_files"),
        ),
        (
            "SVD pickle directory",
            lambda: require_pickle_dir(project_root / "Data" / "SVD_processed" / "pickle_files"),
        ),
        (
            "EENT subjects workbook",
            lambda: require_file(project_root / "metadata" / "subjects" / "EENT_subjects_share_decrypted.xlsx"),
        ),
        (
            "SVD subjects workbook",
            lambda: require_file(project_root / "metadata" / "subjects" / "SVD.xlsx"),
        ),
        (
            "SSAST default checkpoint",
            lambda: check_ssast_default_checkpoint(project_root),
        ),
        ("Package imports", check_imports),
        ("Centralized CNN dummy forward", check_cnn_forward),
        ("Split CNN dummy forward", check_split_cnn_forward),
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run non-training project smoke checks.")
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON instead of human-readable status lines.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = find_project_root(Path.cwd())
    add_source_paths(project_root)

    results = [run_check(name, fn) for name, fn in build_checks(project_root)]
    failed = [result for result in results if not result.ok]

    if args.json:
        payload = {
            "project_root": str(project_root),
            "ok": not failed,
            "results": [asdict(result) for result in results],
        }
        print(json.dumps(payload, indent=2))
    else:
        print(f"Project root: {project_root}")
        for result in results:
            marker = "OK" if result.ok else "FAIL"
            print(f"[{marker}] {result.name}: {result.detail}")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
