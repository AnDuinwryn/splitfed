from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


DEFAULT_EENT_PICKLE_DIR = Path("Data/EENT_processed/pickle_files")
DEFAULT_GERMAN_PICKLE_DIR = Path("Data/SVD_processed/pickle_files")
DEFAULT_EENT_SUBJECTS_XLSX = Path("metadata/subjects/EENT_subjects_share_decrypted.xlsx")
DEFAULT_GERMAN_SUBJECTS_XLSX = Path("metadata/subjects/SVD.xlsx")
DEFAULT_BATCH_SIZE = 256
DEFAULT_SSAST_MODEL_SIZE = "base"
DEFAULT_SSAST_CHECKPOINTS = {
    "base": Path("pretrained_models/SSAST-Base-Patch-400.pth"),
    "tiny": Path("pretrained_models/SSAST-Tiny-Patch-400.pth"),
}


def default_ssast_checkpoint_path(model_size: str = DEFAULT_SSAST_MODEL_SIZE, project_root: Path | None = None) -> Path:
    checkpoint = DEFAULT_SSAST_CHECKPOINTS.get(model_size.lower().strip())
    if checkpoint is None:
        valid = ", ".join(sorted(DEFAULT_SSAST_CHECKPOINTS))
        raise ValueError(f"No default SSAST checkpoint for model_size={model_size!r}; available: {valid}.")
    return project_root / checkpoint if project_root is not None and not checkpoint.is_absolute() else checkpoint


def apply_default_project_paths(args: Any) -> None:
    """Use project data and metadata inputs unless explicit alternatives are supplied."""
    if args.pickle_dir is None and args.pickle_dir_eent is None and args.pickle_dir_svd is None:
        args.pickle_dir_eent = DEFAULT_EENT_PICKLE_DIR
        args.pickle_dir_svd = DEFAULT_GERMAN_PICKLE_DIR
    if args.eent_subjects_xlsx is None:
        args.eent_subjects_xlsx = DEFAULT_EENT_SUBJECTS_XLSX
    if args.german_subjects_xlsx is None:
        args.german_subjects_xlsx = DEFAULT_GERMAN_SUBJECTS_XLSX


@dataclass
class DataPaths:
    """All filesystem inputs for Chinese (EENT) and German (SVD) pipelines."""

    pickle_dir: Optional[Path] = None
    pickle_dir_chinese: Optional[Path] = None  # e.g. Data/EENT_processed/pickle_files
    pickle_dir_german: Optional[Path] = None  # e.g. Data/SVD_processed/pickle_files
    german_subjects_xlsx: Optional[Path] = None
    eent_subjects_xlsx: Optional[Path] = None


@dataclass
class SplitSeeds:
    """Controls dev/test and train/val splits."""

    dev_test_seed: int = 8
    train_val_seed: int = 100


@dataclass
class TrainConfig:
    """Training hyperparameters shared by model families."""

    lr: float = 5e-5
    batch_size: int = DEFAULT_BATCH_SIZE
    max_epochs: int = 250
    early_stopping_patience: int = 10
    dropout: float = 0.5
    kernel_size: tuple[int, int] = (3, 3)
    conv_stride: tuple[int, int] = (1, 1)
    pool_size: tuple[int, int] = (2, 3)


@dataclass
class RunContext:
    paths: DataPaths
    splits: SplitSeeds = field(default_factory=SplitSeeds)
    train: TrainConfig = field(default_factory=TrainConfig)
    save_dir: Path = field(default_factory=lambda: Path("./saved_models"))
    device: Optional[str] = None  # "cuda", "cpu", or None for auto
