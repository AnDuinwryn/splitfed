from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np
import torch


def set_reproducible(
    seed: int,
    *,
    cudnn_benchmark: bool = False,
    cudnn_deterministic: bool = True,
    num_threads: Optional[int] = None,
) -> None:
    """Best-effort deterministic setup for Python, NumPy, and PyTorch."""
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if num_threads is not None:
        torch.set_num_threads(int(num_threads))
        try:
            torch.set_num_interop_threads(int(num_threads))
        except Exception:
            pass

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = cudnn_benchmark
        torch.backends.cudnn.deterministic = cudnn_deterministic

    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


def dataloader_generator(seed: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(int(seed))
    return g
