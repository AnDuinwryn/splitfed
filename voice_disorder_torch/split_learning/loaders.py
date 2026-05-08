"""DataLoader builders for split-learning client partitions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from voice_disorder_torch.config import RunContext
from voice_disorder_torch.data.datasets import MelSegmentDataset, SsastMelDataset
from voice_disorder_torch.data.load import load_all_preprocessed
from voice_disorder_torch.data.partitioning import assign_partitions_by_patient, indices_for_partition


@dataclass
class PartitionedMelLoaders:
    train_loaders: list[DataLoader]
    val_loader: DataLoader
    n_partitions: int
    shape_probe_nchw: torch.Tensor | None = None


def _select_vowel_split(bundle, vowel: Literal["a", "i"]):
    if vowel == "a":
        return bundle.x_train_a, bundle.y_train_a, bundle.x_val_a, bundle.y_val_a, bundle.id_train_a
    return bundle.x_train_i, bundle.y_train_i, bundle.x_val_i, bundle.y_val_i, bundle.id_train_i


def _partition_loaders(
    *,
    train_ds_full,
    patient_ids: list,
    n_partitions: int,
    partition_seed: int,
    batch_size: int,
) -> list[DataLoader]:
    rng = np.random.default_rng(int(partition_seed))
    patient_map = assign_partitions_by_patient(patient_ids, n_partitions, rng)
    loaders: list[DataLoader] = []
    for partition_id in range(n_partitions):
        idxs = indices_for_partition(patient_ids, patient_map, partition_id)
        if not idxs:
            raise ValueError(
                f"Client partition {partition_id} has no training segments "
                f"({n_partitions=} too large or bad split). "
                "Lower --n-partitions or change --partition-seed."
            )
        loaders.append(
            DataLoader(
                Subset(train_ds_full, idxs),
                batch_size=batch_size,
                shuffle=True,
                num_workers=0,
                drop_last=False,
            )
        )
    return loaders


def build_cnn_partition_loaders(
    *,
    ctx: RunContext,
    vowel: Literal["a", "i"],
    n_partitions: int,
    partition_seed: int,
) -> PartitionedMelLoaders:
    bundle = load_all_preprocessed(ctx.paths, ctx.splits, verbose=True)
    x_tr, y_tr, x_va, y_va, id_tr = _select_vowel_split(bundle, vowel)
    train_ds_full = MelSegmentDataset(x_tr, y_tr)
    val_ds = MelSegmentDataset(x_va, y_va)
    return PartitionedMelLoaders(
        train_loaders=_partition_loaders(
            train_ds_full=train_ds_full,
            patient_ids=id_tr,
            n_partitions=n_partitions,
            partition_seed=partition_seed,
            batch_size=ctx.train.batch_size,
        ),
        val_loader=DataLoader(val_ds, batch_size=ctx.train.batch_size, shuffle=False, num_workers=0),
        shape_probe_nchw=train_ds_full[0][0].unsqueeze(0),
        n_partitions=n_partitions,
    )


def build_ssast_partition_loaders(
    *,
    ctx: RunContext,
    vowel: Literal["a", "i"],
    n_partitions: int,
    partition_seed: int,
    input_tdim: int,
    batch_size: int,
) -> PartitionedMelLoaders:
    bundle = load_all_preprocessed(ctx.paths, ctx.splits, verbose=True)
    x_tr, y_tr, x_va, y_va, id_tr = _select_vowel_split(bundle, vowel)
    train_ds_full = SsastMelDataset(x_tr, y_tr, input_tdim=input_tdim)
    val_ds = SsastMelDataset(x_va, y_va, input_tdim=input_tdim)
    return PartitionedMelLoaders(
        train_loaders=_partition_loaders(
            train_ds_full=train_ds_full,
            patient_ids=id_tr,
            n_partitions=n_partitions,
            partition_seed=partition_seed,
            batch_size=batch_size,
        ),
        val_loader=DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0),
        n_partitions=n_partitions,
    )
