"""Patient-level partitioning helpers for split-learning experiments."""

from __future__ import annotations

import numpy as np


def assign_partitions_by_patient(
    patient_ids: list,
    n_partitions: int,
    rng: np.random.Generator,
) -> dict[str, int]:
    """Assign each unique patient ID to exactly one partition."""

    if n_partitions <= 0:
        raise ValueError("n_partitions must be positive.")
    unique = list(dict.fromkeys(str(p) for p in patient_ids))
    order = unique.copy()
    rng.shuffle(order)
    bins = np.array_split(np.array(order, dtype=object), n_partitions)
    out: dict[str, int] = {}
    for partition_id, group in enumerate(bins):
        for patient_id in group.tolist():
            out[str(patient_id)] = int(partition_id)
    return out


def indices_for_partition(
    patient_ids: list,
    patient_to_partition: dict[str, int],
    partition_id: int,
) -> list[int]:
    """Return segment indices whose patient belongs to ``partition_id``."""

    return [i for i, patient_id in enumerate(patient_ids) if patient_to_partition[str(patient_id)] == partition_id]
