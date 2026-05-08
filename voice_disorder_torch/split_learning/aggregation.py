"""State-dict aggregation utilities for split-learning rounds."""

from __future__ import annotations

import copy
from collections.abc import Mapping

import torch


def uniform_average_state_dicts(state_dicts: list[Mapping[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Return the elementwise arithmetic mean of model state dictionaries."""

    if not state_dicts:
        raise ValueError("state_dicts must contain at least one state dict.")
    averaged = copy.deepcopy(dict(state_dicts[0]))
    for key in averaged.keys():
        for state in state_dicts[1:]:
            averaged[key] += state[key]
        averaged[key] = torch.div(averaged[key], len(state_dicts))
    return averaged
