from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from ..config import DataPaths


def create_composite_stratify_key(data_df: pd.DataFrame, age_group_map: dict, gender_map: dict) -> list[str]:
    composite_key: list[str] = []
    for _, row in data_df.iterrows():
        patient_id = row["ID"]
        disease_class = row["Class"]
        age_group = age_group_map.get(patient_id, -1)
        gender = gender_map.get(patient_id, -1)
        composite_key.append(f"{age_group}_{gender}_{disease_class}")
    return composite_key


def stratified_patient_split(
    split_df: pd.DataFrame,
    age_group_map: dict,
    gender_map: dict,
    test_size: float = 0.2,
    random_state: int = 0,
    verbose: bool = True,
) -> tuple[list, list]:
    composite_stratify = create_composite_stratify_key(split_df, age_group_map, gender_map)
    unique_combinations, counts = np.unique(composite_stratify, return_counts=True)
    min_samples = 2
    small_groups = [(c, n) for c, n in zip(unique_combinations, counts) if n < min_samples]

    if small_groups:
        if verbose:
            print(f"Warning: {len(small_groups)} stratify groups small; falling back to Class stratify.")
        dev_ids, test_ids = train_test_split(
            split_df["ID"], test_size=test_size, stratify=split_df["Class"], random_state=random_state
        )
    else:
        dev_ids, test_ids = train_test_split(
            split_df["ID"], test_size=test_size, stratify=composite_stratify, random_state=random_state
        )
    return dev_ids.tolist(), test_ids.tolist()


def create_segment_composite_stratify_key(
    segment_ids: list, segment_labels, age_group_map: dict, gender_map: dict
) -> list[str]:
    composite_key: list[str] = []
    labels = np.asarray(segment_labels).reshape(len(segment_labels), -1)
    for i, patient_id in enumerate(segment_ids):
        label = labels[i].flatten()[0]
        label = int(label)
        age_group = age_group_map.get(patient_id, -1)
        gender = gender_map.get(patient_id, -1)
        composite_key.append(f"{age_group}_{gender}_{label}")
    return composite_key


def stratified_segment_split(
    x_data: np.ndarray,
    y_data: np.ndarray,
    id_data,
    age_group_map: dict,
    gender_map: dict,
    test_size: float = 0.2,
    random_state: int = 300,
    vowel_type: str = "unknown",
    verbose: bool = True,
):
    composite_stratify = create_segment_composite_stratify_key(id_data, y_data, age_group_map, gender_map)
    y_flat = np.asarray(y_data).reshape(len(y_data), -1).ravel()
    unique_combinations, counts = np.unique(composite_stratify, return_counts=True)
    min_samples = 2
    small_groups = [(c, n) for c, n in zip(unique_combinations, counts) if n < min_samples]

    if small_groups:
        if verbose:
            print(f"Warning: vowel {vowel_type}: small groups; stratify by label only.")
        x_train, x_val, y_train, y_val, id_train, id_val = train_test_split(
            x_data, y_data, id_data, test_size=test_size, random_state=random_state, stratify=y_flat
        )
    else:
        x_train, x_val, y_train, y_val, id_train, id_val = train_test_split(
            x_data, y_data, id_data, test_size=test_size, random_state=random_state, stratify=composite_stratify
        )
    return x_train, x_val, y_train, y_val, id_train, id_val


def load_patient_metadata(paths: DataPaths, dataset: str) -> tuple[dict, dict]:
    if dataset == "chinese":
        from .eent_subjects import resolve_chinese_subject_tables

        _, age_group_map, gender_map = resolve_chinese_subject_tables(paths)
        return age_group_map, gender_map
    if dataset == "german":
        if paths.german_subjects_xlsx is None:
            raise ValueError("SVD metadata workbook is not set on DataPaths.")
        from .german_subjects import load_german_subjects_from_xlsx

        _, age_group_map, gender_map = load_german_subjects_from_xlsx(paths.german_subjects_xlsx)
        return age_group_map, gender_map
    raise ValueError("dataset must be 'chinese' or 'german'")


def build_sensitive_attrs_dict(patient_ids: list, age_map: dict, gender_map: dict) -> dict:
    out: dict = {}
    for pid in patient_ids:
        out[pid] = {"age_group": age_map.get(pid, -1), "gender": gender_map.get(pid, -1)}
    return out
