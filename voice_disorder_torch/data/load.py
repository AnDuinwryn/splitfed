from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np

from typing import Optional

from ..config import DataPaths, SplitSeeds
from ..ui.status import status
from .eent_subjects import resolve_chinese_subject_tables
from .preprocess import mel_first_channel, normalize_mel
from .splits import (
    build_sensitive_attrs_dict,
    load_patient_metadata,
    stratified_patient_split,
    stratified_segment_split,
)


@dataclass
class LoadedDataBundle:
    x_train_a: np.ndarray
    x_val_a: np.ndarray
    x_test_a: np.ndarray
    y_train_a: np.ndarray
    y_val_a: np.ndarray
    y_test_a: np.ndarray
    id_train_a: list
    id_val_a: list
    id_test_a: list
    x_train_i: np.ndarray
    x_val_i: np.ndarray
    x_test_i: np.ndarray
    y_train_i: np.ndarray
    y_val_i: np.ndarray
    y_test_i: np.ndarray
    id_train_i: list
    id_val_i: list
    id_test_i: list
    x_ger_a: np.ndarray
    y_ger_a: np.ndarray
    id_ger_a: list
    x_ger_i: np.ndarray
    y_ger_i: np.ndarray
    id_ger_i: list
    sensitive_attrs_train: dict
    sensitive_attrs_val: dict
    sensitive_attrs_test: dict
    sensitive_attrs_ger: dict


_BUNDLE_CACHE: dict[tuple, LoadedDataBundle] = {}


def _cache_key(paths: DataPaths, splits: SplitSeeds) -> tuple:
    def norm(p: Optional[Path]) -> str | None:
        return str(p.resolve()) if isinstance(p, Path) else None

    return (
        norm(paths.pickle_dir),
        norm(paths.pickle_dir_chinese),
        norm(paths.pickle_dir_german),
        norm(paths.eent_subjects_xlsx),
        norm(paths.german_subjects_xlsx),
        int(splits.dev_test_seed),
        int(splits.train_val_seed),
    )


def _chinese_pickle_dir(paths: DataPaths) -> Path:
    d: Optional[Path] = paths.pickle_dir_chinese or paths.pickle_dir
    if d is None:
        raise ValueError("Set --pickle-dir or --pickle-dir-eent (with --pickle-dir-svd).")
    return d


def _german_pickle_dir(paths: DataPaths) -> Path:
    d: Optional[Path] = paths.pickle_dir_german or paths.pickle_dir
    if d is None:
        raise ValueError("Set --pickle-dir or --pickle-dir-svd (with --pickle-dir-eent).")
    return d


def _load_chinese_pickles(pickle_dir: Path):
    read_path_a = pickle_dir / "vowel-a_mel_ch.pkl"
    read_path_i = pickle_dir / "vowel-i_mel_ch.pkl"
    dataset_a = joblib.load(read_path_a)
    dataset_i = joblib.load(read_path_i)
    return (
        dataset_a["features"],
        dataset_a["labels"],
        dataset_a["ids"],
        dataset_i["features"],
        dataset_i["labels"],
        dataset_i["ids"],
    )


def _collect_segments_for_patients(features_raw, labels_raw, ids_raw, patient_ids: list):
    feats, labs, ids = [], [], []
    for pt_id in patient_ids:
        indices = [i for i, tar in enumerate(ids_raw) if str(tar) == str(pt_id)]
        for idx in indices:
            feats.append(features_raw[idx])
            labs.append(labels_raw[idx])
            ids.append(ids_raw[idx])
    return feats, labs, ids


def load_chinese_tensors(
    paths: DataPaths,
    splits: SplitSeeds,
    val_fraction_within_dev: float = 0.2,
    verbose: bool = True,
) -> tuple:
    split_df, age_group_map, gender_map = resolve_chinese_subject_tables(paths)
    features_raw_a, labels_raw_a, ids_raw_a, features_raw_i, labels_raw_i, ids_raw_i = _load_chinese_pickles(
        _chinese_pickle_dir(paths)
    )

    pt_id_develop, pt_id_test = stratified_patient_split(
        split_df, age_group_map, gender_map, test_size=0.2, random_state=splits.dev_test_seed, verbose=verbose
    )

    fa, la, ida = _collect_segments_for_patients(features_raw_a, labels_raw_a, ids_raw_a, pt_id_develop)
    fi, li, idi = _collect_segments_for_patients(features_raw_i, labels_raw_i, ids_raw_i, pt_id_develop)

    x_develop_a = np.vstack(fa)
    y_develop_a = np.where(np.equal(np.vstack(la), 0), 0, 1)
    x_develop_i = np.vstack(fi)
    y_develop_i = np.where(np.equal(np.vstack(li), 0), 0, 1)

    x_train_a, x_val_a, y_train_a, y_val_a, id_train_a, id_val_a = stratified_segment_split(
        x_develop_a,
        y_develop_a,
        ida,
        age_group_map,
        gender_map,
        test_size=val_fraction_within_dev,
        random_state=splits.train_val_seed,
        vowel_type="a",
        verbose=verbose,
    )
    x_train_i, x_val_i, y_train_i, y_val_i, id_train_i, id_val_i = stratified_segment_split(
        x_develop_i,
        y_develop_i,
        idi,
        age_group_map,
        gender_map,
        test_size=val_fraction_within_dev,
        random_state=splits.train_val_seed,
        vowel_type="i",
        verbose=verbose,
    )

    fta, lta, idta = _collect_segments_for_patients(features_raw_a, labels_raw_a, ids_raw_a, pt_id_test)
    fti, lti, idti = _collect_segments_for_patients(features_raw_i, labels_raw_i, ids_raw_i, pt_id_test)

    x_test_a = np.vstack(fta)
    y_test_a = np.where(np.equal(np.vstack(lta), 0), 0, 1)
    x_test_i = np.vstack(fti)
    y_test_i = np.where(np.equal(np.vstack(lti), 0), 0, 1)
    id_test_a = idta
    id_test_i = idti

    x_train_a, x_val_a, x_test_a = (
        mel_first_channel(x_train_a),
        mel_first_channel(x_val_a),
        mel_first_channel(x_test_a),
    )
    x_train_i, x_val_i, x_test_i = (
        mel_first_channel(x_train_i),
        mel_first_channel(x_val_i),
        mel_first_channel(x_test_i),
    )
    x_train_a = normalize_mel(x_train_a, delta=0, norm_mode="per sample")
    x_val_a = normalize_mel(x_val_a, delta=0, norm_mode="per sample")
    x_test_a = normalize_mel(x_test_a, delta=0, norm_mode="per sample")
    x_train_i = normalize_mel(x_train_i, delta=0, norm_mode="per sample")
    x_val_i = normalize_mel(x_val_i, delta=0, norm_mode="per sample")
    x_test_i = normalize_mel(x_test_i, delta=0, norm_mode="per sample")

    train_patients = list(set(id_train_a + id_train_i))
    val_patients = list(set(id_val_a + id_val_i))
    test_patients = list(set(id_test_a + id_test_i))
    sensitive_train = build_sensitive_attrs_dict(train_patients, age_group_map, gender_map)
    sensitive_val = build_sensitive_attrs_dict(val_patients, age_group_map, gender_map)
    sensitive_test = build_sensitive_attrs_dict(test_patients, age_group_map, gender_map)

    return (
        x_train_a,
        x_val_a,
        x_test_a,
        y_train_a,
        y_val_a,
        y_test_a,
        id_train_a,
        id_val_a,
        id_test_a,
        x_train_i,
        x_val_i,
        x_test_i,
        y_train_i,
        y_val_i,
        y_test_i,
        id_train_i,
        id_val_i,
        id_test_i,
        sensitive_train,
        sensitive_val,
        sensitive_test,
    )


def load_german_tensors(paths: DataPaths, verbose: bool = True):
    age_group_map, gender_map = load_patient_metadata(paths, "german")
    from .german_subjects import german_labels_frame_from_paths

    df = german_labels_frame_from_paths(paths)
    picked_ids = set(df["ID"].tolist())
    df = df.set_index("ID")

    ger_dir = _german_pickle_dir(paths)
    read_path_a = ger_dir / "a_mel_ger.pkl"
    read_path_i = ger_dir / "i_mel_ger.pkl"
    dataset_a = joblib.load(read_path_a)
    dataset_i = joblib.load(read_path_i)

    x_ger_a, y_ger_a, id_ger_a = [], [], []
    for i, pid in enumerate(dataset_a["ids"]):
        pid = int(pid)
        if pid in picked_ids:
            x_ger_a.append(dataset_a["features"][i])
            y_ger_a.append(df.loc[pid, "Class"])
            id_ger_a.append(pid)
    x_ger_a = np.vstack(x_ger_a)
    y_ger_a = np.array(y_ger_a, ndmin=2).T

    x_ger_i, y_ger_i, id_ger_i = [], [], []
    for i, pid in enumerate(dataset_i["ids"]):
        pid = int(pid)
        if pid in picked_ids:
            x_ger_i.append(dataset_i["features"][i])
            y_ger_i.append(df.loc[pid, "Class"])
            id_ger_i.append(pid)
    x_ger_i = np.vstack(x_ger_i)
    y_ger_i = np.array(y_ger_i, ndmin=2).T

    x_ger_a, x_ger_i = mel_first_channel(x_ger_a), mel_first_channel(x_ger_i)
    x_ger_a = normalize_mel(x_ger_a, delta=0, norm_mode="per sample")
    x_ger_i = normalize_mel(x_ger_i, delta=0, norm_mode="per sample")

    ger_patients = list(set(id_ger_a + id_ger_i))
    sensitive_ger = build_sensitive_attrs_dict(ger_patients, age_group_map, gender_map)
    return x_ger_a, y_ger_a, id_ger_a, x_ger_i, y_ger_i, id_ger_i, sensitive_ger


def load_all_preprocessed(paths: DataPaths, splits: SplitSeeds, verbose: bool = True) -> LoadedDataBundle:
    key = _cache_key(paths, splits)
    cached = _BUNDLE_CACHE.get(key)
    if cached is not None:
        return cached
    if verbose:
        with status("Loading EENT dataset"):
            ch = load_chinese_tensors(paths, splits, verbose=False)
        with status("Loading SVD dataset"):
            ger = load_german_tensors(paths, verbose=False)
    else:
        ch = load_chinese_tensors(paths, splits, verbose=False)
        ger = load_german_tensors(paths, verbose=False)
    (
        x_train_a,
        x_val_a,
        x_test_a,
        y_train_a,
        y_val_a,
        y_test_a,
        id_train_a,
        id_val_a,
        id_test_a,
        x_train_i,
        x_val_i,
        x_test_i,
        y_train_i,
        y_val_i,
        y_test_i,
        id_train_i,
        id_val_i,
        id_test_i,
        sensitive_attrs_train,
        sensitive_attrs_val,
        sensitive_attrs_test,
    ) = ch
    bundle = LoadedDataBundle(
        x_train_a=x_train_a,
        x_val_a=x_val_a,
        x_test_a=x_test_a,
        y_train_a=y_train_a,
        y_val_a=y_val_a,
        y_test_a=y_test_a,
        id_train_a=id_train_a,
        id_val_a=id_val_a,
        id_test_a=id_test_a,
        x_train_i=x_train_i,
        x_val_i=x_val_i,
        x_test_i=x_test_i,
        y_train_i=y_train_i,
        y_val_i=y_val_i,
        y_test_i=y_test_i,
        id_train_i=id_train_i,
        id_val_i=id_val_i,
        id_test_i=id_test_i,
        x_ger_a=ger[0],
        y_ger_a=ger[1],
        id_ger_a=ger[2],
        x_ger_i=ger[3],
        y_ger_i=ger[4],
        id_ger_i=ger[5],
        sensitive_attrs_train=sensitive_attrs_train,
        sensitive_attrs_val=sensitive_attrs_val,
        sensitive_attrs_test=sensitive_attrs_test,
        sensitive_attrs_ger=ger[6],
    )
    _BUNDLE_CACHE[key] = bundle
    return bundle


def load_test_only_bundle(paths: DataPaths, splits: SplitSeeds) -> dict:
    """Rebuild splits with same seeds; return only Chinese test + German (for evaluation)."""
    b = load_all_preprocessed(paths, splits, verbose=False)
    return {
        "chinese": {
            "x_test_a": b.x_test_a,
            "y_test_a": b.y_test_a,
            "id_test_a": b.id_test_a,
            "x_test_i": b.x_test_i,
            "y_test_i": b.y_test_i,
            "id_test_i": b.id_test_i,
            "sensitive_attrs": b.sensitive_attrs_test,
        },
        "german": {
            "x_ger_a": b.x_ger_a,
            "y_ger_a": b.y_ger_a,
            "id_ger_a": b.id_ger_a,
            "x_ger_i": b.x_ger_i,
            "y_ger_i": b.y_ger_i,
            "id_ger_i": b.id_ger_i,
            "sensitive_attrs": b.sensitive_attrs_ger,
        },
    }
