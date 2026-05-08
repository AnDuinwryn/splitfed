"""Evaluation orchestration for split-learning checkpoints."""

from __future__ import annotations

from pathlib import Path

import torch

from voice_disorder_torch.config import DataPaths, SplitSeeds, TrainConfig
from voice_disorder_torch.data.load import load_test_only_bundle
from voice_disorder_torch.evaluation.patient_eval import combined_vowel_ai_eval, model_eval_by_id
from voice_disorder_torch.naming import parse_model_seeds_from_name

from .inference import (
    build_loaded_cnn_split,
    build_loaded_ssast_split,
    predict_cnn_split_proba,
    predict_ssast_split_proba,
)
from .metadata import load_split_metadata


def evaluate_split_model_pair_test_only(
    *,
    paths: DataPaths,
    save_dir: Path,
    model_a_stem: str,
    model_i_stem: str,
    model_type: str,
    train_cfg: TrainConfig | None = None,
    device: str | None = None,
    patient_eval_strategy: str = "fixed",
    patient_prob_threshold: float = 0.5,
    verbose: bool = False,
) -> dict:
    train_cfg = train_cfg or TrainConfig()
    seeds = parse_model_seeds_from_name(model_a_stem) or {}
    dev_seed = int(seeds.get("dev_test_seed", 8))
    splits = SplitSeeds(dev_test_seed=dev_seed, train_val_seed=int(seeds.get("train_val_seed", 100)))
    test_data = load_test_only_bundle(paths, splits)
    device_t = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model_key = model_type.lower().strip()

    chinese = test_data["chinese"]
    german = test_data["german"]
    x0 = chinese["x_test_a"]

    if model_key in {"cnn", "cnn2d", "cnn2d_original"}:
        client_a, server_a = build_loaded_cnn_split(save_dir, model_a_stem, x0, train_cfg, device_t)
        client_i, server_i = build_loaded_cnn_split(save_dir, model_i_stem, x0, train_cfg, device_t)

        def proba_fn(client, server, x):
            return predict_cnn_split_proba(client, server, x, device_t, train_cfg.batch_size)

    elif model_key in {"ssast", "ast"}:
        client_a, server_a = build_loaded_ssast_split(save_dir, model_a_stem, device_t)
        client_i, server_i = build_loaded_ssast_split(save_dir, model_i_stem, device_t)
        meta_a = load_split_metadata(save_dir, model_a_stem)
        input_tdim = int(meta_a["ssast"]["input_tdim"])

        def proba_fn(client, server, x):
            return predict_ssast_split_proba(client, server, x, device_t, train_cfg.batch_size, input_tdim)

    else:
        raise ValueError(f"Unknown model_type for split eval: {model_type}")

    def run_block(name: str, block: dict, sensitive_attrs: dict | None) -> dict:
        xa, ya, ida = block["x_test_a"], block["y_test_a"], block["id_test_a"]
        xi, yi, idi = block["x_test_i"], block["y_test_i"], block["id_test_i"]
        pa = proba_fn(client_a, server_a, xa)
        pi = proba_fn(client_i, server_i, xi)
        result_a = model_eval_by_id(
            xa,
            ya,
            list(ida),
            pa,
            vowel_type="a",
            dataset_type=name,
            sensitive_attrs=sensitive_attrs,
            strategy=patient_eval_strategy,  # type: ignore[arg-type]
            patient_prob_threshold=patient_prob_threshold,
            verbose=verbose,
        )
        result_i = model_eval_by_id(
            xi,
            yi,
            list(idi),
            pi,
            vowel_type="i",
            dataset_type=name,
            sensitive_attrs=sensitive_attrs,
            strategy=patient_eval_strategy,  # type: ignore[arg-type]
            patient_prob_threshold=patient_prob_threshold,
            verbose=verbose,
        )
        combined = combined_vowel_ai_eval(
            pa,
            pi,
            ya,
            yi,
            ida,
            idi,
            dataset_type=name,
            strategy=patient_eval_strategy,  # type: ignore[arg-type]
            patient_prob_threshold=patient_prob_threshold,
            verbose=verbose,
        )
        return {"single_a": result_a, "single_i": result_i, "combined": combined}

    return {
        "patient_eval_strategy": patient_eval_strategy,
        "patient_prob_threshold": float(patient_prob_threshold),
        "chinese": run_block("Chinese-Test", chinese, chinese.get("sensitive_attrs")),
        "german": run_block(
            "German-Test",
            {
                "x_test_a": german["x_ger_a"],
                "y_test_a": german["y_ger_a"],
                "id_test_a": german["id_ger_a"],
                "x_test_i": german["x_ger_i"],
                "y_test_i": german["y_ger_i"],
                "id_test_i": german["id_ger_i"],
            },
            german.get("sensitive_attrs"),
        ),
        "dev_test_seed": dev_seed,
        "model_a": model_a_stem,
        "model_i": model_i_stem,
        "split_learning": True,
    }
