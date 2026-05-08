from __future__ import annotations

from pathlib import Path

import torch

from ..config import DataPaths, SplitSeeds, TrainConfig
from ..data.load import load_test_only_bundle
from ..evaluation.patient_eval import combined_vowel_ai_eval, model_eval_by_id
from ..evaluation.torch_predict import predict_positive_proba
from ..models.factory import build_trainable_backbone
from ..naming import parse_model_seeds_from_name
from ..io.artifacts import load_state_dict


def _infer_bundle_shapes(chinese: dict) -> tuple[int, int, int, int]:
    x = chinese["x_test_a"]
    n, h, w, c = x.shape
    return n, h, w, c


def evaluate_model_pair_test_only(
    *,
    paths: DataPaths,
    save_dir: Path,
    model_a_stem: str,
    model_i_stem: str,
    model_type: str = "cnn",
    train_cfg: TrainConfig | None = None,
    device: str | None = None,
    patient_eval_strategy: str = "fixed",
    patient_prob_threshold: float = 0.5,
    verbose: bool = False,
) -> dict:
    """
    Load two checkpoints (same architecture), rebuild Chinese+German test tensors with dev seed from names.
    """
    train_cfg = train_cfg or TrainConfig()
    seeds = parse_model_seeds_from_name(model_a_stem) or {}
    dev_seed = seeds.get("dev_test_seed", 8)
    splits = SplitSeeds(dev_test_seed=int(dev_seed), train_val_seed=int(seeds.get("train_val_seed", 100)))
    test_data = load_test_only_bundle(paths, splits)
    device_t = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    _, h, w, c = _infer_bundle_shapes(test_data["chinese"])
    dummy = torch.zeros(1, int(c), int(h), int(w), dtype=torch.float32)

    model_a = build_trainable_backbone(model_type, dummy, train_cfg, init_seed=0)
    model_i = build_trainable_backbone(model_type, dummy, train_cfg, init_seed=0)
    model_a.load_state_dict(load_state_dict(save_dir / f"{model_a_stem}.pt", map_location="cpu"))
    model_i.load_state_dict(load_state_dict(save_dir / f"{model_i_stem}.pt", map_location="cpu"))
    model_a.to(device_t)
    model_i.to(device_t)

    ch = test_data["chinese"]
    ge = test_data["german"]

    def run_block(name: str, block: dict, sensitive_attrs: dict | None) -> dict:
        xa, ya, ida = block["x_test_a"], block["y_test_a"], block["id_test_a"]
        xi, yi, idi = block["x_test_i"], block["y_test_i"], block["id_test_i"]
        pa = predict_positive_proba(model_a, xa, device_t, batch_size=train_cfg.batch_size)
        pi = predict_positive_proba(model_i, xi, device_t, batch_size=train_cfg.batch_size)
        ra = model_eval_by_id(
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
        ri = model_eval_by_id(
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
        rc = combined_vowel_ai_eval(
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
        return {"single_a": ra, "single_i": ri, "combined": rc}

    out = {
        "patient_eval_strategy": patient_eval_strategy,
        "patient_prob_threshold": float(patient_prob_threshold),
        "chinese": run_block("Chinese-Test", ch, ch.get("sensitive_attrs")),
        "german": run_block(
            "German-Test",
            {
                "x_test_a": ge["x_ger_a"],
                "y_test_a": ge["y_ger_a"],
                "id_test_a": ge["id_ger_a"],
                "x_test_i": ge["x_ger_i"],
                "y_test_i": ge["y_ger_i"],
                "id_test_i": ge["id_ger_i"],
            },
            ge.get("sensitive_attrs"),
        ),
        "dev_test_seed": dev_seed,
        "model_a": model_a_stem,
        "model_i": model_i_stem,
    }
    return out
