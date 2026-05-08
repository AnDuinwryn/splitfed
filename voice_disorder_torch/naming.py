from __future__ import annotations


def generate_model_name(
    model_type: str,
    vowel_type: str,
    dev_test_seed: int,
    train_val_seed: int,
    model_init_seed: int,
) -> str:
    """Centralized checkpoint stem: ``{type}_{vowel}_d{dev}_t{train}_i{init}``."""
    return f"{model_type}_{vowel_type}_d{dev_test_seed}_t{train_val_seed}_i{model_init_seed}"


def generate_split_cnn_stem(
    vowel_type: str,
    dev_test_seed: int,
    train_val_seed: int,
    model_init_seed: int,
) -> str:
    return f"split_cnn_{vowel_type}_d{dev_test_seed}_t{train_val_seed}_i{model_init_seed}"


def generate_split_ssast_stem(
    vowel_type: str,
    dev_test_seed: int,
    train_val_seed: int,
    model_init_seed: int,
    n_client_blocks: int,
) -> str:
    return (
        f"split_ssast_{vowel_type}_d{dev_test_seed}_t{train_val_seed}_i{model_init_seed}_b{n_client_blocks}"
    )


def parse_model_seeds_from_name(model_name: str) -> dict | None:
    """Parse seeds from compact stems only (centralized, split CNN, split SSAST)."""
    try:
        parts = model_name.split("_")
        if (
            len(parts) == 7
            and parts[0] == "split"
            and parts[1] == "ssast"
            and parts[3].startswith("d")
            and parts[4].startswith("t")
            and parts[5].startswith("i")
            and parts[6].startswith("b")
        ):
            return {
                "dev_test_seed": int(parts[3][1:]),
                "train_val_seed": int(parts[4][1:]),
                "model_init_seed": int(parts[5][1:]),
            }
        if len(parts) < 5:
            return None
        if parts[-4] not in ("a", "i"):
            return None
        d_part, t_part, i_part = parts[-3], parts[-2], parts[-1]
        if not (d_part.startswith("d") and t_part.startswith("t") and i_part.startswith("i")):
            return None
        return {
            "dev_test_seed": int(d_part[1:]),
            "train_val_seed": int(t_part[1:]),
            "model_init_seed": int(i_part[1:]),
        }
    except (ValueError, IndexError):
        return None
