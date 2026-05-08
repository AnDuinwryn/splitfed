from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_german_subjects_from_xlsx(xlsx_path: Path) -> tuple[pd.DataFrame, dict, dict]:
    """
    Read SVD subject table from Excel (e.g. metadata/subjects/SVD.xlsx).

    - ``ID``: column ``Keep ID`` if present, else ``ID``, else the first column.
    - ``Class``: column ``Class`` (coerced to int).
    - ``Gender`` / ``Age``: for stratification maps; age groups match ``load_patient_metadata`` (german CSV).
    """
    raw = pd.read_excel(xlsx_path, header=0, engine="openpyxl")
    if raw.shape[1] < 2:
        raise ValueError(f"SVD xlsx needs at least 2 columns, got {raw.shape[1]}")

    if "Keep ID" in raw.columns:
        id_series = raw["Keep ID"]
    elif "ID" in raw.columns:
        id_series = raw["ID"]
    else:
        id_series = raw.iloc[:, 0]

    if "Class" not in raw.columns:
        raise ValueError("SVD xlsx must contain a 'Class' column (first row header).")
    sex_col = "Gender" if "Gender" in raw.columns else None
    age_col = "Age" if "Age" in raw.columns else None
    if sex_col is None or age_col is None:
        raise ValueError("SVD xlsx must contain 'Gender' and 'Age' columns.")

    ids = pd.to_numeric(id_series, errors="coerce")
    cls = pd.to_numeric(raw["Class"], errors="coerce").astype("Int64")
    t = pd.DataFrame(
        {
            "ID": ids,
            "Class": cls,
            "Gender": raw[sex_col],
            "Age": pd.to_numeric(raw[age_col], errors="coerce"),
        }
    )
    t = t[t["ID"].notna() & t["Class"].notna()]
    t = t.dropna(subset=["Age"])
    t["ID"] = t["ID"].astype(int)
    t["Class"] = t["Class"].astype(int)

    if t.empty:
        raise ValueError("No valid SVD subject rows after cleaning (check ID / Class / Age).")

    split_df = t.groupby("ID", as_index=False).agg({"Class": "first"})

    age_group_map: dict = {}
    gender_map: dict = {}
    for pid, g in t.groupby("ID"):
        pid = int(pid)
        row = g.iloc[0]
        age = float(row["Age"])
        gender = row["Gender"]
        if age < 35:
            age_group = 0
        elif 35 <= age <= 50:
            age_group = 1
        else:
            age_group = 2
        age_group_map[pid] = age_group
        gstr = str(gender).strip().lower()
        gender_map[pid] = 0 if gstr == "m" else 1

    return split_df, age_group_map, gender_map


def german_labels_frame_from_paths(paths) -> pd.DataFrame:
    """DataFrame with columns ID (int), Class (int), for ``load_german_tensors``."""
    if paths.german_subjects_xlsx is None:
        raise ValueError("SVD metadata workbook is not set on DataPaths.")
    split_df, _, _ = load_german_subjects_from_xlsx(paths.german_subjects_xlsx)
    return split_df
