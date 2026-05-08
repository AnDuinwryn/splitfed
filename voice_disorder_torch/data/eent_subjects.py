from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..config import DataPaths


def load_eent_subjects_from_xlsx(xlsx_path: Path) -> tuple[pd.DataFrame, dict, dict]:
    """
    Read EENT subject table from Excel.

    - Uses the **first row as column headers** (header=0); the header row is not dropped as data.
    - Patient ``ID`` is taken from **``Final Random ID``** (the usual second column; first column
      e.g. ``Model Development`` is not used), or if that header is missing, from **column index 1**.
    - ``Class``: 0 if Diagnosis is Normal (exact string after strip), else 1.
    - Rows whose ID is missing, empty, or the literal ``ID`` (spurious header row duplicated as data) are dropped.
    """
    raw = pd.read_excel(xlsx_path, header=0, engine="openpyxl")
    if raw.shape[1] < 2:
        raise ValueError(f"EENT xlsx needs at least 2 columns, got {raw.shape[1]}")

    if "Final Random ID" in raw.columns:
        id_series = raw["Final Random ID"]
    else:
        id_series = raw.iloc[:, 1]

    diag_col = "Diagnosis" if "Diagnosis" in raw.columns else None
    if diag_col is None:
        raise ValueError("EENT xlsx must contain a 'Diagnosis' column (first row header).")

    sex_col = "Sex" if "Sex" in raw.columns else None
    age_col = "Age" if "Age" in raw.columns else None
    if sex_col is None or age_col is None:
        raise ValueError("EENT xlsx must contain 'Sex' and 'Age' columns.")

    ids = id_series.astype(str).str.strip()
    diagnosis = raw[diag_col].astype(str).str.strip()
    cls = np.where(diagnosis.eq("Normal"), 0, 1).astype(int)

    t = pd.DataFrame(
        {
            "ID": ids,
            "Class": cls,
            "Gender": raw[sex_col],
            "Age": pd.to_numeric(raw[age_col], errors="coerce"),
        }
    )
    t = t[t["ID"].notna() & t["ID"].ne("") & t["ID"].ne("ID")]
    t = t.dropna(subset=["Age"])

    if t.empty:
        raise ValueError("No valid EENT subject rows after cleaning (check ID / Age columns).")

    split_df = t.groupby("ID", as_index=False).agg({"Class": "first"})
    split_df["ID"] = split_df["ID"].astype(str)

    age_group_map: dict = {}
    gender_map: dict = {}
    for pid, g in t.groupby("ID"):
        pid = str(pid)
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
        gender_map[pid] = 0 if str(gender).lower() == "m" else 1

    return split_df, age_group_map, gender_map


def resolve_chinese_subject_tables(paths: DataPaths) -> tuple[pd.DataFrame, dict, dict]:
    """Build ``split_df`` and stratification maps from the EENT subject workbook."""
    if paths.eent_subjects_xlsx is None:
        raise ValueError("EENT metadata workbook is not set on DataPaths.")
    return load_eent_subjects_from_xlsx(paths.eent_subjects_xlsx)
