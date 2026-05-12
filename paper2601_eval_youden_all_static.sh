#!/usr/bin/env bash
set -euo pipefail

# Evaluate all trained Paper2601 static-feature variants with the Youden J
# threshold. In the current project evaluator this is exposed as
# --patient-eval-strategy best_threshold, implemented as argmax(TPR - FPR).
#
# This is a diagnostic thresholding protocol because the threshold is selected
# on the evaluated dataset labels. It is useful for comparing ranking quality
# and calibration sensitivity, not as the primary fixed-threshold deployment
# metric.
#
# Default matrix:
#   static groups: all131, pathology22, pathology_source_tilt, pathology_voicing
#   local variants: local1, local5
#   scaling policies:
#     1. EENT-train static normalizer from Stage 2 metadata
#     2. diagnostic SVD-independent static scaling
#
# Environment overrides:
#   STATIC_GROUPS="all131 pathology22 pathology_source_tilt pathology_voicing"
#   RUN_VARIANTS="local1 local5"
#   EVAL_DATASET=both
#   STATIC_FEATURE_TABLE=paper2601_local_artifacts/paper2601_static_131_features/paper2601_static_131_by_patient_vowel.csv
#   SUMMARY_DIR=paper2601_splitmae_runs_youden_summary
#   DEVICE=cuda
#   EXTRA_ARGS="--pickle-dir-eent ... --pickle-dir-svd ..."

STATIC_GROUPS="${STATIC_GROUPS:-all131 pathology22 pathology_source_tilt pathology_voicing}"
RUN_VARIANTS="${RUN_VARIANTS:-local1 local5}"
EVAL_DATASET="${EVAL_DATASET:-both}"
STATIC_FEATURE_SOURCE="${STATIC_FEATURE_SOURCE:-table}"
STATIC_FEATURE_TABLE="${STATIC_FEATURE_TABLE:-paper2601_local_artifacts/paper2601_static_131_features/paper2601_static_131_by_patient_vowel.csv}"
SUMMARY_DIR="${SUMMARY_DIR:-paper2601_splitmae_runs_youden_summary}"
DEVICE="${DEVICE:-}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

mkdir -p "${SUMMARY_DIR}"
MANIFEST="${SUMMARY_DIR}/youden_eval_manifest.tsv"
SUMMARY_TSV="${SUMMARY_DIR}/youden_eval_summary.tsv"
SUMMARY_JSON="${SUMMARY_DIR}/youden_eval_summary.json"

printf "static_group\tlocal_variant\tscaling\trun_dir\tjson_path\n" > "${MANIFEST}"

DEVICE_ARGS=()
if [[ -n "${DEVICE}" ]]; then
  DEVICE_ARGS=(--device "${DEVICE}")
fi

STATIC_ARGS=()
build_static_args() {
  local preset="$1"
  STATIC_ARGS=(--static-feature-source "${STATIC_FEATURE_SOURCE}" --static-feature-preset "${preset}")
  if [[ -n "${STATIC_FEATURE_TABLE}" ]]; then
    STATIC_ARGS+=(--static-feature-table "${STATIC_FEATURE_TABLE}")
  fi
}

group_run_dir() {
  local group="$1"
  local variant="$2"
  case "${group}" in
    all131) printf "paper2601_splitmae_runs_%s" "${variant}" ;;
    pathology22|pathology|pathology_voice_quality) printf "paper2601_splitmae_runs_ablation_pathology22_%s" "${variant}" ;;
    pathology_source_tilt|pathology_plus_source_tilt) printf "paper2601_splitmae_runs_ablation_pathology_source_tilt_%s" "${variant}" ;;
    pathology_voicing|pathology_plus_voicing) printf "paper2601_splitmae_runs_ablation_pathology_voicing_%s" "${variant}" ;;
    *)
      echo "Unknown STATIC_GROUPS entry: ${group}" >&2
      return 1
      ;;
  esac
}

group_preset() {
  local group="$1"
  case "${group}" in
    all131) printf "all" ;;
    pathology22|pathology|pathology_voice_quality) printf "pathology" ;;
    pathology_source_tilt|pathology_plus_source_tilt) printf "pathology_source_tilt" ;;
    pathology_voicing|pathology_plus_voicing) printf "pathology_voicing" ;;
    *)
      echo "Unknown STATIC_GROUPS entry: ${group}" >&2
      return 1
      ;;
  esac
}

require_pair_metadata() {
  local run_dir="$1"
  if [[ ! -f "${run_dir}/ast_stage2_a_metadata.json" || ! -f "${run_dir}/ast_stage2_i_metadata.json" ]]; then
    echo "Missing Stage 2 metadata in ${run_dir}" >&2
    echo "Expected ast_stage2_a_metadata.json and ast_stage2_i_metadata.json." >&2
    exit 1
  fi
}

run_youden_eval() {
  local group="$1"
  local variant="$2"
  local run_dir
  local preset
  run_dir="$(group_run_dir "${group}" "${variant}")"
  preset="$(group_preset "${group}")"
  require_pair_metadata "${run_dir}"
  build_static_args "${preset}"

  local eent_scaling_json="${run_dir}/ast_stage2_ai_eval_youden_eent_train_scaling.json"
  local svd_independent_json_name="ast_stage2_ai_eval_youden_svd_independent_static_scaling.json"
  local svd_independent_json="${run_dir}/${svd_independent_json_name}"

  echo
  echo "=== Youden evaluate: group=${group} preset=${preset} variant=${variant} scaling=eent_train ==="
  uv run --no-sync python paper2601_splitmae_cli.py evaluate-stage2-pair \
    --metadata-a "${run_dir}/ast_stage2_a_metadata.json" \
    --metadata-i "${run_dir}/ast_stage2_i_metadata.json" \
    --eval-dataset "${EVAL_DATASET}" \
    --patient-eval-strategy best_threshold \
    --results-json "${eent_scaling_json}" \
    "${DEVICE_ARGS[@]}" \
    "${STATIC_ARGS[@]}" \
    ${EXTRA_ARGS}
  printf "%s\t%s\t%s\t%s\t%s\n" "${group}" "${variant}" "eent_train_scaling" "${run_dir}" "${eent_scaling_json}" >> "${MANIFEST}"

  echo
  echo "=== Youden evaluate: group=${group} preset=${preset} variant=${variant} scaling=svd_independent ==="
  uv run --no-sync python paper2601_eval_svd_independent_scaling.py \
    --run-dir "${run_dir}" \
    --eval-dataset "${EVAL_DATASET}" \
    --patient-eval-strategy best_threshold \
    --results-name "${svd_independent_json_name}" \
    "${DEVICE_ARGS[@]}" \
    "${STATIC_ARGS[@]}" \
    ${EXTRA_ARGS}
  printf "%s\t%s\t%s\t%s\t%s\n" "${group}" "${variant}" "svd_independent_scaling" "${run_dir}" "${svd_independent_json}" >> "${MANIFEST}"
}

for group in ${STATIC_GROUPS}; do
  for variant in ${RUN_VARIANTS}; do
    run_youden_eval "${group}" "${variant}"
  done
done

uv run --no-sync python - "${MANIFEST}" "${SUMMARY_TSV}" "${SUMMARY_JSON}" <<'PY'
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

manifest = Path(sys.argv[1])
summary_tsv = Path(sys.argv[2])
summary_json = Path(sys.argv[3])

rows = []
with manifest.open("r", encoding="utf-8", newline="") as f:
    reader = csv.DictReader(f, delimiter="\t")
    for item in reader:
        path = Path(item["json_path"])
        if not path.is_file():
            raise SystemExit(f"Missing evaluation JSON: {path}")
        data = json.loads(path.read_text(encoding="utf-8"))
        for dataset, section in sorted((data.get("datasets") or {}).items()):
            combined = section.get("combined") or {}
            metrics = combined.get("metrics") or {}
            cm = combined.get("confusion_matrix") or metrics.get("confusion_matrix") or [[None, None], [None, None]]
            try:
                tn, fp = cm[0]
                fn, tp = cm[1]
            except Exception:
                tn = fp = fn = tp = None
            row = {
                "static_group": item["static_group"],
                "local_variant": item["local_variant"],
                "scaling": item["scaling"],
                "dataset": dataset,
                "youden_threshold": combined.get("optimal_threshold"),
                "accuracy": metrics.get("accuracy"),
                "precision": metrics.get("precision"),
                "recall": metrics.get("recall"),
                "specificity": metrics.get("specificity"),
                "f1": metrics.get("f1_score"),
                "auc": combined.get("auc", combined.get("roc_auc")),
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "tp": tp,
                "json_path": str(path),
            }
            rows.append(row)

columns = [
    "static_group",
    "local_variant",
    "scaling",
    "dataset",
    "youden_threshold",
    "accuracy",
    "precision",
    "recall",
    "specificity",
    "f1",
    "auc",
    "tn",
    "fp",
    "fn",
    "tp",
    "json_path",
]
with summary_tsv.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=columns, delimiter="\t")
    writer.writeheader()
    writer.writerows(rows)
summary_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")

def fmt(value):
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)

print()
print(f"Wrote Youden summary TSV: {summary_tsv.resolve()}")
print(f"Wrote Youden summary JSON: {summary_json.resolve()}")
print()
print("Combined /a/+/i/ Youden summary:")
headers = ["static", "local", "scaling", "dataset", "thr", "acc", "sens", "spec", "f1", "auc", "CM"]
table = []
for row in rows:
    cm_text = f"[{row['tn']} {row['fp']}; {row['fn']} {row['tp']}]"
    table.append([
        row["static_group"],
        row["local_variant"],
        row["scaling"].replace("_scaling", ""),
        row["dataset"],
        fmt(row["youden_threshold"]),
        fmt(row["accuracy"]),
        fmt(row["recall"]),
        fmt(row["specificity"]),
        fmt(row["f1"]),
        fmt(row["auc"]),
        cm_text,
    ])
widths = [len(h) for h in headers]
for r in table:
    for idx, cell in enumerate(r):
        widths[idx] = max(widths[idx], len(str(cell)))
print("  ".join(h.ljust(widths[i]) for i, h in enumerate(headers)))
print("  ".join("-" * widths[i] for i in range(len(headers))))
for r in table:
    print("  ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(r)))
PY
