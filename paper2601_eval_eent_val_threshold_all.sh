#!/usr/bin/env bash
set -euo pipefail

# Run from repository root on Linux/Ubuntu.
#
# This does not train anything. It re-evaluates finished Stage 2 runs by:
#   1. predicting EENT validation patients,
#   2. selecting one patient-level threshold on EENT validation,
#   3. applying that same threshold to EENT test and SVD.
#
# Defaults:
#   THRESHOLD_METRIC=macro_f1
#   RUN_DIRS unset -> auto-scan paper2601_splitmae_runs*/ directories with
#                     ast_stage2_a_metadata.json and ast_stage2_i_metadata.json
#
# Examples:
#   bash paper2601_eval_eent_val_threshold_all.sh
#   THRESHOLD_METRIC=f1 bash paper2601_eval_eent_val_threshold_all.sh
#   RUN_DIRS="paper2601_splitmae_runs_control_all131_gated_clip_local5 paper2601_splitmae_runs_control_source_tilt_gated_clip_local5" bash paper2601_eval_eent_val_threshold_all.sh

THRESHOLD_METRIC="${THRESHOLD_METRIC:-macro_f1}"
EVAL_DATASET="${EVAL_DATASET:-both}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
MAKE_REPORT="${MAKE_REPORT:-1}"
DEVICE="${DEVICE:-}"

DEVICE_ARGS=()
if [[ -n "${DEVICE}" ]]; then
  DEVICE_ARGS=(--device "${DEVICE}")
fi

discover_run_dirs() {
  local dirs=()
  shopt -s nullglob
  for d in paper2601_splitmae_runs*/; do
    d="${d%/}"
    if [[ -f "${d}/ast_stage2_a_metadata.json" && -f "${d}/ast_stage2_i_metadata.json" ]]; then
      dirs+=("${d}")
    fi
  done
  shopt -u nullglob
  printf "%s\n" "${dirs[@]}"
}

if [[ -n "${RUN_DIRS:-}" ]]; then
  mapfile -t TARGET_DIRS < <(printf "%s\n" ${RUN_DIRS})
else
  mapfile -t TARGET_DIRS < <(discover_run_dirs)
fi

if [[ "${#TARGET_DIRS[@]}" -eq 0 ]]; then
  echo "No Stage 2 run directories found." >&2
  echo "Set RUN_DIRS explicitly or run from the repository root containing paper2601_splitmae_runs*." >&2
  exit 1
fi

echo "EENT validation threshold metric: ${THRESHOLD_METRIC}"
echo "Evaluation dataset: ${EVAL_DATASET}"
echo "Run directories:"
printf -- "- %s\n" "${TARGET_DIRS[@]}"

for run_dir in "${TARGET_DIRS[@]}"; do
  if [[ ! -f "${run_dir}/ast_stage2_a_metadata.json" || ! -f "${run_dir}/ast_stage2_i_metadata.json" ]]; then
    echo "[skip] missing paired Stage 2 metadata: ${run_dir}" >&2
    continue
  fi
  echo
  echo "=== EENT-val threshold evaluate: ${run_dir} ==="
  uv run --no-sync python paper2601_eval_eent_val_threshold.py \
    --run-dir "${run_dir}" \
    --eval-dataset "${EVAL_DATASET}" \
    --threshold-metric "${THRESHOLD_METRIC}" \
    --results-json "${run_dir}/ast_stage2_ai_eval_eent_val_threshold_${THRESHOLD_METRIC}.json" \
    "${DEVICE_ARGS[@]}" \
    ${EXTRA_ARGS}
done

if [[ "${MAKE_REPORT}" == "1" && -f paper2601_make_results_report.py ]]; then
  echo
  echo "=== Rebuilding HTML/TSV report ==="
  uv run --no-sync python paper2601_make_results_report.py
fi

echo
echo "Done. New files are named:"
echo "  */ast_stage2_ai_eval_eent_val_threshold_${THRESHOLD_METRIC}.json"
if [[ "${MAKE_REPORT}" == "1" ]]; then
  echo "  paper2601_results_report.html"
  echo "  paper2601_results_report.tsv"
fi
