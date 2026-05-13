#!/usr/bin/env bash
set -euo pipefail

# Run from the repository root on Linux/Ubuntu.
#
# This experiment tests whether the Paper2601 MAE/AST branch can keep most of
# its EENT value while static fusion is restricted to cross-domain stable
# handcrafted features.
#
# Steps:
#   1. Build a filtered static CSV from the 131D table using unlabeled SVD
#      distribution-shift diagnostics.
#   2. Reuse or train Stage 1 MAE checkpoints.
#   3. Train controlled gated Stage 2 with stable static features.
#   4. Evaluate fixed 0.5, diagnostic Youden, and EENT-validation threshold.
#   5. Rebuild the result and domain-shift HTML reports.
#
# Environment overrides:
#   RUN_VARIANTS="local5"                      # or "local1 local5"
#   SOURCE_STATIC_FEATURE_TABLE=paper2601_local_artifacts/paper2601_static_131_features/paper2601_static_131_by_patient_vowel.csv
#   STABLE_STATIC_TABLE=paper2601_local_artifacts/paper2601_stable_static_features/paper2601_stable_static_by_patient_vowel.csv
#   STABLE_MAX_ABS_MEAN_SHIFT_Z=3.0
#   STABLE_MAX_KS=0.75
#   STABLE_MAX_SVD_PCT_ABS_Z_GT3=0.30
#   STABLE_KEEP_CORPUS_SENSITIVE_NAMES=0
#   STAGE1_RUN_DIR=paper2601_splitmae_runs_control_stage1
#   STABLE_RUN_PREFIX=paper2601_splitmae_runs_stable_static
#   MODEL_SIZE=base384
#   INPUT_FDIM=128
#   INPUT_TDIM=259
#   N_CLIENT_BLOCKS=2
#   N_PARTITIONS=5
#   N_GLOBAL_STAGE1=120
#   N_GLOBAL_STAGE2=250
#   N_LOCAL_STAGE1=5
#   BATCH_SIZE=64
#   STATIC_Z_CLIP=3.0
#   STATIC_PROJECTION_DIM=32
#   STATIC_DROPOUT=0.30
#   STATIC_GATE_INIT=0.25
#   MODEL_INIT_SEED=2718
#   DEV_TEST_SEED=8
#   TRAIN_VAL_SEED=100
#   PARTITION_SEED=42
#   DEVICE=cuda
#   EXTRA_ARGS="--pickle-dir-eent ... --pickle-dir-svd ..."

RUN_VARIANTS="${RUN_VARIANTS:-local5}"
SOURCE_STATIC_FEATURE_TABLE="${SOURCE_STATIC_FEATURE_TABLE:-paper2601_local_artifacts/paper2601_static_131_features/paper2601_static_131_by_patient_vowel.csv}"
STABLE_STATIC_TABLE="${STABLE_STATIC_TABLE:-paper2601_local_artifacts/paper2601_stable_static_features/paper2601_stable_static_by_patient_vowel.csv}"
STABLE_SELECTION_JSON="${STABLE_SELECTION_JSON:-paper2601_local_artifacts/paper2601_stable_static_features/paper2601_stable_static_selection.json}"
STABLE_MAX_ABS_MEAN_SHIFT_Z="${STABLE_MAX_ABS_MEAN_SHIFT_Z:-3.0}"
STABLE_MAX_KS="${STABLE_MAX_KS:-0.75}"
STABLE_MAX_SVD_PCT_ABS_Z_GT3="${STABLE_MAX_SVD_PCT_ABS_Z_GT3:-0.30}"
STABLE_KEEP_CORPUS_SENSITIVE_NAMES="${STABLE_KEEP_CORPUS_SENSITIVE_NAMES:-0}"
STAGE1_RUN_DIR="${STAGE1_RUN_DIR:-paper2601_splitmae_runs_control_stage1}"
STABLE_RUN_PREFIX="${STABLE_RUN_PREFIX:-paper2601_splitmae_runs_stable_static}"
MODEL_SIZE="${MODEL_SIZE:-base384}"
INPUT_FDIM="${INPUT_FDIM:-128}"
INPUT_TDIM="${INPUT_TDIM:-259}"
N_CLIENT_BLOCKS="${N_CLIENT_BLOCKS:-2}"
N_PARTITIONS="${N_PARTITIONS:-5}"
N_GLOBAL_STAGE1="${N_GLOBAL_STAGE1:-120}"
N_GLOBAL_STAGE2="${N_GLOBAL_STAGE2:-250}"
N_LOCAL_STAGE1="${N_LOCAL_STAGE1:-5}"
BATCH_SIZE="${BATCH_SIZE:-64}"
STATIC_Z_CLIP="${STATIC_Z_CLIP:-3.0}"
STATIC_PROJECTION_DIM="${STATIC_PROJECTION_DIM:-32}"
STATIC_DROPOUT="${STATIC_DROPOUT:-0.30}"
STATIC_GATE_INIT="${STATIC_GATE_INIT:-0.25}"
MODEL_INIT_SEED="${MODEL_INIT_SEED:-2718}"
DEV_TEST_SEED="${DEV_TEST_SEED:-8}"
TRAIN_VAL_SEED="${TRAIN_VAL_SEED:-100}"
PARTITION_SEED="${PARTITION_SEED:-42}"
DEVICE="${DEVICE:-}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

export PYTHONHASHSEED="${PYTHONHASHSEED:-${MODEL_INIT_SEED}}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

DEVICE_ARGS=()
if [[ -n "${DEVICE}" ]]; then
  DEVICE_ARGS=(--device "${DEVICE}")
fi

COMMON_ARGS=(
  --model-size "${MODEL_SIZE}"
  --input-fdim "${INPUT_FDIM}"
  --input-tdim "${INPUT_TDIM}"
  --n-client-blocks "${N_CLIENT_BLOCKS}"
  --n-partitions "${N_PARTITIONS}"
  --batch-size "${BATCH_SIZE}"
  --model-init-seed "${MODEL_INIT_SEED}"
  --dev-test-seed "${DEV_TEST_SEED}"
  --train-val-seed "${TRAIN_VAL_SEED}"
  --partition-seed "${PARTITION_SEED}"
  "${DEVICE_ARGS[@]}"
)

local_epochs_for_variant() {
  case "$1" in
    local1) printf "1" ;;
    local5) printf "5" ;;
    *)
      if [[ -z "${N_LOCAL_STAGE2:-}" ]]; then
        echo "Set N_LOCAL_STAGE2 for custom RUN_VARIANTS entry '$1'." >&2
        return 1
      fi
      printf "%s" "${N_LOCAL_STAGE2}"
      ;;
  esac
}

ensure_stage1() {
  mkdir -p "${STAGE1_RUN_DIR}"
  local missing=0
  for vowel in a i; do
    if [[ ! -f "${STAGE1_RUN_DIR}/ast_stage1_${vowel}_client.pt" || ! -f "${STAGE1_RUN_DIR}/ast_stage1_${vowel}_server.pt" ]]; then
      missing=1
    fi
  done
  if [[ "${missing}" -eq 0 ]]; then
    echo "Stage 1 checkpoints found in ${STAGE1_RUN_DIR}; reusing them."
    return
  fi

  echo "Stage 1 checkpoints missing; training shared Stage 1 MAE into ${STAGE1_RUN_DIR}."
  for vowel in a i; do
    uv run --no-sync python paper2601_splitmae_cli.py train-stage1 \
      "${COMMON_ARGS[@]}" \
      --vowel "${vowel}" \
      --n-global-rounds "${N_GLOBAL_STAGE1}" \
      --n-local-epochs "${N_LOCAL_STAGE1}" \
      --save-dir "${STAGE1_RUN_DIR}" \
      --run-name "ast_stage1_${vowel}" \
      ${EXTRA_ARGS}
  done
}

build_stable_table() {
  if [[ ! -f "${SOURCE_STATIC_FEATURE_TABLE}" ]]; then
    echo "Missing 131D source table: ${SOURCE_STATIC_FEATURE_TABLE}" >&2
    exit 1
  fi
  local keep_name_args=()
  if [[ "${STABLE_KEEP_CORPUS_SENSITIVE_NAMES}" == "1" ]]; then
    keep_name_args=(--keep-corpus-sensitive-names)
  fi
  echo
  echo "=== Building stable static table ==="
  uv run --no-sync python paper2601_make_stable_static_table.py \
    "${COMMON_ARGS[@]}" \
    --static-feature-source table \
    --static-feature-table "${SOURCE_STATIC_FEATURE_TABLE}" \
    --max-abs-mean-shift-z "${STABLE_MAX_ABS_MEAN_SHIFT_Z}" \
    --max-ks "${STABLE_MAX_KS}" \
    --max-svd-pct-abs-z-gt3 "${STABLE_MAX_SVD_PCT_ABS_Z_GT3}" \
    --out-table "${STABLE_STATIC_TABLE}" \
    --out-json "${STABLE_SELECTION_JSON}" \
    "${keep_name_args[@]}" \
    ${EXTRA_ARGS}
}

evaluate_stable_run() {
  local run_dir="$1"

  echo
  echo "=== Fixed-threshold evaluate: ${run_dir} ==="
  uv run --no-sync python paper2601_controlled_fusion_cli.py evaluate-stage2-pair-controlled \
    "${COMMON_ARGS[@]}" \
    --static-feature-source table \
    --static-feature-table "${STABLE_STATIC_TABLE}" \
    --static-feature-preset all \
    --metadata-a "${run_dir}/ast_stage2_a_metadata.json" \
    --metadata-i "${run_dir}/ast_stage2_i_metadata.json" \
    --eval-dataset both \
    --patient-eval-strategy fixed \
    --patient-prob-threshold 0.5 \
    --results-json "${run_dir}/ast_stage2_ai_eval_fixed.json" \
    --fusion-mode gated \
    --static-projection-dim "${STATIC_PROJECTION_DIM}" \
    --static-dropout "${STATIC_DROPOUT}" \
    --static-gate-init "${STATIC_GATE_INIT}" \
    --static-z-clip "${STATIC_Z_CLIP}" \
    ${EXTRA_ARGS}

  echo
  echo "=== Diagnostic Youden evaluate: ${run_dir} ==="
  uv run --no-sync python paper2601_controlled_fusion_cli.py evaluate-stage2-pair-controlled \
    "${COMMON_ARGS[@]}" \
    --static-feature-source table \
    --static-feature-table "${STABLE_STATIC_TABLE}" \
    --static-feature-preset all \
    --metadata-a "${run_dir}/ast_stage2_a_metadata.json" \
    --metadata-i "${run_dir}/ast_stage2_i_metadata.json" \
    --eval-dataset both \
    --patient-eval-strategy best_threshold \
    --results-json "${run_dir}/ast_stage2_ai_eval_youden_eent_train_scaling.json" \
    --fusion-mode gated \
    --static-projection-dim "${STATIC_PROJECTION_DIM}" \
    --static-dropout "${STATIC_DROPOUT}" \
    --static-gate-init "${STATIC_GATE_INIT}" \
    --static-z-clip "${STATIC_Z_CLIP}" \
    ${EXTRA_ARGS}

  echo
  echo "=== EENT-validation-threshold evaluate: ${run_dir} ==="
  uv run --no-sync python paper2601_eval_eent_val_threshold.py \
    --run-dir "${run_dir}" \
    --eval-dataset both \
    --threshold-metric macro_f1 \
    --results-json "${run_dir}/ast_stage2_ai_eval_eent_val_threshold_macro_f1.json" \
    "${DEVICE_ARGS[@]}" \
    ${EXTRA_ARGS}
}

run_stage2_stable() {
  local variant="$1"
  local n_local_stage2
  n_local_stage2="$(local_epochs_for_variant "${variant}")"
  local run_dir="${STABLE_RUN_PREFIX}_gated_clip_${variant}"
  mkdir -p "${run_dir}"

  echo
  echo "=== Stable-static gated experiment: ${variant} -> ${run_dir} ==="
  echo "stable_table=${STABLE_STATIC_TABLE}"
  echo "local_epochs=${n_local_stage2} z_clip=${STATIC_Z_CLIP} projection=${STATIC_PROJECTION_DIM} dropout=${STATIC_DROPOUT} gate_init=${STATIC_GATE_INIT}"

  for vowel in a i; do
    uv run --no-sync python paper2601_controlled_fusion_cli.py train-stage2-controlled \
      "${COMMON_ARGS[@]}" \
      --static-feature-source table \
      --static-feature-table "${STABLE_STATIC_TABLE}" \
      --static-feature-preset all \
      --vowel "${vowel}" \
      --n-global-rounds "${N_GLOBAL_STAGE2}" \
      --n-local-epochs "${n_local_stage2}" \
      --early-stopping-patience 10 \
      --num-labels 1 \
      --load-client "${STAGE1_RUN_DIR}/ast_stage1_${vowel}_client.pt" \
      --load-server "${STAGE1_RUN_DIR}/ast_stage1_${vowel}_server.pt" \
      --save-dir "${run_dir}" \
      --run-name "ast_stage2_${vowel}" \
      --fusion-mode gated \
      --static-projection-dim "${STATIC_PROJECTION_DIM}" \
      --static-dropout "${STATIC_DROPOUT}" \
      --static-gate-init "${STATIC_GATE_INIT}" \
      --static-z-clip "${STATIC_Z_CLIP}" \
      ${EXTRA_ARGS}
  done

  evaluate_stable_run "${run_dir}"
}

build_stable_table
ensure_stage1

STABLE_RUN_DIRS=()
for variant in ${RUN_VARIANTS}; do
  run_stage2_stable "${variant}"
  STABLE_RUN_DIRS+=("${STABLE_RUN_PREFIX}_gated_clip_${variant}")
done

if [[ -f paper2601_make_results_report.py ]]; then
  echo
  echo "=== Rebuilding global results report ==="
  uv run --no-sync python paper2601_make_results_report.py
fi

if [[ -f paper2601_make_domain_shift_report.py ]]; then
  echo
  echo "=== Rebuilding stable-static domain-shift report ==="
  uv run --no-sync python paper2601_make_domain_shift_report.py \
    --static-feature-source table \
    --static-feature-table "${STABLE_STATIC_TABLE}" \
    --presets all \
    --run-dirs "${STABLE_RUN_DIRS[@]}" \
    --out-html paper2601_stable_static_domain_shift_report.html \
    --out-json paper2601_stable_static_domain_shift_report.json \
    --out-static-tsv paper2601_stable_static_domain_shift.tsv \
    --out-score-tsv paper2601_stable_static_patient_score_shift.tsv \
    ${EXTRA_ARGS}
fi

echo
echo "Stable-static experiment finished."
echo "Stable table: ${STABLE_STATIC_TABLE}"
echo "Selection report: ${STABLE_SELECTION_JSON}"
echo "Run directories:"
printf -- "- %s\n" "${STABLE_RUN_DIRS[@]}"
