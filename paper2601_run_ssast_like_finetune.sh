#!/usr/bin/env bash
set -euo pipefail

# Run from the repository root on Linux/Ubuntu.
#
# Purpose:
#   Fairer finetune ablation for the Paper2601 Stage 1 MAE encoder:
#   mean_patch pooling + LayerNorm Linear(2) + CrossEntropyLoss + Adam + val_loss early stopping.
#   This intentionally removes static features and the attention-FFNN/focal-loss head.

RUN_VARIANTS="${RUN_VARIANTS:-local5}"
STAGE1_RUN_DIR="${STAGE1_RUN_DIR:-paper2601_splitmae_runs_control_stage1}"
SSAST_LIKE_RUN_PREFIX="${SSAST_LIKE_RUN_PREFIX:-paper2601_splitmae_runs_ssast_like}"
MODEL_SIZE="${MODEL_SIZE:-base384}"
INPUT_FDIM="${INPUT_FDIM:-128}"
INPUT_TDIM="${INPUT_TDIM:-259}"
N_CLIENT_BLOCKS="${N_CLIENT_BLOCKS:-2}"
N_PARTITIONS="${N_PARTITIONS:-5}"
N_GLOBAL_STAGE1="${N_GLOBAL_STAGE1:-120}"
N_GLOBAL_STAGE2="${N_GLOBAL_STAGE2:-250}"
N_LOCAL_STAGE1="${N_LOCAL_STAGE1:-5}"
BATCH_SIZE="${BATCH_SIZE:-64}"
MODEL_INIT_SEED="${MODEL_INIT_SEED:-2718}"
DEV_TEST_SEED="${DEV_TEST_SEED:-8}"
TRAIN_VAL_SEED="${TRAIN_VAL_SEED:-100}"
PARTITION_SEED="${PARTITION_SEED:-42}"
CLIENT_LR="${CLIENT_LR:-5e-5}"
SERVER_LR="${SERVER_LR:-5e-5}"
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

evaluate_run() {
  local run_dir="$1"

  echo
  echo "=== SSAST-like fixed-threshold evaluate: ${run_dir} ==="
  uv run --no-sync python paper2601_ssast_like_finetune_cli.py evaluate-stage2-pair \
    "${COMMON_ARGS[@]}" \
    --metadata-a "${run_dir}/ast_stage2_a_metadata.json" \
    --metadata-i "${run_dir}/ast_stage2_i_metadata.json" \
    --eval-dataset both \
    --patient-eval-strategy fixed \
    --patient-prob-threshold 0.5 \
    --results-json "${run_dir}/ast_stage2_ai_eval_fixed.json" \
    ${EXTRA_ARGS}

  echo
  echo "=== SSAST-like diagnostic Youden evaluate: ${run_dir} ==="
  uv run --no-sync python paper2601_ssast_like_finetune_cli.py evaluate-stage2-pair \
    "${COMMON_ARGS[@]}" \
    --metadata-a "${run_dir}/ast_stage2_a_metadata.json" \
    --metadata-i "${run_dir}/ast_stage2_i_metadata.json" \
    --eval-dataset both \
    --patient-eval-strategy best_threshold \
    --results-json "${run_dir}/ast_stage2_ai_eval_youden.json" \
    ${EXTRA_ARGS}

  echo
  echo "=== SSAST-like EENT-validation-threshold evaluate: ${run_dir} ==="
  uv run --no-sync python paper2601_ssast_like_finetune_cli.py evaluate-stage2-pair-eent-val-threshold \
    "${COMMON_ARGS[@]}" \
    --metadata-a "${run_dir}/ast_stage2_a_metadata.json" \
    --metadata-i "${run_dir}/ast_stage2_i_metadata.json" \
    --eval-dataset both \
    --threshold-metric macro_f1 \
    --results-json "${run_dir}/ast_stage2_ai_eval_eent_val_threshold_macro_f1.json" \
    ${EXTRA_ARGS}
}

run_stage2_ssast_like() {
  local variant="$1"
  local n_local_stage2
  n_local_stage2="$(local_epochs_for_variant "${variant}")"
  local run_dir="${SSAST_LIKE_RUN_PREFIX}_${variant}"
  mkdir -p "${run_dir}"

  echo
  echo "=== SSAST-like Paper2601 finetune: ${variant} -> ${run_dir} ==="
  echo "head=mean_patch+LayerNorm+Linear(2) loss=CrossEntropy optimizer=Adam local_epochs=${n_local_stage2}"

  for vowel in a i; do
    uv run --no-sync python paper2601_ssast_like_finetune_cli.py train-stage2-ssast-like \
      "${COMMON_ARGS[@]}" \
      --vowel "${vowel}" \
      --n-global-rounds "${N_GLOBAL_STAGE2}" \
      --n-local-epochs "${n_local_stage2}" \
      --early-stopping-patience 10 \
      --client-lr "${CLIENT_LR}" \
      --server-lr "${SERVER_LR}" \
      --load-client "${STAGE1_RUN_DIR}/ast_stage1_${vowel}_client.pt" \
      --load-server "${STAGE1_RUN_DIR}/ast_stage1_${vowel}_server.pt" \
      --save-dir "${run_dir}" \
      --run-name "ast_stage2_${vowel}" \
      ${EXTRA_ARGS}
  done

  evaluate_run "${run_dir}"
}

ensure_stage1

for variant in ${RUN_VARIANTS}; do
  run_stage2_ssast_like "${variant}"
done

if [[ -f paper2601_make_results_report.py ]]; then
  echo
  echo "=== Rebuilding global results report ==="
  uv run --no-sync python paper2601_make_results_report.py
fi

echo
echo "SSAST-like Paper2601 finetune finished."
echo "Run directories:"
for variant in ${RUN_VARIANTS}; do
  echo "- ${SSAST_LIKE_RUN_PREFIX}_${variant}"
done
