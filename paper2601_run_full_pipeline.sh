#!/usr/bin/env bash
set -euo pipefail

# Run from the repository root on Linux/Ubuntu.
#
# Environment overrides:
#   RUN_DIR=paper2601_splitmae_runs
#   MODEL_SIZE=base384
#   INPUT_FDIM=128
#   INPUT_TDIM=259
#   N_CLIENT_BLOCKS=2
#   N_PARTITIONS=5
#   N_GLOBAL_STAGE1=120
#   N_GLOBAL_STAGE2=250
#   N_LOCAL_EPOCHS=1
#   BATCH_SIZE=256
#   MODEL_INIT_SEED=2718
#   DEV_TEST_SEED=8
#   TRAIN_VAL_SEED=100
#   PARTITION_SEED=42
#   DEVICE=cuda
#   EXTRA_ARGS="--pickle-dir-eent ... --pickle-dir-svd ..."

RUN_DIR="${RUN_DIR:-paper2601_splitmae_runs}"
MODEL_SIZE="${MODEL_SIZE:-base384}"
INPUT_FDIM="${INPUT_FDIM:-128}"
INPUT_TDIM="${INPUT_TDIM:-259}"
N_CLIENT_BLOCKS="${N_CLIENT_BLOCKS:-2}"
N_PARTITIONS="${N_PARTITIONS:-5}"
N_GLOBAL_STAGE1="${N_GLOBAL_STAGE1:-120}"
N_GLOBAL_STAGE2="${N_GLOBAL_STAGE2:-250}"
N_LOCAL_EPOCHS="${N_LOCAL_EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-256}"
MODEL_INIT_SEED="${MODEL_INIT_SEED:-2718}"
DEV_TEST_SEED="${DEV_TEST_SEED:-8}"
TRAIN_VAL_SEED="${TRAIN_VAL_SEED:-100}"
PARTITION_SEED="${PARTITION_SEED:-42}"
DEVICE="${DEVICE:-}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

mkdir -p "${RUN_DIR}"
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
  --n-local-epochs "${N_LOCAL_EPOCHS}"
  --save-dir "${RUN_DIR}"
  "${DEVICE_ARGS[@]}"
)

run_train_stage1() {
  local vowel="$1"
  local run_name="ast_stage1_${vowel}"
  uv run --no-sync python paper2601_splitmae_cli.py train-stage1 \
    "${COMMON_ARGS[@]}" \
    --vowel "${vowel}" \
    --n-global-rounds "${N_GLOBAL_STAGE1}" \
    --run-name "${run_name}" \
    ${EXTRA_ARGS} \
    | tee "${RUN_DIR}/${run_name}.log"
}

run_train_stage2() {
  local vowel="$1"
  local run_name="ast_stage2_${vowel}"
  uv run --no-sync python paper2601_splitmae_cli.py train-stage2 \
    "${COMMON_ARGS[@]}" \
    --vowel "${vowel}" \
    --n-global-rounds "${N_GLOBAL_STAGE2}" \
    --early-stopping-patience 10 \
    --num-labels 1 \
    --load-client "${RUN_DIR}/ast_stage1_${vowel}_client.pt" \
    --load-server "${RUN_DIR}/ast_stage1_${vowel}_server.pt" \
    --run-name "${run_name}" \
    ${EXTRA_ARGS} \
    | tee "${RUN_DIR}/${run_name}.log"
}

run_train_stage1 a
run_train_stage2 a
run_train_stage1 i
run_train_stage2 i

uv run --no-sync python paper2601_splitmae_cli.py evaluate-stage2-pair \
  --metadata-a "${RUN_DIR}/ast_stage2_a_metadata.json" \
  --metadata-i "${RUN_DIR}/ast_stage2_i_metadata.json" \
  --eval-dataset both \
  --results-json "${RUN_DIR}/ast_stage2_ai_eval.json" \
  ${EXTRA_ARGS} \
  | tee "${RUN_DIR}/ast_stage2_ai_eval.log"
