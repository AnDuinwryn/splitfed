#!/usr/bin/env bash
set -euo pipefail

# Run from the repository root on Linux/Ubuntu.
#
# Purpose:
#   1. MAE-only: verify whether the MAE/AST branch has independent signal.
#   2. Static-only: verify how much the 131D handcrafted vector explains alone.
#   3. Controlled all131 fusion: keep all 131D capacity but reduce static-feature
#      dominance through z-score clipping, projection, dropout, and a learnable gate.
#   4. Controlled pathology/source_tilt fusion: compare with lower-bias static subsets.
#
# This script reuses Stage 1 checkpoints. If they are missing, it trains Stage 1
# once into STAGE1_RUN_DIR before running Stage 2 variants.
#
# Environment overrides:
#   RUN_VARIANTS="local5"                    # or "local1 local5"
#   STAGE1_RUN_DIR=paper2601_splitmae_runs_control_stage1
#   CONTROL_RUN_PREFIX=paper2601_splitmae_runs_control
#   MODEL_SIZE=base384
#   INPUT_FDIM=128
#   INPUT_TDIM=259
#   N_CLIENT_BLOCKS=2
#   N_PARTITIONS=5
#   N_GLOBAL_STAGE1=120
#   N_GLOBAL_STAGE2=250
#   N_LOCAL_STAGE1=5
#   BATCH_SIZE=64
#   STATIC_FEATURE_TABLE=paper2601_local_artifacts/paper2601_static_131_features/paper2601_static_131_by_patient_vowel.csv
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
STAGE1_RUN_DIR="${STAGE1_RUN_DIR:-paper2601_splitmae_runs_control_stage1}"
CONTROL_RUN_PREFIX="${CONTROL_RUN_PREFIX:-paper2601_splitmae_runs_control}"
MODEL_SIZE="${MODEL_SIZE:-base384}"
INPUT_FDIM="${INPUT_FDIM:-128}"
INPUT_TDIM="${INPUT_TDIM:-259}"
N_CLIENT_BLOCKS="${N_CLIENT_BLOCKS:-2}"
N_PARTITIONS="${N_PARTITIONS:-5}"
N_GLOBAL_STAGE1="${N_GLOBAL_STAGE1:-120}"
N_GLOBAL_STAGE2="${N_GLOBAL_STAGE2:-250}"
N_LOCAL_STAGE1="${N_LOCAL_STAGE1:-5}"
BATCH_SIZE="${BATCH_SIZE:-64}"
STATIC_FEATURE_TABLE="${STATIC_FEATURE_TABLE:-paper2601_local_artifacts/paper2601_static_131_features/paper2601_static_131_by_patient_vowel.csv}"
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

require_static_table() {
  if [[ ! -f "${STATIC_FEATURE_TABLE}" ]]; then
    echo "Missing static feature table: ${STATIC_FEATURE_TABLE}" >&2
    echo "Generate or copy the 131D table before running controlled static experiments." >&2
    exit 1
  fi
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

run_stage2_controlled() {
  local variant="$1"
  local run_tag="$2"
  local fusion_mode="$3"
  local static_source="$4"
  local static_preset="$5"
  local projection_dim="$6"
  local dropout="$7"
  local gate_init="$8"
  local z_clip="$9"

  local n_local_stage2
  n_local_stage2="$(local_epochs_for_variant "${variant}")"
  local run_dir="${CONTROL_RUN_PREFIX}_${run_tag}_${variant}"
  mkdir -p "${run_dir}"

  echo
  echo "=== Controlled experiment: ${run_tag} ${variant} -> ${run_dir} ==="
  echo "fusion=${fusion_mode} static_source=${static_source} preset=${static_preset} local_epochs=${n_local_stage2}"

  local static_args=(--static-feature-source "${static_source}" --static-feature-preset "${static_preset}")
  if [[ "${static_source}" != "none" ]]; then
    static_args+=(--static-feature-table "${STATIC_FEATURE_TABLE}")
  fi

  for vowel in a i; do
    uv run --no-sync python paper2601_controlled_fusion_cli.py train-stage2-controlled \
      "${COMMON_ARGS[@]}" \
      "${static_args[@]}" \
      --vowel "${vowel}" \
      --n-global-rounds "${N_GLOBAL_STAGE2}" \
      --n-local-epochs "${n_local_stage2}" \
      --early-stopping-patience 10 \
      --num-labels 1 \
      --load-client "${STAGE1_RUN_DIR}/ast_stage1_${vowel}_client.pt" \
      --load-server "${STAGE1_RUN_DIR}/ast_stage1_${vowel}_server.pt" \
      --save-dir "${run_dir}" \
      --run-name "ast_stage2_${vowel}" \
      --fusion-mode "${fusion_mode}" \
      --static-projection-dim "${projection_dim}" \
      --static-dropout "${dropout}" \
      --static-gate-init "${gate_init}" \
      --static-z-clip "${z_clip}" \
      ${EXTRA_ARGS}
  done

  uv run --no-sync python paper2601_controlled_fusion_cli.py evaluate-stage2-pair-controlled \
    "${COMMON_ARGS[@]}" \
    "${static_args[@]}" \
    --metadata-a "${run_dir}/ast_stage2_a_metadata.json" \
    --metadata-i "${run_dir}/ast_stage2_i_metadata.json" \
    --eval-dataset both \
    --patient-eval-strategy fixed \
    --patient-prob-threshold 0.5 \
    --results-json "${run_dir}/ast_stage2_ai_eval_fixed.json" \
    --fusion-mode "${fusion_mode}" \
    --static-projection-dim "${projection_dim}" \
    --static-dropout "${dropout}" \
    --static-gate-init "${gate_init}" \
    --static-z-clip "${z_clip}" \
    ${EXTRA_ARGS}
}

ensure_stage1
require_static_table

for variant in ${RUN_VARIANTS}; do
  run_stage2_controlled "${variant}" "mae_only" "audio_only" "none" "all" 0 0.0 "${STATIC_GATE_INIT}" 0
  run_stage2_controlled "${variant}" "static_only_all131" "static_only" "table" "all" 0 0.0 "${STATIC_GATE_INIT}" "${STATIC_Z_CLIP}"
  run_stage2_controlled "${variant}" "all131_gated_clip" "gated" "table" "all" "${STATIC_PROJECTION_DIM}" "${STATIC_DROPOUT}" "${STATIC_GATE_INIT}" "${STATIC_Z_CLIP}"
  run_stage2_controlled "${variant}" "pathology22_gated_clip" "gated" "table" "pathology" "${STATIC_PROJECTION_DIM}" "${STATIC_DROPOUT}" "${STATIC_GATE_INIT}" "${STATIC_Z_CLIP}"
  run_stage2_controlled "${variant}" "source_tilt_gated_clip" "gated" "table" "pathology_source_tilt" "${STATIC_PROJECTION_DIM}" "${STATIC_DROPOUT}" "${STATIC_GATE_INIT}" "${STATIC_Z_CLIP}"
done

echo
echo "Controlled experiments finished."
echo "Result roots:"
for variant in ${RUN_VARIANTS}; do
  for tag in mae_only static_only_all131 all131_gated_clip pathology22_gated_clip source_tilt_gated_clip; do
    echo "- ${CONTROL_RUN_PREFIX}_${tag}_${variant}/ast_stage2_ai_eval_fixed.json"
  done
done
