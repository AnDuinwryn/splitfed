#!/usr/bin/env bash
set -euo pipefail

# Train Paper2601 Stage 2 classifier variants with pathology-focused
# static-feature subsets, then evaluate each with both:
#   1. the normal EENT-train static normalizer saved in Stage 2 metadata
#   2. diagnostic SVD-independent static scaling
#
# This script reuses already trained Stage 1 MAE checkpoints from the matching
# full-static runs. By default it runs both local1 and local5 variants and
# three feature presets:
#   pathology             -> HNR + jitter + shimmer + CPP-like columns
#   pathology_source_tilt -> pathology + H1-H2/H1-A3 source-tilt columns
#   pathology_voicing     -> source_tilt + voiced/unvoiced stability columns
#
# Environment overrides:
#   RUN_VARIANTS="local1 local5"
#   STATIC_FEATURE_PRESETS="pathology pathology_source_tilt pathology_voicing"
#   MODEL_SIZE=base384
#   INPUT_FDIM=128
#   INPUT_TDIM=259
#   N_CLIENT_BLOCKS=2
#   N_PARTITIONS=5
#   N_GLOBAL_STAGE2=250
#   BATCH_SIZE=64
#   STATIC_FEATURE_TABLE=paper2601_local_artifacts/paper2601_static_131_features/paper2601_static_131_by_patient_vowel.csv
#   RUN_DIR_PREFIX=paper2601_splitmae_runs_ablation
#   MODEL_INIT_SEED=2718
#   DEV_TEST_SEED=8
#   TRAIN_VAL_SEED=100
#   PARTITION_SEED=42
#   DEVICE=cuda
#   EXTRA_ARGS="--pickle-dir-eent ... --pickle-dir-svd ..."

RUN_VARIANTS="${RUN_VARIANTS:-local1 local5}"
STATIC_FEATURE_PRESETS="${STATIC_FEATURE_PRESETS:-${STATIC_FEATURE_PRESET:-pathology pathology_source_tilt pathology_voicing}}"
RUN_DIR_PREFIX="${RUN_DIR_PREFIX:-paper2601_splitmae_runs_ablation}"
MODEL_SIZE="${MODEL_SIZE:-base384}"
INPUT_FDIM="${INPUT_FDIM:-128}"
INPUT_TDIM="${INPUT_TDIM:-259}"
N_CLIENT_BLOCKS="${N_CLIENT_BLOCKS:-2}"
N_PARTITIONS="${N_PARTITIONS:-5}"
N_GLOBAL_STAGE2="${N_GLOBAL_STAGE2:-250}"
BATCH_SIZE="${BATCH_SIZE:-64}"
STATIC_FEATURE_SOURCE="${STATIC_FEATURE_SOURCE:-table}"
STATIC_FEATURE_TABLE="${STATIC_FEATURE_TABLE:-paper2601_local_artifacts/paper2601_static_131_features/paper2601_static_131_by_patient_vowel.csv}"
STATIC_AUDIO_MANIFEST="${STATIC_AUDIO_MANIFEST:-}"
STATIC_AUDIO_ROOT_EENT="${STATIC_AUDIO_ROOT_EENT:-}"
STATIC_AUDIO_ROOT_SVD="${STATIC_AUDIO_ROOT_SVD:-}"
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

STATIC_ARGS=()

build_static_args() {
  local preset="$1"
  STATIC_ARGS=(--static-feature-source "${STATIC_FEATURE_SOURCE}" --static-feature-preset "${preset}")
  if [[ -n "${STATIC_FEATURE_TABLE}" ]]; then
    STATIC_ARGS+=(--static-feature-table "${STATIC_FEATURE_TABLE}")
  fi
  if [[ -n "${STATIC_AUDIO_MANIFEST}" ]]; then
    STATIC_ARGS+=(--static-audio-manifest "${STATIC_AUDIO_MANIFEST}")
  fi
  if [[ -n "${STATIC_AUDIO_ROOT_EENT}" ]]; then
    STATIC_ARGS+=(--static-audio-root-eent "${STATIC_AUDIO_ROOT_EENT}")
  fi
  if [[ -n "${STATIC_AUDIO_ROOT_SVD}" ]]; then
    STATIC_ARGS+=(--static-audio-root-svd "${STATIC_AUDIO_ROOT_SVD}")
  fi
}

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

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "Missing required file: $1" >&2
    echo "Run the matching full pipeline first, or adjust RUN_VARIANTS/source directories." >&2
    exit 1
  fi
}

preset_dir_tag() {
  case "$1" in
    pathology|pathology_voice_quality) printf "pathology22" ;;
    pathology_source_tilt|pathology_plus_source_tilt) printf "pathology_source_tilt" ;;
    pathology_voicing|pathology_plus_voicing) printf "pathology_voicing" ;;
    *) printf "%s" "$1" | tr -c 'A-Za-z0-9_' '_' ;;
  esac
}

run_experiment() {
  local preset="$1"
  local variant="$2"
  local n_local_stage2
  n_local_stage2="$(local_epochs_for_variant "${variant}")"
  local preset_tag
  preset_tag="$(preset_dir_tag "${preset}")"
  local source_run_dir="paper2601_splitmae_runs_${variant}"
  local run_dir="${RUN_DIR_PREFIX}_${preset_tag}_${variant}"

  build_static_args "${preset}"
  mkdir -p "${run_dir}"
  for vowel in a i; do
    require_file "${source_run_dir}/ast_stage1_${vowel}_client.pt"
    require_file "${source_run_dir}/ast_stage1_${vowel}_server.pt"
  done

  local common_args=(
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
    --save-dir "${run_dir}"
    "${DEVICE_ARGS[@]}"
  )

  echo
  echo "=== Static ablation: preset=${preset} variant=${variant} -> ${run_dir} ==="
  echo "Local Stage 2 epochs: ${n_local_stage2}"

  for vowel in a i; do
    uv run --no-sync python paper2601_splitmae_cli.py train-stage2 \
      "${common_args[@]}" \
      "${STATIC_ARGS[@]}" \
      --vowel "${vowel}" \
      --n-global-rounds "${N_GLOBAL_STAGE2}" \
      --n-local-epochs "${n_local_stage2}" \
      --early-stopping-patience 10 \
      --num-labels 1 \
      --load-client "${source_run_dir}/ast_stage1_${vowel}_client.pt" \
      --load-server "${source_run_dir}/ast_stage1_${vowel}_server.pt" \
      --run-name "ast_stage2_${vowel}" \
      ${EXTRA_ARGS}
  done

  uv run --no-sync python paper2601_splitmae_cli.py evaluate-stage2-pair \
    --metadata-a "${run_dir}/ast_stage2_a_metadata.json" \
    --metadata-i "${run_dir}/ast_stage2_i_metadata.json" \
    --eval-dataset both \
    --results-json "${run_dir}/ast_stage2_ai_eval_eent_train_scaling.json" \
    "${STATIC_ARGS[@]}" \
    ${EXTRA_ARGS}

  uv run --no-sync python paper2601_eval_svd_independent_scaling.py \
    --run-dir "${run_dir}" \
    --eval-dataset both \
    --results-name "ast_stage2_ai_eval_svd_independent_static_scaling.json" \
    "${DEVICE_ARGS[@]}" \
    "${STATIC_ARGS[@]}" \
    ${EXTRA_ARGS}
}

for preset in ${STATIC_FEATURE_PRESETS}; do
  for variant in ${RUN_VARIANTS}; do
    run_experiment "${preset}" "${variant}"
  done
done
