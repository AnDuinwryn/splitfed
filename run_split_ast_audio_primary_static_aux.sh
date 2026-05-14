#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

OUTPUT_DIR="${OUTPUT_DIR:-outputs/audio_primary_static_aux}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-2718}"

uv run --no-sync python scripts/final_pipeline.py \
  --output-dir "$OUTPUT_DIR" \
  --models split_ast_audio_primary_stable_static \
  --seeds "$SEED" \
  --device "$DEVICE" \
  --stage1-global-rounds "${STAGE1_GLOBAL_ROUNDS:-120}" \
  --stage1-local-epochs "${STAGE1_LOCAL_EPOCHS:-5}" \
  --stage2-global-rounds "${STAGE2_GLOBAL_ROUNDS:-250}" \
  --stage2-local-epochs "${STAGE2_LOCAL_EPOCHS:-5}" \
  --ast-batch-size "${AST_BATCH_SIZE:-64}" \
  --audio-primary-static-projection-dim "${STATIC_PROJECTION_DIM:-32}" \
  --audio-primary-static-dropout "${STATIC_DROPOUT:-0.35}" \
  --audio-primary-static-gate-init "${STATIC_GATE_INIT:-0.20}" \
  --audio-primary-static-max-weight "${STATIC_MAX_WEIGHT:-0.35}" \
  --audio-primary-static-anomaly-threshold "${STATIC_ANOMALY_THRESHOLD:-2.5}" \
  --audio-primary-static-anomaly-scale "${STATIC_ANOMALY_SCALE:-1.0}" \
  --audio-primary-static-aux-hidden-dim "${STATIC_AUX_HIDDEN_DIM:-64}" \
  "$@"
