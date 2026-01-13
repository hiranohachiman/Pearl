#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SAVE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORKSPACE_ROOT="$(cd "${SAVE_ROOT}/.." && pwd)"
DATA_ROOT="${WORKSPACE_ROOT}/data"
IMAGES_DIR="${DATA_ROOT}/images"
PIPELINE_DIR="${SAVE_ROOT}/pipelines"
FEATURE_ROOT="${WORKSPACE_ROOT}/features"
export PYTHONPATH="${SAVE_ROOT}:${PYTHONPATH:-}"

BEIT3_ENV="${SAVE_ROOT}/envs/beit3"
BLIP2_ENV="${SAVE_ROOT}/envs/blip2"
CLIPPP_ENV="${SAVE_ROOT}/envs/clippp"
STELLA_ENV="${SAVE_ROOT}/envs/stella"
CHECKPOINT_ROOT="${SAVE_ROOT}/checkpoints"

DEFAULT_BEIT3_TOKENIZER="${CHECKPOINT_ROOT}/beit3.spm"
DEFAULT_BEIT3_CHECKPOINT="${CHECKPOINT_ROOT}/beit3_base_itc_patch16_224.pth"
DEFAULT_CLIPPP_CHECKPOINT="${CHECKPOINT_ROOT}/PAC++_clip_ViT-L-14.pth"
DEFAULT_STELLA_MODEL_PATH="${CHECKPOINT_ROOT}/stella_en_400M_v5"

BEIT3_TOKENIZER="${BEIT3_TOKENIZER:-${DEFAULT_BEIT3_TOKENIZER}}"
BEIT3_CHECKPOINT="${BEIT3_CHECKPOINT:-${DEFAULT_BEIT3_CHECKPOINT}}"
CLIPPP_CHECKPOINT="${CLIPPP_CHECKPOINT:-${DEFAULT_CLIPPP_CHECKPOINT}}"
STELLA_MODEL_PATH="${STELLA_MODEL_PATH:-${DEFAULT_STELLA_MODEL_PATH}}"

STELLA_VECTOR_DIM="${STELLA_VECTOR_DIM:-768}"
STELLA_DENSE_SUBDIR="${STELLA_DENSE_SUBDIR:-2_Dense_${STELLA_VECTOR_DIM}}"

echo "[INFO] save_features root: ${SAVE_ROOT}"
echo "[INFO] data root: ${DATA_ROOT}"
echo "[INFO] images dir: ${IMAGES_DIR}"
echo "[INFO] feature output root: ${FEATURE_ROOT}"

run_poetry_job() {
  local env_dir="$1"
  shift
  pushd "${env_dir}" >/dev/null
  poetry install --sync
  poetry run "$@"
  popd >/dev/null
}

echo ""
echo "[STEP] Extracting BEiT3 features"
run_poetry_job "${BEIT3_ENV}" \
  python "${PIPELINE_DIR}/save_beit3_spica_features.py" \
  --data-root "${DATA_ROOT}" \
  --images-dir "${IMAGES_DIR}" \
  --output-root "${FEATURE_ROOT}/beit3" \
  --tokenizer-path "${BEIT3_TOKENIZER}" \
  --checkpoint-path "${BEIT3_CHECKPOINT}"

echo ""
echo "[STEP] Extracting BLIP2 features"
run_poetry_job "${BLIP2_ENV}" \
  python "${PIPELINE_DIR}/save_blip2_spica_features.py" \
  --data-root "${DATA_ROOT}" \
  --images-dir "${IMAGES_DIR}" \
  --output-root "${FEATURE_ROOT}/blip2"

echo ""
echo "[STEP] Extracting CLIP++ features"
run_poetry_job "${CLIPPP_ENV}" \
  python "${PIPELINE_DIR}/save_clippp_spica_features.py" \
  --data-root "${DATA_ROOT}" \
  --images-dir "${IMAGES_DIR}" \
  --output-root "${FEATURE_ROOT}/clippp" \
  --checkpoint-path "${CLIPPP_CHECKPOINT}"

echo ""
echo "[STEP] Extracting STELLA features"
run_poetry_job "${STELLA_ENV}" \
  python "${PIPELINE_DIR}/save_stella_spica_features.py" \
  --data-root "${DATA_ROOT}" \
  --output-root "${FEATURE_ROOT}/stella" \
  --model-path "${STELLA_MODEL_PATH}" \
  --vector-dim "${STELLA_VECTOR_DIM}" \
  --dense-subdir "${STELLA_DENSE_SUBDIR}"

echo ""
echo "[INFO] All feature extraction jobs have finished"

