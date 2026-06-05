#!/usr/bin/env bash
# Llama-3.3-70B-Instruct vLLM server
# Usage: HF_TOKEN=hf_xxx bash scripts/serve_llama.sh
set -euo pipefail

ENV_NAME=${ENV_NAME:-vllm}
PORT=${PORT:-8001}
LOGDIR=${LOGDIR:-$HOME/vllm-system/logs}
MODEL_PATH=${MODEL_PATH:-ibnzterrell/Meta-Llama-3.3-70B-Instruct-AWQ-INT4}
TENSOR_PARALLEL=${TENSOR_PARALLEL:-2}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-8192}

# conda 환경 활성화
if command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate "${ENV_NAME}" || { echo "[ERR] conda activate failed"; exit 1; }
else
  echo "[ERR] conda not found"; exit 1
fi
PYBIN="$(which python)"

# HuggingFace 캐시 설정
CACHE_ROOT=${CACHE_ROOT:-$HOME/.cache/huggingface}
export HF_HOME="${HF_HOME:-$CACHE_ROOT}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/hub}"
mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE"
export OPENAI_API_KEY="${OPENAI_API_KEY:-dummy}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn

if [ -z "${HF_TOKEN:-}" ]; then
  echo "[WARN] HF_TOKEN not set — assuming model is already cached"
fi

# 포트 체크
if ss -ltnp 2>/dev/null | grep -q ":${PORT} "; then
  echo "[ERR] Port ${PORT} is already in use."
  exit 1
fi

mkdir -p "$LOGDIR"

echo "[INFO] Launching ${MODEL_PATH} on port ${PORT} (tp=${TENSOR_PARALLEL})"
exec "${PYBIN}" -m vllm.entrypoints.openai.api_server \
  --model "${MODEL_PATH}" \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --tensor-parallel-size "${TENSOR_PARALLEL}" \
  --max-num-seqs 8 \
  --max-model-len "${MAX_MODEL_LEN}" \
  ${HF_TOKEN:+--hf-token "${HF_TOKEN}"} \
  --download-dir "${TRANSFORMERS_CACHE}"
