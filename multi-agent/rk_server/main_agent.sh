#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${RKLLM_MODEL_PATH:-/root/models/Qwen3-1.7B-rk3588-w8a8.rkllm}"
TARGET_PLATFORM="${RKLLM_TARGET_PLATFORM:-rk3588}"
LLM_PORT="${LLM_PORT:-8081}"
MAIN_AGENT_PORT="${MAIN_AGENT_PORT:-8084}"
IMAGE_AGENT_URL="${IMAGE_AGENT_URL:-http://127.0.0.1:8083/v1/sub-agents/image/execute}"

export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

wait_for_port() {
  local host="$1"
  local port="$2"
  local retries="${3:-120}"
  for _ in $(seq 1 "${retries}"); do
    if bash -lc "exec 3<>/dev/tcp/${host}/${port}" >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done
  return 1
}

llm_pid=""
agent_pid=""

cleanup() {
  for pid in "${agent_pid}" "${llm_pid}"; do
    if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
      kill "${pid}" 2>/dev/null || true
      wait "${pid}" 2>/dev/null || true
    fi
  done
}
trap cleanup EXIT INT TERM

python3 /root/edge-cluster-scheduler/multi-agent/rk_server/flask_openai_server.py \
  --rkllm_model_path="${MODEL_PATH}" \
  --target_platform="${TARGET_PLATFORM}" &
llm_pid=$!

if ! wait_for_port 127.0.0.1 "${LLM_PORT}" 120; then
  echo "LLM service did not become ready on 127.0.0.1:${LLM_PORT}" >&2
  exit 1
fi

python3 /root/edge-cluster-scheduler/multi-agent/rk_server/main_agent.py \
  --host 0.0.0.0 \
  --port "${MAIN_AGENT_PORT}" \
  --llm_api_url "http://127.0.0.1:${LLM_PORT}/v1/chat/completions" \
  --image_agent_url "${IMAGE_AGENT_URL}" &
agent_pid=$!

wait "${agent_pid}"
