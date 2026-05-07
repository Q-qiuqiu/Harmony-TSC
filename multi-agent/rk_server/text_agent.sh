#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${RKLLM_MODEL_PATH:-/models/Qwen3-1.7B-rk3588-w8a8.rkllm}"
TARGET_PLATFORM="${RKLLM_TARGET_PLATFORM:-rk3588}"
LLM_PORT="${LLM_PORT:-8081}"
LLM_URL="${LLM_API_URL:-http://127.0.0.1:${LLM_PORT}/v1/chat/completions}"
AGENT_PORT="${TEXT_AGENT_PORT:-8085}"

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

if [[ -z "${GATEWAY_HOST:-}" ]]; then
  if [[ -x /sbin/ip ]]; then
    GATEWAY_HOST="$(/sbin/ip route | awk '/default/ {print $3; exit}')"
  elif [[ -x /usr/sbin/ip ]]; then
    GATEWAY_HOST="$(/usr/sbin/ip route | awk '/default/ {print $3; exit}')"
  else
    GATEWAY_HOST="192.168.58.3"
  fi
  export GATEWAY_HOST
fi

: "${GATEWAY_PORT:=6666}"
export GATEWAY_PORT
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

python3 /app/flask_openai_server.py \
  --rkllm_model_path="${MODEL_PATH}" \
  --target_platform="${TARGET_PLATFORM}" &
llm_pid=$!

if ! wait_for_port 127.0.0.1 "${LLM_PORT}" 120; then
  echo "LLM service did not become ready on 127.0.0.1:${LLM_PORT}" >&2
  exit 1
fi

python3 /workspace/multi-agent/server/text_agent.py \
  --host 0.0.0.0 \
  --port "${AGENT_PORT}" \
  --llm_api_url "${LLM_URL}" &
agent_pid=$!

wait "${agent_pid}"
