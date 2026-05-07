#!/usr/bin/env bash
set -euo pipefail

ASCEND_SERVER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

: "${LLM_HOST:=127.0.0.1}"
: "${LLM_PORT:=8081}"
: "${LLM_MODEL_NAME:=deepseek-r1-distill-qwen-1.5b}"
: "${LLM_API_URL:=http://${LLM_HOST}:${LLM_PORT}/v1/chat/completions}"
: "${START_LOCAL_LLM:=1}"
: "${QWEN_ENTRYPOINT:=/app/qwen_server/entrypoint.sh}"
: "${PYTHON_BIN:=}"

if [ -z "${PYTHON_BIN}" ]; then
  if [ -x /opt/qwen-venv/bin/python3 ]; then
    PYTHON_BIN=/opt/qwen-venv/bin/python3
  else
    PYTHON_BIN=python3
  fi
fi

export LLM_API_URL
export LLM_MODEL_NAME
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

wait_for_port() {
  local host="$1"
  local port="$2"
  local timeout="${3:-180}"
  "${PYTHON_BIN}" - "$host" "$port" "$timeout" <<'PY'
import socket
import sys
import time

host = sys.argv[1]
port = int(sys.argv[2])
timeout = float(sys.argv[3])
deadline = time.time() + timeout

while time.time() < deadline:
    try:
        with socket.create_connection((host, port), timeout=1.0):
            sys.exit(0)
    except OSError:
        time.sleep(1)

print(f"timeout waiting for {host}:{port}", file=sys.stderr)
sys.exit(1)
PY
}

start_local_llm() {
  if [ "${START_LOCAL_LLM}" != "1" ]; then
    return 0
  fi

  if wait_for_port "${LLM_HOST}" "${LLM_PORT}" 2 >/dev/null 2>&1; then
    echo "Qwen OpenAI server is already listening on ${LLM_HOST}:${LLM_PORT}"
    return 0
  fi

  if [ ! -x "${QWEN_ENTRYPOINT}" ]; then
    echo "Qwen entrypoint not found or not executable: ${QWEN_ENTRYPOINT}" >&2
    exit 1
  fi

  echo "Starting local Qwen OpenAI server on ${LLM_HOST}:${LLM_PORT}"
  HOST="${LLM_HOST}" \
  PORT="${LLM_PORT}" \
  SERVED_MODEL_NAME="${LLM_MODEL_NAME}" \
  "${QWEN_ENTRYPOINT}" &
  LLM_PID=$!

  cleanup_llm() {
    if [ -n "${LLM_PID:-}" ] && kill -0 "${LLM_PID}" >/dev/null 2>&1; then
      kill "${LLM_PID}" >/dev/null 2>&1 || true
      wait "${LLM_PID}" >/dev/null 2>&1 || true
    fi
  }
  trap cleanup_llm EXIT INT TERM

  wait_for_port "${LLM_HOST}" "${LLM_PORT}" "${LLM_STARTUP_TIMEOUT_SEC:-240}"
}

run_python_agent() {
  local agent_file="$1"
  local agent_port="$2"
  start_local_llm
  "${PYTHON_BIN}" "${ASCEND_SERVER_DIR}/${agent_file}" \
    --host "${AGENT_HOST:-0.0.0.0}" \
    --port "${agent_port}" \
    --llm_api_url "${LLM_API_URL}" \
    --llm_model_name "${LLM_MODEL_NAME}"
}
