#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/ascend_common.sh"

start_local_llm

"${PYTHON_BIN}" "${ASCEND_SERVER_DIR}/main_agent.py" \
  --host "${AGENT_HOST:-0.0.0.0}" \
  --port "${MAIN_AGENT_PORT:-8084}" \
  --llm_api_url "${LLM_API_URL}" \
  --llm_model_name "${LLM_MODEL_NAME}" \
  --image_agent_url "${IMAGE_AGENT_URL:-http://127.0.0.1:8083/v1/sub-agents/image/execute}" \
  --text_agent_url "${TEXT_AGENT_URL:-http://127.0.0.1:8085/v1/sub-agents/text/execute}" \
  --segmentation_agent_url "${SEGMENTATION_AGENT_URL:-http://127.0.0.1:8086/v1/sub-agents/segmentation/execute}"
