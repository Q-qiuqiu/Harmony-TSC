#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/ascend_common.sh"

run_python_agent text_agent.py "${TEXT_AGENT_PORT:-8085}"
