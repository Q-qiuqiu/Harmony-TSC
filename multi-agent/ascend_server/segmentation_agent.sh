#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/ascend_common.sh"

run_python_agent segmentation_agent.py "${SEGMENTATION_AGENT_PORT:-8086}"
