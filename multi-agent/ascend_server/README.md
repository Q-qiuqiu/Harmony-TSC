# Ascend Multi-Agent Server

This directory mirrors `multi-agent/server`, but the local reasoning model is the Qwen OpenAI-compatible server from the Ascend LLM image.

## Files

- `main_agent.py`: main scheduler agent, exposed at `/v1/multi-agent/chat`.
- `image_agent.py`: routes image requests to `YoloV5`, `ResNet50`, or `MobileNet`.
- `text_agent.py`: routes text requests to `Bert`.
- `segmentation_agent.py`: routes segmentation requests to `deeplabv3`.
- `*_agent.sh`: starts the local Qwen server first, waits for port `8081`, then starts the Python agent.

## Required Image Contents

The scheduler runtime directly uses the base image `qwen-openai-server:orangepi`.
It must contain:

- `/app/qwen_server/entrypoint.sh`
- `/app/qwen_server/qwen_openai_server`
- Python 3 with `flask` available in `/opt/qwen-venv`

The agent code is not copied into the image. It is mounted at container start:

```text
/root/Harmony-TSC/multi-agent:/workspace/multi-agent
```

Different sub-agent containers are created from the same image by running different scripts:

```text
/workspace/multi-agent/ascend_server/image_agent.sh
/workspace/multi-agent/ascend_server/text_agent.sh
/workspace/multi-agent/ascend_server/segmentation_agent.sh
```

The model directory should be mounted as:

```bash
/models/DeepSeek-R1-Distill-Qwen-1.5B_server
```

with this layout:

```text
DeepSeek-R1-Distill-Qwen-1.5B_server/
  config/config.json
  tokenizer/
  converted/
```

## Manual Start

```bash
bash /workspace/multi-agent/ascend_server/image_agent.sh
bash /workspace/multi-agent/ascend_server/text_agent.sh
bash /workspace/multi-agent/ascend_server/segmentation_agent.sh
bash /workspace/multi-agent/ascend_server/main_agent.sh
```

Useful environment variables:

- `MODEL_DIR=/models/DeepSeek-R1-Distill-Qwen-1.5B_server`
- `LLM_PORT=8081`
- `LLM_MODEL_NAME=deepseek-r1-distill-qwen-1.5b`
- `START_LOCAL_LLM=1`
- `GATEWAY_HOST=<gateway ip>`
- `GATEWAY_PORT=6666`
