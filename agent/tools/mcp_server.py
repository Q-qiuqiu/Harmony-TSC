#!/usr/bin/env python3
import json
import sys
from typing import Any

from cluster_resource_tool import fetch_cluster_resources
from task_catalog_tool import fetch_task_catalog
from vision_execute_tool import run_vision_task_on_node


GET_CLUSTER_RESOURCES_TOOL = "get_cluster_resources"
GET_TASK_CATALOG_TOOL = "get_task_catalog"
RUN_VISION_TASK_ON_NODE_TOOL = "run_vision_task_on_node"


def make_response(request_id: Any, result: dict[str, Any]) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def make_error(request_id: Any, code: int, message: str) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "error": {"code": code, "message": message},
    }


def tool_response_text(payload: dict[str, Any], request_id: Any) -> dict[str, Any]:
    return make_response(
        request_id,
        {
            "content": [{"type": "text", "text": json.dumps(payload, ensure_ascii=False)}]
        },
    )


def tool_response_error(message: str, request_id: Any) -> dict[str, Any]:
    return make_response(
        request_id,
        {
            "content": [{"type": "text", "text": message}],
            "isError": True,
        },
    )


def handle_initialize(request_id: Any) -> dict[str, Any]:
    return make_response(
        request_id,
        {
            "protocolVersion": "2024-11-05",
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "edge-cluster-tools", "version": "0.3.0"},
        },
    )


def handle_tools_list(request_id: Any) -> dict[str, Any]:
    return make_response(
        request_id,
        {
            "tools": [
                {
                    "name": GET_CLUSTER_RESOURCES_TOOL,
                    "description": "Fetch the latest cluster resource snapshot collected by the gateway.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {},
                        "additionalProperties": False,
                    },
                },
                {
                    "name": GET_TASK_CATALOG_TOOL,
                    "description": "Fetch a compact task catalog from static_info.json. Each item contains device_type, model_name, task_type, and overhead. Optionally filter by available_device_types from get_cluster_resources.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "available_device_types": {
                                "type": "array",
                                "items": {"type": "string"},
                            }
                        },
                        "additionalProperties": False,
                    },
                },
                {
                    "name": RUN_VISION_TASK_ON_NODE_TOOL,
                    "description": "Run a vision task on a specific board selected by the model. The gateway will create or reuse the target container on that board and forward the image inference request.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "task_type": {
                                "type": "string",
                                "enum": ["Bert", "MobileNet", "ResNet50", "YoloV5", "deeplabv3"],
                            },
                            "target_global_id": {"type": "string"},
                            "image_path": {"type": "string"},
                            "real_url": {"type": "string", "default": "predict"},
                            "file_field_name": {"type": "string", "default": "image"},
                        },
                        "required": ["task_type", "target_global_id", "image_path"],
                        "additionalProperties": False,
                    },
                },
            ]
        },
    )


def handle_tools_call(request_id: Any, params: dict[str, Any]) -> dict[str, Any]:
    name = params.get("name")
    arguments = params.get("arguments", {})

    try:
        if name == GET_CLUSTER_RESOURCES_TOOL:
            return tool_response_text(fetch_cluster_resources(), request_id)
        if name == GET_TASK_CATALOG_TOOL:
            return tool_response_text(fetch_task_catalog(arguments.get("available_device_types")), request_id)
        if name == RUN_VISION_TASK_ON_NODE_TOOL:
            return tool_response_text(run_vision_task_on_node(arguments), request_id)
        return make_error(request_id, -32602, f"unknown tool: {name}")
    except RuntimeError as exc:
        return tool_response_error(str(exc), request_id)


def handle_request(message: dict[str, Any]) -> dict[str, Any] | None:
    method = message.get("method")
    request_id = message.get("id")
    params = message.get("params", {})

    if method == "initialize":
        return handle_initialize(request_id)
    if method == "notifications/initialized":
        return None
    if method == "tools/list":
        return handle_tools_list(request_id)
    if method == "tools/call":
        return handle_tools_call(request_id, params)

    if request_id is None:
        return None
    return make_error(request_id, -32601, f"method not found: {method}")


def main() -> int:
    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue

        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            response = make_error(None, -32700, "parse error")
            sys.stdout.write(json.dumps(response) + "\n")
            sys.stdout.flush()
            continue

        response = handle_request(message)
        if response is None:
            continue

        sys.stdout.write(json.dumps(response, ensure_ascii=False) + "\n")
        sys.stdout.flush()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
