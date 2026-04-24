#!/usr/bin/env python3
import json
import mimetypes
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
import uuid
from typing import Any


GET_CLUSTER_RESOURCES_TOOL = "get_cluster_resources"
RUN_VISION_TASK_ON_NODE_TOOL = "run_vision_task_on_node"

TASK_TYPE_TO_TASK_ID = {
    "YoloV5": "YoloV5",
    "MobileNet": "MobileNet",
    "ResNet50": "ResNet50",
    "Bert": "Bert",
    "deeplabv3": "deeplabv3",
}


def _gateway_host() -> str:
    return os.environ.get("GATEWAY_HOST", "127.0.0.1")


def _gateway_port() -> str:
    return os.environ.get("GATEWAY_PORT", "6666")


def _gateway_url(path: str) -> str:
    return f"http://{_gateway_host()}:{_gateway_port()}{path}"


def fetch_json(url: str) -> dict[str, Any]:
    request = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"gateway returned HTTP {exc.code} for {url}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"failed to reach gateway at {url}: {exc.reason}") from exc

    try:
        payload = json.loads(body)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"gateway returned invalid JSON: {exc}") from exc

    if not isinstance(payload, dict):
        raise RuntimeError("gateway returned non-object JSON payload")
    return payload


def fetch_cluster_resources() -> dict[str, Any]:
    path = os.environ.get("GATEWAY_CLUSTER_RESOURCES_PATH", "/cluster_resources")
    return fetch_json(_gateway_url(path))


def encode_multipart_formdata(file_field_name: str, file_path: str) -> tuple[bytes, str]:
    boundary = f"----edge-cluster-{uuid.uuid4().hex}"
    filename = os.path.basename(file_path)
    content_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"
    with open(file_path, "rb") as file_obj:
        file_bytes = file_obj.read()

    body = bytearray()
    body.extend(f"--{boundary}\r\n".encode("utf-8"))
    body.extend(
        (
            f'Content-Disposition: form-data; name="{file_field_name}"; '
            f'filename="{filename}"\r\n'
        ).encode("utf-8")
    )
    body.extend(f"Content-Type: {content_type}\r\n\r\n".encode("utf-8"))
    body.extend(file_bytes)
    body.extend(f"\r\n--{boundary}--\r\n".encode("utf-8"))
    return bytes(body), f"multipart/form-data; boundary={boundary}"


def run_vision_task_on_node(arguments: dict[str, Any]) -> dict[str, Any]:
    task_type = arguments.get("task_type")
    target_global_id = arguments.get("target_global_id")
    image_path = arguments.get("image_path")
    real_url = arguments.get("real_url", "predict")
    file_field_name = arguments.get("file_field_name", "image")

    if task_type not in TASK_TYPE_TO_TASK_ID:
        raise RuntimeError(f"unsupported task_type: {task_type}")
    if not target_global_id:
        raise RuntimeError("target_global_id is required")
    if not image_path:
        raise RuntimeError("image_path is required")
    if not os.path.exists(image_path):
        raise RuntimeError(f"image_path does not exist: {image_path}")

    path = os.environ.get("GATEWAY_QUEST_ON_NODE_PATH", "/quest_on_node")
    query = urllib.parse.urlencode(
        {
            "taskid": TASK_TYPE_TO_TASK_ID[task_type],
            "target_global_id": target_global_id,
            "real_url": real_url,
        }
    )
    url = f"{_gateway_url(path)}?{query}"

    body, content_type = encode_multipart_formdata(file_field_name, image_path)
    request = urllib.request.Request(url, data=body, method="POST")
    request.add_header("Content-Type", content_type)
    request.add_header("Content-Length", str(len(body)))

    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            response_body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"gateway returned HTTP {exc.code} for {url}: {error_body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"failed to reach gateway at {url}: {exc.reason}") from exc

    try:
        return json.loads(response_body)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"gateway returned invalid JSON: {exc}") from exc


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
            "serverInfo": {"name": "edge-cluster-tools", "version": "0.2.0"},
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
                    "name": RUN_VISION_TASK_ON_NODE_TOOL,
                    "description": "Run a vision task on a specific board selected by the model. The gateway will create or reuse the target container on that board and forward the image inference request.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "task_type": {
                                "type": "string",
                                "enum": sorted(TASK_TYPE_TO_TASK_ID.keys()),
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
