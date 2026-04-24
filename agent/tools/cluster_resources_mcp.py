#!/usr/bin/env python3
import json
import os
import sys
import urllib.error
import urllib.request
from typing import Any


TOOL_NAME = "get_cluster_resources"
TOOL_DESCRIPTION = (
    "Fetch the latest cluster resource snapshot collected by the gateway. "
    "Returns all registered nodes with their current cpu, memory, xpu, and network metrics."
)


def _gateway_url() -> str:
    host = os.environ.get("GATEWAY_HOST", "127.0.0.1")
    port = os.environ.get("GATEWAY_PORT", "6666")
    path = os.environ.get("GATEWAY_CLUSTER_RESOURCES_PATH", "/cluster/resources")
    return f"http://{host}:{port}{path}"


def fetch_cluster_resources() -> dict[str, Any]:
    url = _gateway_url()
    request = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
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


def make_response(request_id: Any, result: dict[str, Any]) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def make_error(request_id: Any, code: int, message: str) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "error": {
            "code": code,
            "message": message,
        },
    }


def handle_initialize(request_id: Any) -> dict[str, Any]:
    return make_response(
        request_id,
        {
            "protocolVersion": "2024-11-05",
            "capabilities": {"tools": {}},
            "serverInfo": {
                "name": "edge-cluster-tools",
                "version": "0.1.0",
            },
        },
    )


def handle_tools_list(request_id: Any) -> dict[str, Any]:
    return make_response(
        request_id,
        {
            "tools": [
                {
                    "name": TOOL_NAME,
                    "description": TOOL_DESCRIPTION,
                    "inputSchema": {
                        "type": "object",
                        "properties": {},
                        "additionalProperties": False,
                    },
                }
            ]
        },
    )


def handle_tools_call(request_id: Any, params: dict[str, Any]) -> dict[str, Any]:
    name = params.get("name")
    if name != TOOL_NAME:
        return make_error(request_id, -32602, f"unknown tool: {name}")

    try:
        payload = fetch_cluster_resources()
    except RuntimeError as exc:
        return make_response(
            request_id,
            {
                "content": [{"type": "text", "text": str(exc)}],
                "isError": True,
            },
        )

    return make_response(
        request_id,
        {
            "content": [
                {
                    "type": "text",
                    "text": json.dumps(payload, ensure_ascii=False),
                }
            ]
        },
    )


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
