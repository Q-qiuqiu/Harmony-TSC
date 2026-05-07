import json
import os
import urllib.error
import urllib.parse
import urllib.request
from typing import Any


def _gateway_host() -> str:
    return os.environ.get("GATEWAY_HOST", "127.0.0.1")


def _gateway_port() -> str:
    return os.environ.get("GATEWAY_PORT", "6666")


def _gateway_url(path: str) -> str:
    return f"http://{_gateway_host()}:{_gateway_port()}{path}"


def start_sub_agent(arguments: dict[str, Any]) -> dict[str, Any]:
    agent_name = arguments.get("agent_name")
    target_global_id = arguments.get("target_global_id")

    if not agent_name:
        raise RuntimeError("agent_name is required")
    if not target_global_id:
        raise RuntimeError("target_global_id is required")

    path = os.environ.get("GATEWAY_START_SUB_AGENT_PATH", "/start_sub_agent")
    query = urllib.parse.urlencode(
        {
            "agent_name": agent_name,
            "target_global_id": target_global_id,
        }
    )
    url = f"{_gateway_url(path)}?{query}"
    request = urllib.request.Request(url, data=b"", method="POST")
    request.add_header("Content-Length", "0")

    try:
        with urllib.request.urlopen(request, timeout=300) as response:
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
