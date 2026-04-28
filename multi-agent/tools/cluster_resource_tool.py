import json
import os
import urllib.error
import urllib.request
from typing import Any


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
