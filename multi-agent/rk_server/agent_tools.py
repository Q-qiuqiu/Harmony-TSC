import base64
import json
import os
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
import uuid


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MULTI_AGENT_DIR = os.path.normpath(os.path.join(BASE_DIR, ".."))
REPO_ROOT = os.path.normpath(os.path.join(MULTI_AGENT_DIR, ".."))
DEFAULT_MCP_SERVER_PATH = os.path.join(MULTI_AGENT_DIR, "tools", "mcp_server.py")
DEFAULT_LLM_API_URL = "http://127.0.0.1:8081/v1/chat/completions"
DEFAULT_LLM_MODEL_NAME = "rkllm-default"
DEFAULT_IMAGE_AGENT_URL = "http://127.0.0.1:8083/v1/sub-agents/image/execute"
DEFAULT_TEXT_AGENT_URL = "http://127.0.0.1:8085/v1/sub-agents/text/execute"
DEFAULT_SEGMENTATION_AGENT_URL = "http://127.0.0.1:8086/v1/sub-agents/segmentation/execute"
DEFAULT_SUB_AGENT_PROFILE_PATH = os.path.join(REPO_ROOT, "config_files", "multi_agent_info.json")


def post_json(url, payload, timeout=180):
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url, data=body, method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("Content-Length", str(len(body)))

    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} for {url}: {error_body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"failed to reach {url}: {exc.reason}") from exc


def post_multipart(url, fields, files, timeout=180):
    boundary = f"----multi-agent-{uuid.uuid4().hex}"
    body = bytearray()

    for key, value in fields.items():
        if value is None:
            continue
        body.extend(f"--{boundary}\r\n".encode("utf-8"))
        body.extend(f'Content-Disposition: form-data; name="{key}"\r\n\r\n'.encode("utf-8"))
        body.extend(str(value).encode("utf-8"))
        body.extend(b"\r\n")

    for file_item in files:
        filename = file_item.get("filename", "upload.bin")
        field_name = file_item.get("field_name", "file")
        content_type = file_item.get("content_type", "application/octet-stream")
        content = file_item.get("content", b"")
        body.extend(f"--{boundary}\r\n".encode("utf-8"))
        body.extend(
            f'Content-Disposition: form-data; name="{field_name}"; filename="{filename}"\r\n'.encode("utf-8")
        )
        body.extend(f"Content-Type: {content_type}\r\n\r\n".encode("utf-8"))
        body.extend(content)
        body.extend(b"\r\n")

    body.extend(f"--{boundary}--\r\n".encode("utf-8"))

    req = urllib.request.Request(url, data=bytes(body), method="POST")
    req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
    req.add_header("Content-Length", str(len(body)))

    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} for {url}: {error_body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"failed to reach {url}: {exc.reason}") from exc


def load_sub_agent_catalog(profile_path=DEFAULT_SUB_AGENT_PROFILE_PATH):
    try:
        with open(profile_path, "r", encoding="utf-8") as file_obj:
            payload = json.load(file_obj)
    except OSError as exc:
        raise RuntimeError(f"failed to read sub agent profile file {profile_path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"invalid JSON in sub agent profile file {profile_path}: {exc}") from exc

    sub_agents = payload.get("sub_agents")
    if not isinstance(sub_agents, dict):
        raise RuntimeError("multi_agent_info.json must contain an object field named 'sub_agents'")
    return sub_agents


def get_sub_agent_profile(agent_name, profile_path=DEFAULT_SUB_AGENT_PROFILE_PATH):
    catalog = load_sub_agent_catalog(profile_path)
    profile = catalog.get(agent_name)
    if not isinstance(profile, dict):
        raise RuntimeError(f"sub agent profile not found: {agent_name}")
    return profile


def call_llm(messages, llm_api_url=DEFAULT_LLM_API_URL, model_name=DEFAULT_LLM_MODEL_NAME):
    raw = post_json(
        llm_api_url,
        {
            "model": model_name,
            "messages": messages,
            "stream": False,
        },
    )
    payload = json.loads(raw)
    choices = payload.get("choices", [])
    if not choices:
        raise RuntimeError("LLM API returned no choices")
    message = choices[0].get("message", {})
    content = message.get("content", "")
    if not content:
        raise RuntimeError("LLM API returned empty content")
    usage = payload.get("usage", {})
    return {
        "content": content,
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
    }


def call_mcp_server(requests, server_path=DEFAULT_MCP_SERVER_PATH, timeout=60):
    process = subprocess.Popen(
        [sys.executable, server_path],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    request_payload = "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in requests)

    try:
        stdout, stderr = process.communicate(request_payload, timeout=timeout)
    except subprocess.TimeoutExpired:
        process.kill()
        stdout, stderr = process.communicate()
        raise RuntimeError("MCP request timed out")

    if process.returncode != 0:
        raise RuntimeError(f"MCP server exited with code {process.returncode}: {stderr.strip()}")

    responses = []
    for line in stdout.splitlines():
        line = line.strip()
        if line:
            responses.append(json.loads(line))
    return responses


def list_mcp_tools(server_path=DEFAULT_MCP_SERVER_PATH, timeout=30):
    responses = call_mcp_server(
        [
            {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
            {"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}},
            {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
        ],
        server_path=server_path,
        timeout=timeout,
    )
    for item in responses:
        if item.get("id") == 2:
            return item.get("result", {}).get("tools", [])
    raise RuntimeError("MCP server did not return tools/list response")


def call_mcp_tool(tool_name, arguments=None, server_path=DEFAULT_MCP_SERVER_PATH, timeout=30):
    if arguments is None:
        arguments = {}

    responses = call_mcp_server(
        [
            {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
            {"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}},
            {"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {"name": tool_name, "arguments": arguments}},
        ],
        server_path=server_path,
        timeout=timeout,
    )

    for item in responses:
        if item.get("id") != 2:
            continue
        result = item.get("result", {})
        if result.get("isError"):
            content = result.get("content", [])
            if content and isinstance(content[0], dict):
                raise RuntimeError(content[0].get("text", "MCP tool returned an error"))
            raise RuntimeError("MCP tool returned an error")

        content = result.get("content", [])
        if not content or not isinstance(content[0], dict):
            raise RuntimeError("MCP tool returned empty content")
        return content[0].get("text", "")

    raise RuntimeError("MCP tool did not return a tools/call response")


def parse_json_object(text):
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if len(lines) >= 3:
            cleaned = "\n".join(lines[1:-1]).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        if start == -1:
            raise

        depth = 0
        in_string = False
        escape = False
        for index in range(start, len(cleaned)):
            ch = cleaned[index]
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == "\"":
                    in_string = False
                continue

            if ch == "\"":
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return json.loads(cleaned[start:index + 1])
        raise


def extract_text_fields_from_form(form_data):
    text_fields = {}
    for key in form_data.keys():
        value = form_data.get(key)
        if value is not None and value.strip():
            text_fields[key] = value.strip()
    return text_fields


def choose_user_text(text_fields):
    preferred_keys = ["text", "prompt", "query", "message", "instruction"]
    for key in preferred_keys:
        value = text_fields.get(key)
        if value:
            return value
    if text_fields:
        return "\n".join(f"{key}: {value}" for key, value in text_fields.items())
    raise RuntimeError("no non-empty text field found in multipart form data")


def save_uploaded_file(uploaded_file):
    suffix = os.path.splitext(uploaded_file.filename or "")[1] or ".bin"
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    uploaded_file.save(temp_file.name)
    temp_file.close()
    return temp_file.name


def encode_bytes_to_b64(file_bytes):
    return base64.b64encode(file_bytes).decode("ascii")
