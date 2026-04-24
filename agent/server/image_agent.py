import argparse
import json
import mimetypes
import os
import subprocess
import sys
import tempfile
import threading
import urllib.error
import urllib.parse
import urllib.request
from flask import Flask, Response, jsonify, request

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TOOLS_DIR = os.path.normpath(os.path.join(BASE_DIR, "..", "tools"))
DEFAULT_MCP_SERVER_PATH = os.path.join(TOOLS_DIR, "cluster_resources_mcp.py")
DEFAULT_MCP_TOOL_NAME = "get_cluster_resources"

lock = threading.Lock()
is_blocking = False
LLM_API_URL = "http://127.0.0.1:8081/v1/chat/completions"
LLM_MODEL_NAME = "rkllm-default"


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
        raise RuntimeError(f"LLM API returned HTTP {exc.code}: {error_body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"failed to reach LLM API at {url}: {exc.reason}") from exc


def call_llm(messages):
    raw = post_json(
        LLM_API_URL,
        {
            "model": LLM_MODEL_NAME,
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


def call_mcp_tool(server_path, tool_name, arguments=None, timeout=30):
    if arguments is None:
        arguments = {}

    process = subprocess.Popen(
        [sys.executable, server_path],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    requests = [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
        {"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {"name": tool_name, "arguments": arguments}},
    ]
    request_payload = "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in requests)

    try:
        stdout, stderr = process.communicate(request_payload, timeout=timeout)
    except subprocess.TimeoutExpired:
        process.kill()
        stdout, stderr = process.communicate()
        raise RuntimeError("MCP tool invocation timed out")

    if process.returncode != 0:
        raise RuntimeError(f"MCP server exited with code {process.returncode}: {stderr.strip()}")

    responses = []
    for line in stdout.splitlines():
        line = line.strip()
        if line:
            responses.append(json.loads(line))

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


def execute_tool(tool_name, arguments=None):
    if not os.path.exists(DEFAULT_MCP_SERVER_PATH):
        raise RuntimeError(f"MCP server not found: {DEFAULT_MCP_SERVER_PATH}")
    return call_mcp_tool(DEFAULT_MCP_SERVER_PATH, tool_name, arguments=arguments or {})


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


def save_uploaded_image(uploaded_file):
    suffix = os.path.splitext(uploaded_file.filename or "")[1] or ".bin"
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    uploaded_file.save(temp_file.name)
    temp_file.close()
    return temp_file.name


def select_node_for_image_request(user_text, cluster_resources):
    messages = [
        {
            "role": "system",
            "content": (
                "You are selecting a board for an edge vision task.\n"
                "Use only the user text and the provided cluster resources.\n"
                "Return JSON only with this schema:\n"
                "{\"task_type\":\"YoloV5\",\"target_global_id\":\"<uuid>\",\"real_url\":\"predict\",\"reason\":\"<short reason>\"}\n"
                "Rules:\n"
                "1. Prefer YoloV5 for generic image detection requests.\n"
                "2. target_global_id must be one of the provided nodes.\n"
                "3. real_url should usually be predict."
            ),
        },
        {
            "role": "user",
            "content": (
                f"user_text:\n{user_text}\n\n"
                f"cluster_resources:\n{json.dumps(cluster_resources, ensure_ascii=False)}"
            ),
        },
    ]
    model_result = call_llm(messages)
    selection = parse_json_object(model_result["content"])
    for key in ["task_type", "target_global_id", "real_url"]:
        if not selection.get(key):
            raise RuntimeError(f"model selection result missing required field: {key}")

    available_ids = {
        node.get("global_id")
        for node in cluster_resources.get("result", [])
        if isinstance(node, dict) and node.get("global_id")
    }
    if selection["target_global_id"] not in available_ids:
        raise RuntimeError("model selected a target_global_id that is not in cluster resources")

    selection["_usage"] = {
        "prompt_tokens": model_result["prompt_tokens"],
        "completion_tokens": model_result["completion_tokens"],
    }
    return selection


def summarize_image_result(user_text, selection, tool_result):
    messages = [
        {
            "role": "system",
            "content": (
                "You are an assistant explaining an image recognition result in Chinese.\n"
                "Use only the user text and the tool result.\n"
                "Reply in natural language.\n"
                "Do not output JSON."
            ),
        },
        {
            "role": "user",
            "content": (
                f"user_text:\n{user_text}\n\n"
                f"selection:\n{json.dumps(selection, ensure_ascii=False)}\n\n"
                f"tool_result:\n{json.dumps(tool_result, ensure_ascii=False)}"
            ),
        },
    ]
    model_result = call_llm(messages)
    answer = model_result["content"].strip()
    if not answer:
        raise RuntimeError("model summary result is empty")
    return {
        "answer": answer,
        "_usage": {
            "prompt_tokens": model_result["prompt_tokens"],
            "completion_tokens": model_result["completion_tokens"],
        },
    }


@app.route("/v1/chat/cluster", methods=["POST"])
def chat_cluster():
    global is_blocking

    if is_blocking:
        return jsonify({
            "error": {
                "message": "Image agent is busy! Please try again later.",
                "type": "server_error",
                "param": None,
                "code": None,
            }
        }), 503

    image_file = request.files.get("image")
    if image_file is None or not image_file.filename:
        return jsonify({
            "error": {
                "message": "multipart form must include an image field named 'image'",
                "type": "invalid_request_error",
                "param": "image",
                "code": None,
            }
        }), 400

    text_fields = extract_text_fields_from_form(request.form)
    try:
        user_text = choose_user_text(text_fields)
    except RuntimeError as exc:
        return jsonify({
            "error": {
                "message": str(exc),
                "type": "invalid_request_error",
                "param": "text",
                "code": None,
            }
        }), 400

    lock.acquire()
    temp_image_path = None
    try:
        is_blocking = True
        temp_image_path = save_uploaded_image(image_file)

        cluster_resources = json.loads(execute_tool(DEFAULT_MCP_TOOL_NAME))
        selection = select_node_for_image_request(user_text, cluster_resources)
        tool_result = json.loads(execute_tool(
            "run_vision_task_on_node",
            {
                "task_type": selection["task_type"],
                "target_global_id": selection["target_global_id"],
                "image_path": temp_image_path,
                "real_url": selection.get("real_url", "predict"),
                "file_field_name": "image",
            }
        ))
        summary = summarize_image_result(user_text, selection, tool_result)

        total_prompt_tokens = selection["_usage"]["prompt_tokens"] + summary["_usage"]["prompt_tokens"]
        total_completion_tokens = selection["_usage"]["completion_tokens"] + summary["_usage"]["completion_tokens"]

        return Response(json.dumps({
            "object": "chat.cluster",
            "message": summary["answer"],
            "selected_node": {
                "task_type": selection["task_type"],
                "target_global_id": selection["target_global_id"],
                "real_url": selection.get("real_url", "predict"),
                "reason": selection.get("reason", ""),
            },
            "tool_result": tool_result,
            "usage": {
                "prompt_tokens": total_prompt_tokens,
                "completion_tokens": total_completion_tokens,
                "total_tokens": total_prompt_tokens + total_completion_tokens,
            }
        }, ensure_ascii=False), content_type="application/json")
    except Exception as exc:
        return jsonify({
            "error": {
                "message": str(exc),
                "type": "server_error",
                "param": None,
                "code": None,
            }
        }), 500
    finally:
        if temp_image_path and os.path.exists(temp_image_path):
            os.unlink(temp_image_path)
        lock.release()
        is_blocking = False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0", help="Bind host for the image agent server")
    parser.add_argument("--port", type=int, default=8082, help="Bind port for the image agent server")
    parser.add_argument("--llm_api_url", default="http://127.0.0.1:8081/v1/chat/completions", help="Base LLM API endpoint")
    parser.add_argument("--llm_model_name", default="rkllm-default", help="Model name passed to the local LLM API")
    args = parser.parse_args()

    LLM_API_URL = args.llm_api_url
    LLM_MODEL_NAME = args.llm_model_name

    app.run(host=args.host, port=args.port, threaded=True, debug=False)
