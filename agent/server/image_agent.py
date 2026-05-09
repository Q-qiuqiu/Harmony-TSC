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
DEFAULT_MCP_SERVER_PATH = os.path.join(TOOLS_DIR, "mcp_server.py")

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


def call_mcp_server(server_path, requests, timeout=30):
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
        raise RuntimeError("MCP tool invocation timed out")

    if process.returncode != 0:
        raise RuntimeError(f"MCP server exited with code {process.returncode}: {stderr.strip()}")

    responses = []
    for line in stdout.splitlines():
        line = line.strip()
        if line:
            responses.append(json.loads(line))
    return responses


def list_mcp_tools(server_path, timeout=30):
    responses = call_mcp_server(
        server_path,
        [
            {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
            {"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}},
            {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
        ],
        timeout=timeout,
    )
    for item in responses:
        if item.get("id") == 2:
            return item.get("result", {}).get("tools", [])
    raise RuntimeError("MCP server did not return tools/list response")


def call_mcp_tool(server_path, tool_name, arguments=None, timeout=30):
    if arguments is None:
        arguments = {}

    responses = call_mcp_server(
        server_path,
        [
            {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
            {"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}},
            {"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {"name": tool_name, "arguments": arguments}},
        ],
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


def execute_tool(tool_name, arguments=None):
    if not os.path.exists(DEFAULT_MCP_SERVER_PATH):
        raise RuntimeError(f"MCP server not found: {DEFAULT_MCP_SERVER_PATH}")
    return call_mcp_tool(DEFAULT_MCP_SERVER_PATH, tool_name, arguments=arguments or {})


def discover_tools():
    if not os.path.exists(DEFAULT_MCP_SERVER_PATH):
        raise RuntimeError(f"MCP server not found: {DEFAULT_MCP_SERVER_PATH}")
    return list_mcp_tools(DEFAULT_MCP_SERVER_PATH)


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


def build_agent_messages(user_text, image_path, tools):
    tool_list_json = json.dumps(tools, ensure_ascii=False)
    return [
        {
            "role": "system",
            "content": (
                "You are an image task orchestration agent.\n"
                "You must decide which tools to call, in what order, and with what arguments.\n"
                "The uploaded image is already saved locally and tools can use the local image path.\n"
                "Use tools to inspect cluster resources, available task catalog, and execute the image task.\n"
                "Before calling the execution tool, you must first call get_cluster_resources.\n"
                "If model choice matters, call get_task_catalog before the execution tool.\n"
                "When calling get_task_catalog, pass available_device_types derived from the node types returned by get_cluster_resources so the context only contains models that match currently available platforms.\n"
                "For execution, prefer lower resource overhead models when they can satisfy the request.\n"
                "For generic multi-object detection requests, YoloV5 is often suitable; for lightweight classification-style requests, MobileNet or ResNet50 may be more appropriate.\n"
                "Available tools:\n"
                f"{tool_list_json}\n"
                "Rules:\n"
                "1. If you want to call a tool, reply with JSON only:\n"
                "{\"action\":\"tool_call\",\"tool_name\":\"<tool name>\",\"arguments\":{}}\n"
                "2. Tool arguments must match the tool schema.\n"
                "3. Never call run_vision_task_on_node until you already know a real target_global_id from tool output.\n"
                "4. target_global_id must come from get_cluster_resources result and must never be a placeholder such as default, unknown, null, or empty string.\n"
                "5. When calling run_vision_task_on_node, include the provided image_path exactly as given.\n"
                "6. After receiving tool results, continue reasoning and either call another tool or produce the final answer.\n"
                "7. If required information is missing, call another tool instead of guessing.\n"
                "8. Do not invent node ids, task types, device types, or tool results.\n"
                "9. When you have enough information, reply with JSON only:\n"
                "{\"action\":\"final\",\"content\":\"<natural language chinese answer>\"}\n"
                "10. Never output anything except a single JSON object."
            ),
        },
        {
            "role": "user",
            "content": (
                f"user_text:\n{user_text}\n\n"
                f"image_path:\n{image_path}"
            ),
        },
    ]


def run_image_agent(user_text, image_path, max_steps=6):
    working_messages = build_agent_messages(user_text, image_path, discover_tools())
    total_prompt_tokens = 0
    total_completion_tokens = 0
    selected_execution = None
    last_tool_result = None

    for _ in range(max_steps):
        model_result = call_llm(working_messages)
        total_prompt_tokens += model_result["prompt_tokens"]
        total_completion_tokens += model_result["completion_tokens"]

        try:
            agent_action = parse_json_object(model_result["content"])
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"model did not return valid JSON for agent control: {exc}")

        action = agent_action.get("action")
        if action == "final":
            content = agent_action.get("content", "").strip()
            if not content:
                raise RuntimeError("model returned empty final content")
            return {
                "message": content,
                "selected_node": selected_execution,
                "tool_result": last_tool_result,
                "usage": {
                    "prompt_tokens": total_prompt_tokens,
                    "completion_tokens": total_completion_tokens,
                    "total_tokens": total_prompt_tokens + total_completion_tokens,
                },
            }

        if action != "tool_call":
            raise RuntimeError(f"unknown agent action: {action}")

        tool_name = agent_action.get("tool_name")
        tool_arguments = agent_action.get("arguments", {})
        if not tool_name:
            raise RuntimeError("model tool_call is missing tool_name")

        tool_result_text = execute_tool(tool_name, tool_arguments)
        try:
            tool_result_value = json.loads(tool_result_text)
        except json.JSONDecodeError:
            tool_result_value = tool_result_text

        if tool_name == "run_vision_task_on_node":
            selected_execution = {
                "task_type": tool_arguments.get("task_type"),
                "target_global_id": tool_arguments.get("target_global_id"),
                "real_url": tool_arguments.get("real_url", "predict"),
            }
            last_tool_result = tool_result_value

        working_messages.append({"role": "assistant", "content": model_result["content"]})
        working_messages.append({
            "role": "tool",
            "content": f"tool_name={tool_name}\nresult={json.dumps(tool_result_value, ensure_ascii=False)}"
        })

    raise RuntimeError("image agent exceeded max tool-calling steps without producing a final answer")


@app.route("/v1/agent/chat", methods=["POST"])
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

        agent_result = run_image_agent(user_text, temp_image_path)

        return Response(json.dumps({
            "object": "chat.cluster",
            "message": agent_result["message"],
            "selected_node": agent_result["selected_node"],
            "tool_result": agent_result["tool_result"],
            "usage": agent_result["usage"],
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
