import argparse
import json
import mimetypes
import os
import re
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


def execute_tool(tool_name, arguments=None, timeout=30):
    if not os.path.exists(DEFAULT_MCP_SERVER_PATH):
        raise RuntimeError(f"MCP server not found: {DEFAULT_MCP_SERVER_PATH}")
    return call_mcp_tool(DEFAULT_MCP_SERVER_PATH, tool_name, arguments=arguments or {}, timeout=timeout)


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


def build_model_selection_messages(user_text, image_name, execution_candidates):
    return [
        {
            "role": "system",
            "content": (
                "You are an image execution selector.\n"
                "The program has already collected cluster resources and execution candidates.\n"
                "Choose exactly one valid task_type and target_global_id from the provided execution candidates.\n"
                "Return JSON only with this schema:\n"
                "{\"task_type\":\"YoloV5\",\"target_global_id\":\"<uuid>\",\"reason\":\"<short reason>\"}\n"
                "Rules:\n"
                "1. task_type and target_global_id must be chosen from the provided execution candidates.\n"
                "2. YoloV5 is for object detection and bounding boxes, not image classification.\n"
                "3. MobileNet and ResNet50 are for image classification.\n"
                "4. For classification requests, prefer MobileNet when speed is important; use ResNet50 when accuracy is preferred.\n"
                "5. Prefer lower expected overhead when it still satisfies the task intent.\n"
                "6. Never call tools. Never output anything except a single JSON object."
            ),
        },
        {
            "role": "user",
            "content": (
                f"user_text:\n{user_text}\n\n"
                f"image_name:\n{image_name}\n\n"
                f"execution_candidates:\n{json.dumps(execution_candidates, ensure_ascii=False)}"
            ),
        },
    ]


def build_result_summary_messages(user_text, selected_execution, tool_result):
    return [
        {
            "role": "system",
            "content": (
                "You are an image agent.\n"
                "Summarize the actual vision execution result in concise Chinese for the end user.\n"
                "Base the answer only on the provided execution result.\n"
                "If the result is empty or unclear, say so explicitly.\n"
                "Return plain Chinese text only."
            ),
        },
        {
            "role": "user",
            "content": (
                f"user_text:\n{user_text}\n\n"
                f"selected_execution:\n{json.dumps(selected_execution, ensure_ascii=False)}\n\n"
                f"tool_result:\n{json.dumps(tool_result, ensure_ascii=False)}"
            ),
        },
    ]


def build_execution_candidates(cluster_resources, task_catalog):
    supported_task_types = {"YoloV5", "MobileNet", "ResNet50"}
    nodes = [
        node for node in cluster_resources.get("result", [])
        if isinstance(node, dict)
    ]
    candidates = []
    for node in nodes:
        node_type = node.get("type")
        models = []
        for item in task_catalog.get("result", []):
            if not isinstance(item, dict):
                continue
            task_type = item.get("task_type")
            if item.get("device_type") != node_type or task_type not in supported_task_types:
                continue
            models.append(
                {
                    "task_type": task_type,
                    "model_name": item.get("model_name"),
                    "overhead": item.get("overhead"),
                }
            )
        if models:
            candidates.append(
                {
                    "target_global_id": node.get("global_id"),
                    "ip_address": node.get("ip_address"),
                    "device_type": node_type,
                    "resource": node.get("resource"),
                    "models": models,
                }
            )
    if not candidates:
        raise RuntimeError("no image execution candidates available on the current cluster")
    return {"status": "success", "result": candidates}


def find_matching_candidate(execution_candidates, task_type, target_global_id):
    if not task_type or not target_global_id:
        return None
    for node_item in execution_candidates.get("result", []):
        if not isinstance(node_item, dict) or node_item.get("target_global_id") != target_global_id:
            continue
        for model_item in node_item.get("models", []):
            if isinstance(model_item, dict) and model_item.get("task_type") == task_type:
                return {
                    "target_global_id": node_item.get("target_global_id"),
                    "ip_address": node_item.get("ip_address"),
                    "device_type": node_item.get("device_type"),
                    "resource": node_item.get("resource"),
                    "task_type": model_item.get("task_type"),
                    "model_name": model_item.get("model_name"),
                    "overhead": model_item.get("overhead"),
                }
    return None


def flatten_execution_candidates(execution_candidates):
    flattened = []
    for node_item in execution_candidates.get("result", []):
        if not isinstance(node_item, dict):
            continue
        for model_item in node_item.get("models", []):
            if not isinstance(model_item, dict):
                continue
            flattened.append(
                {
                    "target_global_id": node_item.get("target_global_id"),
                    "ip_address": node_item.get("ip_address"),
                    "device_type": node_item.get("device_type"),
                    "resource": node_item.get("resource"),
                    "task_type": model_item.get("task_type"),
                    "model_name": model_item.get("model_name"),
                    "overhead": model_item.get("overhead") or {},
                }
            )
    return flattened


def choose_fallback_candidate(user_text, execution_candidates):
    candidates = flatten_execution_candidates(execution_candidates)
    if not candidates:
        raise RuntimeError("no image execution candidates available for fallback selection")

    normalized_text = user_text.lower()
    wants_detection = bool(re.search(r"检测|目标|物体|框|detect|detection|object", normalized_text))
    wants_speed = bool(re.search(r"快|速度|fast|faster|speed", normalized_text))
    wants_classification = bool(re.search(r"分类|种类|类别|class|classification", normalized_text))

    def by_proc_time(item):
        overhead = item.get("overhead") or {}
        return float(overhead.get("proc_time", 1e9))

    if wants_detection:
        detection_candidates = [item for item in candidates if item.get("task_type") == "YoloV5"]
        if detection_candidates:
            return min(detection_candidates, key=by_proc_time)

    if wants_speed or wants_classification:
        classification_candidates = [
            item for item in candidates
            if item.get("task_type") in {"MobileNet", "ResNet50"}
        ]
        if classification_candidates:
            return min(classification_candidates, key=by_proc_time)

    return min(
        candidates,
        key=lambda item: (
            float((item.get("resource") or {}).get("mem_used", 0.0)),
            by_proc_time(item),
        )
    )


def choose_execution_target(user_text, image_name, execution_candidates):
    try:
        model_result = call_llm(build_model_selection_messages(user_text, image_name, execution_candidates))
        raw_content = model_result["content"].strip()
    except Exception as exc:
        model_result = {
            "content": "",
            "prompt_tokens": 0,
            "completion_tokens": 0,
        }
        raw_content = ""
        print(f"image agent model selection failed: {exc}", flush=True)

    print(f"image agent raw model selection: {raw_content}", flush=True)
    try:
        selection = parse_json_object(raw_content)
    except Exception:
        selection = {}

    task_type = selection.get("task_type")
    target_global_id = selection.get("target_global_id")
    matched_candidate = find_matching_candidate(execution_candidates, task_type, target_global_id)
    if matched_candidate is None:
        matched_candidate = choose_fallback_candidate(user_text, execution_candidates)
        selection = {
            "task_type": matched_candidate["task_type"],
            "target_global_id": matched_candidate["target_global_id"],
            "reason": "model selection did not return a valid execution pair; used deterministic fallback",
        }
        print(f"image agent fallback selection: {json.dumps(selection, ensure_ascii=False)}", flush=True)

    selection["candidate"] = matched_candidate
    return selection, model_result


def build_deterministic_summary(selected_execution, tool_result):
    task_type = selected_execution.get("task_type")
    results = tool_result.get("results") if isinstance(tool_result, dict) else None
    result = tool_result.get("result") if isinstance(tool_result, dict) else None
    exec_time = tool_result.get("exec_time") if isinstance(tool_result, dict) else None
    if exec_time is None and isinstance(tool_result, dict):
        exec_time = tool_result.get("gateway_time")

    if task_type in {"MobileNet", "ResNet50"} and isinstance(results, list) and results:
        top_result = results[0]
        if isinstance(top_result, dict):
            class_name = top_result.get("class")
            confidence = top_result.get("confidence")
            if class_name:
                parts = [f"图像分类结果：最可能是 {class_name}"]
                if isinstance(confidence, (int, float)):
                    parts.append(f"置信度 {confidence * 100:.2f}%")
                if isinstance(exec_time, (int, float)):
                    parts.append(f"耗时 {exec_time:.2f} ms")
                parts.append(f"使用模型：{task_type}")
                return "，".join(parts) + "。"

    if task_type in {"MobileNet", "ResNet50"} and isinstance(result, dict):
        index = result.get("index")
        score = result.get("score")
        parts = ["图像分类完成"]
        if index is not None:
            parts.append(f"分类索引为 {index}")
        if isinstance(score, (int, float)):
            parts.append(f"置信度 {score * 100:.2f}%")
        if isinstance(exec_time, (int, float)):
            parts.append(f"耗时 {exec_time:.2f} ms")
        parts.append(f"使用模型：{task_type}")
        return "，".join(parts) + "。"

    if task_type == "YoloV5" and isinstance(results, list):
        parts = [f"目标检测完成，共检测到 {len(results)} 个结果"]
        if isinstance(exec_time, (int, float)):
            parts.append(f"耗时 {exec_time:.2f} ms")
        parts.append("使用模型：YoloV5")
        return "，".join(parts) + "。"

    return None


def summarize_result(user_text, selected_execution, tool_result):
    deterministic_summary = build_deterministic_summary(selected_execution, tool_result)
    if deterministic_summary:
        return deterministic_summary, {
            "prompt_tokens": 0,
            "completion_tokens": 0,
        }

    model_result = call_llm(build_result_summary_messages(user_text, selected_execution, tool_result))
    content = model_result["content"].strip()
    if not content:
        return f"图像任务已完成，使用模型：{selected_execution.get('task_type', 'unknown')}。", model_result
    return content, model_result


def run_image_agent(user_text, image_path):
    total_prompt_tokens = 0
    total_completion_tokens = 0

    cluster_resources = json.loads(execute_tool("get_cluster_resources", timeout=30))
    available_device_types = sorted(
        {
            node.get("type")
            for node in cluster_resources.get("result", [])
            if isinstance(node, dict) and node.get("type")
        }
    )
    task_catalog = json.loads(
        execute_tool(
            "get_task_catalog",
            {"available_device_types": available_device_types},
            timeout=30,
        )
    )
    execution_candidates = build_execution_candidates(cluster_resources, task_catalog)

    image_name = os.path.basename(image_path)
    selection, selection_usage = choose_execution_target(user_text, image_name, execution_candidates)
    total_prompt_tokens += selection_usage["prompt_tokens"]
    total_completion_tokens += selection_usage["completion_tokens"]

    selected_execution = {
        "task_type": selection["task_type"],
        "target_global_id": selection["target_global_id"],
        "real_url": "predict",
        "reason": selection.get("reason", ""),
        "candidate": selection["candidate"],
    }

    tool_result = json.loads(
        execute_tool(
            "run_vision_task_on_node",
            {
                "task_type": selection["task_type"],
                "target_global_id": selection["target_global_id"],
                "image_path": image_path,
                "real_url": "predict",
                "file_field_name": "image",
            },
            timeout=150,
        )
    )

    message, summary_usage = summarize_result(user_text, selected_execution, tool_result)
    total_prompt_tokens += summary_usage["prompt_tokens"]
    total_completion_tokens += summary_usage["completion_tokens"]

    return {
        "message": message,
        "selected_node": selected_execution,
        "tool_result": tool_result,
        "usage": {
            "prompt_tokens": total_prompt_tokens,
            "completion_tokens": total_completion_tokens,
            "total_tokens": total_prompt_tokens + total_completion_tokens,
        },
    }


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
