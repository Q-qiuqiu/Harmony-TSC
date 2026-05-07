import argparse
import base64
import json
import re
import threading
from flask import Flask, Response, jsonify, request

from agent_tools import (
    DEFAULT_LLM_API_URL,
    DEFAULT_LLM_MODEL_NAME,
    call_llm,
    call_mcp_tool,
    parse_json_object,
)


app = Flask(__name__)

lock = threading.Lock()
is_blocking = False
LLM_API_URL = DEFAULT_LLM_API_URL
LLM_MODEL_NAME = DEFAULT_LLM_MODEL_NAME


def compact_sub_agent_profile(sub_agent_profile):
    tools = sub_agent_profile.get("tools", {})
    compact_tools = []
    for tool_name, tool_info in tools.items():
        if not isinstance(tool_info, dict):
            continue
        compact_tools.append(
            {
                "tool": tool_name,
                "task_type": tool_info.get("task_type"),
                "capabilities": tool_info.get("capabilities", []),
            }
        )
    return {"tools": compact_tools}


def build_model_selection_messages(user_text, image_name, target_node, task_catalog, sub_agent_profile):
    profile_json = json.dumps(compact_sub_agent_profile(sub_agent_profile), ensure_ascii=False)
    return [
        {
            "role": "system",
            "content": (
                "You are a sub image agent.\n"
                "A main agent has already selected the target board for you.\n"
                "Your current task is to choose the most suitable vision model and the most suitable execution board for the request.\n"
                "Use the provided execution candidates and sub agent profile as the source of truth.\n"
                "Sub agent profile:\n"
                f"{profile_json}\n"
                "Return JSON only with this schema:\n"
                "{\"task_type\":\"YoloV5\",\"target_global_id\":\"<uuid>\",\"reason\":\"<short reason>\"}\n"
                "Rules:\n"
                "1. task_type and target_global_id must be chosen from the provided execution candidates.\n"
                "2. YoloV5 is mainly for object detection and bounding box coordinates, not for image classification.\n"
                "3. MobileNet and ResNet50 are for image classification.\n"
                "4. For classification requests, prefer MobileNet when speed is important; use ResNet50 when accuracy is preferred over speed.\n"
                "5. Prefer lower expected overhead when it still satisfies the task intent.\n"
                "6. Never output anything except a single JSON object."
            ),
        },
        {
            "role": "user",
            "content": (
                f"user_text:\n{user_text}\n\n"
                f"image_name:\n{image_name}\n\n"
                f"image_agent_node:\n{json.dumps(target_node, ensure_ascii=False)}\n\n"
                f"execution_candidates:\n{json.dumps(task_catalog, ensure_ascii=False)}"
            ),
        },
    ]


def build_result_summary_messages(user_text, selected_execution, tool_result):
    return [
        {
            "role": "system",
            "content": (
                "You are a sub image agent.\n"
                "Your task is to summarize the actual vision execution result in concise Chinese for the end user.\n"
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


def build_execution_candidates(cluster_resources, task_catalog, sub_agent_profile):
    tools = sub_agent_profile.get("tools", {})
    supported_task_types = {
        tool_info.get("task_type")
        for tool_info in tools.values()
        if isinstance(tool_info, dict) and tool_info.get("task_type")
    }
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
        raise RuntimeError("no execution candidates available for the sub agent tools on the current cluster")
    return {"status": "success", "result": candidates}


def choose_execution_target(user_text, image_name, target_node, execution_candidates, sub_agent_profile):
    model_result = call_llm(
        build_model_selection_messages(user_text, image_name, target_node, execution_candidates, sub_agent_profile),
        llm_api_url=LLM_API_URL,
        model_name=LLM_MODEL_NAME,
        max_tokens=64,
    )
    raw_content = model_result["content"].strip()
    print(f"image agent raw model selection: {raw_content}", flush=True)
    try:
        selection = parse_json_object(raw_content)
    except json.JSONDecodeError:
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
        raise RuntimeError("no execution candidates available for fallback selection")

    normalized_text = user_text.lower()
    wants_detection = bool(re.search(r"检测|目标|物体|框|detect|detection|object", normalized_text))
    wants_speed = bool(re.search(r"快|速度|fast|faster|speed", normalized_text))
    wants_classification = bool(re.search(r"分类|种类|类别|class|classification", normalized_text))

    def by_proc_time(item):
        overhead = item.get("overhead") or {}
        return float(overhead.get("proc_time", 1e9))

    if wants_detection:
        for item in candidates:
            if item.get("task_type") == "YoloV5":
                return item

    if wants_speed or wants_classification:
        classification_candidates = [
            item for item in candidates
            if item.get("task_type") in {"MobileNet", "ResNet50"}
        ]
        if classification_candidates:
            return min(classification_candidates, key=by_proc_time)

    return min(candidates, key=by_proc_time)


def summarize_result(user_text, selected_execution, tool_result):
    model_result = call_llm(
        build_result_summary_messages(user_text, selected_execution, tool_result),
        llm_api_url=LLM_API_URL,
        model_name=LLM_MODEL_NAME,
    )
    content = model_result["content"].strip()
    print(f"image agent raw model summary: {content}", flush=True)
    if not content:
        raise RuntimeError("sub image agent returned empty summary")
    return content, model_result


def run_image_sub_agent(user_text, image_name, image_b64, target_node, sub_agent_profile):
    total_prompt_tokens = 0
    total_completion_tokens = 0

    cluster_resources = json.loads(call_mcp_tool("get_cluster_resources"))
    available_device_types = sorted(
        {
            node.get("type")
            for node in cluster_resources.get("result", [])
            if isinstance(node, dict) and node.get("type")
        }
    )
    task_catalog = json.loads(
        call_mcp_tool(
            "get_task_catalog",
            {"available_device_types": available_device_types},
        )
    )
    execution_candidates = build_execution_candidates(cluster_resources, task_catalog, sub_agent_profile)

    selection, selection_usage = choose_execution_target(
        user_text,
        image_name,
        target_node,
        execution_candidates,
        sub_agent_profile,
    )
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
        call_mcp_tool(
            "run_vision_task_on_node",
            {
                "task_type": selection["task_type"],
                "target_global_id": selection["target_global_id"],
                "image_b64": image_b64,
                "image_name": image_name,
                "real_url": "predict",
                "file_field_name": "image",
            },
            timeout=60,
        )
    )

    message, summary_usage = summarize_result(user_text, selected_execution, tool_result)
    total_prompt_tokens += summary_usage["prompt_tokens"]
    total_completion_tokens += summary_usage["completion_tokens"]

    return {
        "message": message,
        "selected_execution": selected_execution,
        "tool_result": tool_result,
        "usage": {
            "prompt_tokens": total_prompt_tokens,
            "completion_tokens": total_completion_tokens,
            "total_tokens": total_prompt_tokens + total_completion_tokens,
        },
    }


@app.route("/v1/sub-agents/image/execute", methods=["POST"])
def execute_image_task():
    global is_blocking

    if is_blocking:
        return jsonify({
            "error": {
                "message": "Sub image agent is busy! Please try again later.",
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

    user_text = (request.form.get("user_text") or "").strip()
    target_node_raw = request.form.get("target_node")
    sub_agent_profile_raw = request.form.get("sub_agent_profile")
    if not user_text or not target_node_raw or not sub_agent_profile_raw:
        return jsonify({
            "error": {
                "message": "user_text, target_node and sub_agent_profile are required",
                "type": "invalid_request_error",
                "param": None,
                "code": None,
            }
        }), 400

    try:
        target_node = json.loads(target_node_raw)
        sub_agent_profile = json.loads(sub_agent_profile_raw)
    except json.JSONDecodeError as exc:
        return jsonify({
            "error": {
                "message": f"Invalid JSON form field: {exc}",
                "type": "invalid_request_error",
                "param": None,
                "code": None,
            }
        }), 400

    if not isinstance(target_node, dict) or not isinstance(sub_agent_profile, dict):
        return jsonify({
            "error": {
                "message": "target_node and sub_agent_profile must be JSON objects",
                "type": "invalid_request_error",
                "param": None,
                "code": None,
            }
        }), 400

    image_bytes = image_file.read()
    image_b64 = base64.b64encode(image_bytes).decode("ascii")
    image_name = image_file.filename or "upload.bin"

    lock.acquire()
    try:
        is_blocking = True
        result = run_image_sub_agent(user_text, image_name, image_b64, target_node, sub_agent_profile)
        return Response(json.dumps({
            "object": "sub_agent.image.execute",
            "message": result["message"],
            "selected_execution": result["selected_execution"],
            "tool_result": result["tool_result"],
            "usage": result["usage"],
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
        lock.release()
        is_blocking = False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8083)
    parser.add_argument("--llm_api_url", default=DEFAULT_LLM_API_URL)
    parser.add_argument("--llm_model_name", default=DEFAULT_LLM_MODEL_NAME)
    args = parser.parse_args()

    LLM_API_URL = args.llm_api_url
    LLM_MODEL_NAME = args.llm_model_name

    app.run(host=args.host, port=args.port, threaded=True, debug=False)
