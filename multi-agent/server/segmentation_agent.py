import argparse
import base64
import json
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
SEGMENTATION_TASK_TYPE = "deeplabv3"


def build_execution_candidates(cluster_resources, task_catalog, sub_agent_profile):
    tools = sub_agent_profile.get("tools", {})
    supported_task_types = {
        tool_info.get("task_type")
        for tool_info in tools.values()
        if isinstance(tool_info, dict) and tool_info.get("task_type")
    }
    candidates = []
    for node in cluster_resources.get("result", []):
        if not isinstance(node, dict):
            continue
        node_type = node.get("type")
        for item in task_catalog.get("result", []):
            if not isinstance(item, dict):
                continue
            task_type = item.get("task_type")
            if item.get("device_type") != node_type or task_type not in supported_task_types:
                continue
            candidates.append({
                "target_global_id": node.get("global_id"),
                "ip_address": node.get("ip_address"),
                "device_type": node_type,
                "resource": node.get("resource"),
                "task_type": task_type,
                "model_name": item.get("model_name"),
                "overhead": item.get("overhead"),
            })
    if not candidates:
        raise RuntimeError("no segmentation execution candidates available for the current cluster")
    return candidates


def build_execution_selection_messages(user_text, image_name, execution_candidates, sub_agent_profile):
    return [
        {
            "role": "system",
            "content": (
                "You are a segmentation agent.\n"
                "Your job is to choose the best deeplabv3 execution target for the image segmentation request.\n"
                "Use only the provided execution candidates and sub agent profile.\n"
                "Return JSON only with this schema:\n"
                "{\"task_type\":\"deeplabv3\",\"target_global_id\":\"<uuid>\",\"reason\":\"<short reason>\"}\n"
                "Rules:\n"
                "1. task_type and target_global_id must come from execution_candidates.\n"
                "2. Prefer candidates that satisfy the task with enough remaining resources and lower overhead.\n"
                "3. Never output anything except a single JSON object."
            ),
        },
        {
            "role": "user",
            "content": (
                f"user_text:\n{user_text}\n\n"
                f"image_name:\n{image_name}\n\n"
                f"sub_agent_profile:\n{json.dumps(sub_agent_profile, ensure_ascii=False)}\n\n"
                f"execution_candidates:\n{json.dumps(execution_candidates, ensure_ascii=False)}"
            ),
        },
    ]


def choose_execution_target(user_text, image_name, execution_candidates, sub_agent_profile):
    model_result = call_llm(
        build_execution_selection_messages(user_text, image_name, execution_candidates, sub_agent_profile),
        llm_api_url=LLM_API_URL,
        model_name=LLM_MODEL_NAME,
    )
    try:
        selection = parse_json_object(model_result["content"])
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"segmentation agent selection returned invalid JSON: {model_result['content']}") from exc
    task_type = selection.get("task_type")
    target_global_id = selection.get("target_global_id")

    candidate = None
    for item in execution_candidates:
        if item.get("task_type") == task_type and item.get("target_global_id") == target_global_id:
            candidate = item
            break
    if candidate is None:
        raise RuntimeError(
            f"segmentation agent selected unsupported execution pair: task_type={task_type}, target_global_id={target_global_id}"
        )

    return {
        "task_type": task_type,
        "target_global_id": target_global_id,
        "real_url": "segmentation",
        "reason": selection.get("reason", ""),
        "candidate": candidate,
    }, model_result


def redact_large_binary_fields(value):
    if isinstance(value, dict):
        redacted = {}
        for key, item in value.items():
            if key.endswith("_b64") or key in {"binary_b64", "image_b64"}:
                redacted[key] = "<binary image data omitted>"
            else:
                redacted[key] = redact_large_binary_fields(item)
        return redacted
    if isinstance(value, list):
        return [redact_large_binary_fields(item) for item in value]
    return value


def build_summary_messages(user_text, selected_execution, tool_result):
    summarized_tool_result = redact_large_binary_fields(tool_result)
    return [
        {
            "role": "system",
            "content": (
                "You are a segmentation agent.\n"
                "Summarize the segmentation execution result in concise Chinese.\n"
                "If an output image is available, mention that the segmentation image has been returned.\n"
                "Return plain Chinese text only."
            ),
        },
        {
            "role": "user",
            "content": (
                f"user_text:\n{user_text}\n\n"
                f"selected_execution:\n{json.dumps(selected_execution, ensure_ascii=False)}\n\n"
                f"tool_result:\n{json.dumps(summarized_tool_result, ensure_ascii=False)}"
            ),
        },
    ]


def summarize_result(user_text, selected_execution, tool_result):
    model_result = call_llm(
        build_summary_messages(user_text, selected_execution, tool_result),
        llm_api_url=LLM_API_URL,
        model_name=LLM_MODEL_NAME,
    )
    content = model_result["content"].strip()
    if not content:
        raise RuntimeError("segmentation agent returned empty summary")
    return content, model_result


def extract_result_image(tool_result):
    if not isinstance(tool_result, dict):
        return None
    for key in ("image_b64", "result_image_b64", "output_image_b64", "binary_b64"):
        value = tool_result.get(key)
        if isinstance(value, str) and value:
            return {
                "image_b64": value,
                "content_type": tool_result.get("content_type", "image/png"),
            }
    result = tool_result.get("result")
    if isinstance(result, dict):
        return extract_result_image(result)
    return None


def run_segmentation_sub_agent(user_text, image_name, image_b64, target_node, sub_agent_profile):
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
        call_mcp_tool("get_task_catalog", {"available_device_types": available_device_types})
    )

    execution_candidates = build_execution_candidates(cluster_resources, task_catalog, sub_agent_profile)
    selected_execution, selection_usage = choose_execution_target(
        user_text,
        image_name,
        execution_candidates,
        sub_agent_profile,
    )
    total_prompt_tokens += selection_usage["prompt_tokens"]
    total_completion_tokens += selection_usage["completion_tokens"]
    if selected_execution["task_type"] != SEGMENTATION_TASK_TYPE:
        raise RuntimeError(f"segmentation agent selected unsupported task_type: {selected_execution['task_type']}")

    tool_result_raw = call_mcp_tool(
        "run_vision_task_on_node",
        {
            "task_type": SEGMENTATION_TASK_TYPE,
            "target_global_id": selected_execution["target_global_id"],
            "image_b64": image_b64,
            "image_name": image_name,
            "real_url": selected_execution["real_url"],
            "file_field_name": "image",
        },
        timeout=180,
    )
    try:
        tool_result = json.loads(tool_result_raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"deeplabv3 gateway returned invalid JSON: {tool_result_raw[:500]}") from exc

    message, summary_usage = summarize_result(user_text, selected_execution, tool_result)
    total_prompt_tokens += summary_usage["prompt_tokens"]
    total_completion_tokens += summary_usage["completion_tokens"]

    return {
        "message": message,
        "segmentation_agent_node": target_node,
        "selected_execution": selected_execution,
        "tool_result": tool_result,
        "result_image": extract_result_image(tool_result),
        "usage": {
            "prompt_tokens": total_prompt_tokens,
            "completion_tokens": total_completion_tokens,
            "total_tokens": total_prompt_tokens + total_completion_tokens,
        },
    }


@app.route("/v1/sub-agents/segmentation/execute", methods=["POST"])
def execute_segmentation_task():
    global is_blocking

    if is_blocking:
        return jsonify({
            "error": {
                "message": "Sub segmentation agent is busy! Please try again later.",
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

    image_b64 = base64.b64encode(image_file.read()).decode("ascii")
    image_name = image_file.filename or "upload.bin"

    lock.acquire()
    try:
        is_blocking = True
        result = run_segmentation_sub_agent(user_text, image_name, image_b64, target_node, sub_agent_profile)
        return Response(json.dumps({
            "object": "sub_agent.segmentation.execute",
            "message": result["message"],
            "segmentation_agent_node": result["segmentation_agent_node"],
            "selected_execution": result["selected_execution"],
            "tool_result": result["tool_result"],
            "result_image": result["result_image"],
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
    parser.add_argument("--port", type=int, default=8086)
    parser.add_argument("--llm_api_url", default=DEFAULT_LLM_API_URL)
    parser.add_argument("--llm_model_name", default=DEFAULT_LLM_MODEL_NAME)
    args = parser.parse_args()

    LLM_API_URL = args.llm_api_url
    LLM_MODEL_NAME = args.llm_model_name

    app.run(host=args.host, port=args.port, threaded=True, debug=False)
