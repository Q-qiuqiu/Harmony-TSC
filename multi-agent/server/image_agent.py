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


def build_model_selection_messages(user_text, image_name, target_node, task_catalog, sub_agent_profile):
    profile_json = json.dumps(sub_agent_profile, ensure_ascii=False)
    return [
        {
            "role": "system",
            "content": (
                "You are a sub image agent.\n"
                "A main agent has already selected the target board for you.\n"
                "Your current task is only to choose the most suitable vision model for the request.\n"
                "Use the provided task catalog and sub agent profile as the source of truth.\n"
                "Sub agent profile:\n"
                f"{profile_json}\n"
                "Return JSON only with this schema:\n"
                "{\"task_type\":\"YoloV5\",\"reason\":\"<short reason>\"}\n"
                "Rules:\n"
                "1. task_type must be chosen from the provided task catalog.\n"
                "2. For generic multi-object detection requests, YoloV5 is usually appropriate.\n"
                "3. For lightweight classification-style requests where speed is critical, MobileNet or ResNet50 may be more suitable.\n"
                "4. Never output anything except a single JSON object."
            ),
        },
        {
            "role": "user",
            "content": (
                f"user_text:\n{user_text}\n\n"
                f"image_name:\n{image_name}\n\n"
                f"target_node:\n{json.dumps(target_node, ensure_ascii=False)}\n\n"
                f"task_catalog:\n{json.dumps(task_catalog, ensure_ascii=False)}"
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


def choose_task_type(user_text, image_name, target_node, task_catalog, sub_agent_profile):
    model_result = call_llm(
        build_model_selection_messages(user_text, image_name, target_node, task_catalog, sub_agent_profile),
        llm_api_url=LLM_API_URL,
        model_name=LLM_MODEL_NAME,
    )
    selection = parse_json_object(model_result["content"])
    task_type = selection.get("task_type")
    if not task_type:
        raise RuntimeError("sub image agent model selection missing task_type")
    available_task_types = {
        item.get("task_type")
        for item in task_catalog.get("result", [])
        if isinstance(item, dict)
    }
    if task_type not in available_task_types:
        raise RuntimeError(f"sub image agent selected unsupported task_type: {task_type}")
    return selection, model_result


def summarize_result(user_text, selected_execution, tool_result):
    model_result = call_llm(
        build_result_summary_messages(user_text, selected_execution, tool_result),
        llm_api_url=LLM_API_URL,
        model_name=LLM_MODEL_NAME,
    )
    content = model_result["content"].strip()
    if not content:
        raise RuntimeError("sub image agent returned empty summary")
    return content, model_result


def run_image_sub_agent(user_text, image_name, image_b64, target_node, sub_agent_profile):
    total_prompt_tokens = 0
    total_completion_tokens = 0

    task_catalog = json.loads(
        call_mcp_tool(
            "get_task_catalog",
            {"available_device_types": [target_node["type"]]},
        )
    )

    selection, selection_usage = choose_task_type(
        user_text,
        image_name,
        target_node,
        task_catalog,
        sub_agent_profile,
    )
    total_prompt_tokens += selection_usage["prompt_tokens"]
    total_completion_tokens += selection_usage["completion_tokens"]

    selected_execution = {
        "task_type": selection["task_type"],
        "target_global_id": target_node["global_id"],
        "real_url": "predict",
        "reason": selection.get("reason", ""),
    }

    tool_result = json.loads(
        call_mcp_tool(
            "run_vision_task_on_node",
            {
                "task_type": selection["task_type"],
                "target_global_id": target_node["global_id"],
                "image_b64": image_b64,
                "image_name": image_name,
                "real_url": "predict",
                "file_field_name": "image",
            },
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
