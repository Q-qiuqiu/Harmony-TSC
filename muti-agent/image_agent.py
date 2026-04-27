import argparse
import json
import os
import threading
from flask import Flask, Response, jsonify, request

from agent_tools import (
    DEFAULT_LLM_API_URL,
    DEFAULT_LLM_MODEL_NAME,
    call_llm,
    call_mcp_tool,
    list_mcp_tools,
    parse_json_object,
)


app = Flask(__name__)

lock = threading.Lock()
is_blocking = False
LLM_API_URL = DEFAULT_LLM_API_URL
LLM_MODEL_NAME = DEFAULT_LLM_MODEL_NAME


def build_agent_messages(user_text, image_path, target_node, tools):
    return build_agent_messages_with_profile(user_text, image_path, target_node, tools, {})


def build_agent_messages_with_profile(user_text, image_path, target_node, tools, sub_agent_profile):
    profile_json = json.dumps(sub_agent_profile, ensure_ascii=False)
    return [
        {
            "role": "system",
            "content": (
                "You are a sub image agent.\n"
                "A main agent has already selected the target board for you.\n"
                "Your job is to inspect the available vision task catalog for this board type, choose the most suitable model, call the execution tool, and then summarize the result in Chinese.\n"
                "Your static profile and tool capabilities are provided below and must be treated as the source of truth for what you can use.\n"
                "Use tools dynamically from the provided tool list.\n"
                "When using get_task_catalog, pass available_device_types containing only the selected node type.\n"
                "When using run_vision_task_on_node, you must reuse the exact target_global_id and image_path provided in the user context.\n"
                "For generic multi-object detection requests, YoloV5 is usually appropriate.\n"
                "For lightweight classification-style requests where speed is critical, MobileNet or ResNet50 may be more suitable.\n"
                "Sub agent profile:\n"
                f"{profile_json}\n"
                "Available tools:\n"
                f"{json.dumps(tools, ensure_ascii=False)}\n"
                "Rules:\n"
                "1. If you want to call a tool, reply with JSON only:\n"
                "{\"action\":\"tool_call\",\"tool_name\":\"<tool name>\",\"arguments\":{}}\n"
                "2. If you have enough information, reply with JSON only:\n"
                "{\"action\":\"final\",\"content\":\"<natural language chinese answer>\"}\n"
                "3. Never invent tool results.\n"
                "4. Never output anything except a single JSON object."
            ),
        },
        {
            "role": "user",
            "content": (
                f"user_text:\n{user_text}\n\n"
                f"image_path:\n{image_path}\n\n"
                f"target_node:\n{json.dumps(target_node, ensure_ascii=False)}"
            ),
        },
    ]


def run_image_sub_agent(user_text, image_path, target_node, sub_agent_profile, max_steps=6):
    working_messages = build_agent_messages_with_profile(
        user_text,
        image_path,
        target_node,
        list_mcp_tools(),
        sub_agent_profile,
    )
    total_prompt_tokens = 0
    total_completion_tokens = 0
    selected_execution = None
    last_tool_result = None

    for _ in range(max_steps):
        model_result = call_llm(working_messages, llm_api_url=LLM_API_URL, model_name=LLM_MODEL_NAME)
        total_prompt_tokens += model_result["prompt_tokens"]
        total_completion_tokens += model_result["completion_tokens"]

        try:
            agent_action = parse_json_object(model_result["content"])
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"sub image agent returned invalid JSON control output: {exc}")

        action = agent_action.get("action")
        if action == "final":
            content = agent_action.get("content", "").strip()
            if not content:
                raise RuntimeError("sub image agent returned empty final content")
            return {
                "message": content,
                "selected_execution": selected_execution,
                "tool_result": last_tool_result,
                "usage": {
                    "prompt_tokens": total_prompt_tokens,
                    "completion_tokens": total_completion_tokens,
                    "total_tokens": total_prompt_tokens + total_completion_tokens,
                },
            }

        if action != "tool_call":
            raise RuntimeError(f"unknown sub image agent action: {action}")

        tool_name = agent_action.get("tool_name")
        tool_arguments = agent_action.get("arguments", {})
        if not tool_name:
            raise RuntimeError("sub image agent tool_call is missing tool_name")

        if tool_name == "run_vision_task_on_node":
            tool_arguments["target_global_id"] = target_node["global_id"]
            tool_arguments["image_path"] = image_path
            tool_arguments.setdefault("real_url", "predict")
            tool_arguments.setdefault("file_field_name", "image")
        elif tool_name == "get_task_catalog":
            tool_arguments["available_device_types"] = [target_node["type"]]

        tool_result_text = call_mcp_tool(tool_name, tool_arguments)
        try:
            tool_result_value = json.loads(tool_result_text)
        except json.JSONDecodeError:
            tool_result_value = tool_result_text

        if tool_name == "run_vision_task_on_node":
            selected_execution = {
                "task_type": tool_arguments.get("task_type"),
                "target_global_id": target_node["global_id"],
                "real_url": tool_arguments.get("real_url", "predict"),
            }
            last_tool_result = tool_result_value

        working_messages.append({"role": "assistant", "content": model_result["content"]})
        working_messages.append({
            "role": "tool",
            "content": f"tool_name={tool_name}\nresult={json.dumps(tool_result_value, ensure_ascii=False)}"
        })

    raise RuntimeError("sub image agent exceeded max tool-calling steps without producing a final answer")


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

    data = request.get_json(silent=True)
    if not data:
        return jsonify({
            "error": {
                "message": "Invalid JSON request",
                "type": "invalid_request_error",
                "param": None,
                "code": None,
            }
        }), 400

    user_text = data.get("user_text", "").strip()
    image_path = data.get("image_path", "").strip()
    target_node = data.get("target_node")
    sub_agent_profile = data.get("sub_agent_profile")

    if not user_text or not image_path or not isinstance(target_node, dict) or not isinstance(sub_agent_profile, dict):
        return jsonify({
            "error": {
                "message": "user_text, image_path, target_node and sub_agent_profile are required",
                "type": "invalid_request_error",
                "param": None,
                "code": None,
            }
        }), 400

    if not os.path.exists(image_path):
        return jsonify({
            "error": {
                "message": f"image_path does not exist: {image_path}",
                "type": "invalid_request_error",
                "param": "image_path",
                "code": None,
            }
        }), 400

    lock.acquire()
    try:
        is_blocking = True
        result = run_image_sub_agent(user_text, image_path, target_node, sub_agent_profile)
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
