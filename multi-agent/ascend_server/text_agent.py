import argparse
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
        for item in task_catalog.get("result", []):
            if not isinstance(item, dict):
                continue
            task_type = item.get("task_type")
            if item.get("device_type") != node_type or task_type not in supported_task_types:
                continue
            candidates.append(
                {
                    "target_global_id": node.get("global_id"),
                    "device_type": node_type,
                    "resource": node.get("resource"),
                    "task_type": task_type,
                    "model_name": item.get("model_name"),
                    "overhead": item.get("overhead"),
                }
            )

    if not candidates:
        raise RuntimeError("No text execution candidates available for the current cluster")
    return candidates


def build_execution_selection_messages(user_text, execution_candidates, sub_agent_profile):
    compact_profile = compact_sub_agent_profile(sub_agent_profile)
    return [
        {
            "role": "system",
            "content": (
                "You are a text agent.\n"
                "Your job is to choose the best text model execution target for the request.\n"
                "Use only the provided execution candidates and sub agent profile.\n"
                "Return JSON only with this schema:\n"
                "{\"task_type\":\"Bert\",\"target_global_id\":\"<uuid>\",\"reason\":\"<short reason>\"}\n"
                "Rules:\n"
                "1. task_type and target_global_id must come from execution_candidates.\n"
                "2. Prefer candidates that satisfy the task intent with enough remaining resources and lower overhead.\n"
                "3. Never output anything except a single JSON object."
            ),
        },
        {
            "role": "user",
            "content": (
                f"user_text:\n{user_text}\n\n"
                f"sub_agent_profile:\n{json.dumps(compact_profile, ensure_ascii=False)}\n\n"
                f"execution_candidates:\n{json.dumps(execution_candidates, ensure_ascii=False)}"
            ),
        },
    ]


def choose_execution_target(user_text, execution_candidates, sub_agent_profile):
    model_result = call_llm(
        build_execution_selection_messages(user_text, execution_candidates, sub_agent_profile),
        llm_api_url=LLM_API_URL,
        model_name=LLM_MODEL_NAME,
    )
    raw_content = model_result["content"].strip()
    print(f"text agent raw model selection: {raw_content}", flush=True)
    selection = parse_json_object(raw_content)
    task_type = selection.get("task_type")
    target_global_id = selection.get("target_global_id")

    candidate = None
    for item in execution_candidates:
        if item.get("task_type") == task_type and item.get("target_global_id") == target_global_id:
            candidate = item
            break
    if candidate is None:
        raise RuntimeError(
            f"text agent selected unsupported execution pair: task_type={task_type}, target_global_id={target_global_id}"
        )

    return {
        "task_type": task_type,
        "target_global_id": target_global_id,
        "real_url": "textclassify",
        "reason": selection.get("reason", ""),
        "candidate": candidate,
    }, model_result


def summarize_tool_result(tool_result):
    if isinstance(tool_result, dict):
        prediction_label = tool_result.get("prediction_label")
        if isinstance(prediction_label, str) and prediction_label.strip():
            return f"文本分类结果是：{prediction_label.strip()}。"

        for key in ("message", "label", "prediction", "result", "output", "text"):
            value = tool_result.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

        result_value = tool_result.get("result")
        if isinstance(result_value, dict):
            prediction_label = result_value.get("prediction_label")
            if isinstance(prediction_label, str) and prediction_label.strip():
                return f"文本分类结果是：{prediction_label.strip()}。"

            for key in ("message", "label", "prediction", "output", "text"):
                value = result_value.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()

    return json.dumps(tool_result, ensure_ascii=False)


def run_text_sub_agent(user_text, target_node, sub_agent_profile):
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
    selected_execution, selection_usage = choose_execution_target(user_text, execution_candidates, sub_agent_profile)
    tool_result = json.loads(
        call_mcp_tool(
            "run_task_on_node",
            {
                "task_type": selected_execution["task_type"],
                "target_global_id": selected_execution["target_global_id"],
                "form_fields": {"text": user_text},
                "real_url": selected_execution["real_url"],
            },
            timeout=120,
        )
    )

    return {
        "message": summarize_tool_result(tool_result),
        "text_agent_node": target_node,
        "selected_execution": selected_execution,
        "tool_result": tool_result,
        "usage": {
            "prompt_tokens": selection_usage["prompt_tokens"],
            "completion_tokens": selection_usage["completion_tokens"],
            "total_tokens": selection_usage["prompt_tokens"] + selection_usage["completion_tokens"],
        },
    }


@app.route("/v1/sub-agents/text/execute", methods=["POST"])
def execute_text_task():
    global is_blocking

    if is_blocking:
        return jsonify({
            "error": {
                "message": "Sub text agent is busy! Please try again later.",
                "type": "server_error",
                "param": None,
                "code": None,
            }
        }), 503

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

    lock.acquire()
    try:
        is_blocking = True
        result = run_text_sub_agent(user_text, target_node, sub_agent_profile)
        return Response(json.dumps({
            "object": "sub_agent.text.execute",
            "message": result["message"],
            "text_agent_node": result["text_agent_node"],
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
    parser.add_argument("--port", type=int, default=8085)
    parser.add_argument("--llm_api_url", default=DEFAULT_LLM_API_URL)
    parser.add_argument("--llm_model_name", default=DEFAULT_LLM_MODEL_NAME)
    args = parser.parse_args()

    LLM_API_URL = args.llm_api_url
    LLM_MODEL_NAME = args.llm_model_name

    app.run(host=args.host, port=args.port, threaded=True, debug=False)
