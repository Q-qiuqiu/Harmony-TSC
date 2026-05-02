import argparse
import json
import threading
from flask import Flask, Response, jsonify, request

from agent_tools import call_mcp_tool


app = Flask(__name__)

lock = threading.Lock()
is_blocking = False


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
                    "ip_address": node.get("ip_address"),
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


def choose_execution_target(execution_candidates):
    ranked_candidates = sorted(
        execution_candidates,
        key=lambda item: (
            item.get("overhead", {}).get("xpu_usage", 1e9),
            item.get("overhead", {}).get("proc_time", 1e9),
        ),
    )
    candidate = ranked_candidates[0]
    return {
        "task_type": candidate["task_type"],
        "target_global_id": candidate["target_global_id"],
        "real_url": "textclassify",
        "reason": "text_agent selected the available Atlas Bert endpoint with the lowest declared overhead.",
        "candidate": candidate,
    }


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
    selected_execution = choose_execution_target(execution_candidates)
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
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
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
    args = parser.parse_args()
    app.run(host=args.host, port=args.port, threaded=True, debug=False)
