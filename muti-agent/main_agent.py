import argparse
import json
import os
import threading
from flask import Flask, Response, jsonify, request

from agent_tools import (
    DEFAULT_IMAGE_AGENT_URL,
    DEFAULT_LLM_API_URL,
    DEFAULT_LLM_MODEL_NAME,
    call_llm,
    call_mcp_tool,
    extract_text_fields_from_form,
    get_sub_agent_profile,
    load_sub_agent_catalog,
    choose_user_text,
    parse_json_object,
    post_json,
    save_uploaded_file,
)


app = Flask(__name__)

lock = threading.Lock()
is_blocking = False
LLM_API_URL = DEFAULT_LLM_API_URL
LLM_MODEL_NAME = DEFAULT_LLM_MODEL_NAME
IMAGE_AGENT_URL = DEFAULT_IMAGE_AGENT_URL


def build_main_agent_messages(user_text, cluster_resources, sub_agent_catalog):
    return [
        {
            "role": "system",
            "content": (
                "You are a main multi-agent scheduler.\n"
                "Your job is to decide which board should launch the requested sub agent, based on cluster resources and the sub agent startup overhead.\n"
                "The image sub agent can handle image reasoning and can later choose a board-local vision model via gateway tools.\n"
                "Return JSON only with this schema:\n"
                "{\"sub_agent\":\"image_agent\",\"target_global_id\":\"<uuid>\",\"reason\":\"<short reason>\"}\n"
                "Rules:\n"
                "1. target_global_id must come from cluster_resources.\n"
                "2. Consider sub agent startup overhead and current board resource usage.\n"
                "3. Prefer the board with enough remaining resources and lower expected impact.\n"
                "4. Never output anything except a single JSON object."
            ),
        },
        {
            "role": "user",
            "content": (
                f"user_text:\n{user_text}\n\n"
                f"cluster_resources:\n{json.dumps(cluster_resources, ensure_ascii=False)}\n\n"
                f"sub_agent_catalog:\n{json.dumps(sub_agent_catalog, ensure_ascii=False)}"
            ),
        },
    ]


def select_target_node_for_sub_agent(user_text, cluster_resources):
    sub_agent_catalog = load_sub_agent_catalog()
    model_result = call_llm(
        build_main_agent_messages(user_text, cluster_resources, sub_agent_catalog),
        llm_api_url=LLM_API_URL,
        model_name=LLM_MODEL_NAME,
    )
    selection = parse_json_object(model_result["content"])

    if selection.get("sub_agent") != "image_agent":
        raise RuntimeError("main agent selected an unsupported sub agent")

    target_global_id = selection.get("target_global_id")
    if not target_global_id:
        raise RuntimeError("main agent selection missing target_global_id")

    for node in cluster_resources.get("result", []):
        if isinstance(node, dict) and node.get("global_id") == target_global_id:
            selection["_usage"] = {
                "prompt_tokens": model_result["prompt_tokens"],
                "completion_tokens": model_result["completion_tokens"],
            }
            selection["target_node"] = node
            return selection

    raise RuntimeError("main agent selected a target_global_id not present in cluster_resources")


@app.route("/v1/multi-agent/chat", methods=["POST"])
def multi_agent_chat():
    global is_blocking

    if is_blocking:
        return jsonify({
            "error": {
                "message": "Main agent is busy! Please try again later.",
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
        temp_image_path = save_uploaded_file(image_file)

        cluster_resources = json.loads(call_mcp_tool("get_cluster_resources"))
        selection = select_target_node_for_sub_agent(user_text, cluster_resources)

        sub_agent_response_raw = post_json(
            IMAGE_AGENT_URL,
            {
                "user_text": user_text,
                "image_path": temp_image_path,
                "target_node": selection["target_node"],
                "sub_agent_profile": get_sub_agent_profile("image_agent"),
            },
            timeout=240,
        )
        sub_agent_response = json.loads(sub_agent_response_raw)
        if "error" in sub_agent_response:
            raise RuntimeError(sub_agent_response["error"].get("message", "sub image agent returned an error"))

        main_usage = selection["_usage"]
        sub_usage = sub_agent_response.get("usage", {})
        prompt_tokens = main_usage["prompt_tokens"] + sub_usage.get("prompt_tokens", 0)
        completion_tokens = main_usage["completion_tokens"] + sub_usage.get("completion_tokens", 0)

        return Response(json.dumps({
            "object": "multi_agent.chat",
            "message": sub_agent_response.get("message", ""),
            "main_agent_selection": {
                "sub_agent": selection["sub_agent"],
                "target_global_id": selection["target_global_id"],
                "reason": selection.get("reason", ""),
                "target_node": selection["target_node"],
            },
            "sub_agent_result": sub_agent_response,
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
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
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8084)
    parser.add_argument("--llm_api_url", default=DEFAULT_LLM_API_URL)
    parser.add_argument("--llm_model_name", default=DEFAULT_LLM_MODEL_NAME)
    parser.add_argument("--image_agent_url", default=DEFAULT_IMAGE_AGENT_URL)
    args = parser.parse_args()

    LLM_API_URL = args.llm_api_url
    LLM_MODEL_NAME = args.llm_model_name
    IMAGE_AGENT_URL = args.image_agent_url

    app.run(host=args.host, port=args.port, threaded=True, debug=False)
