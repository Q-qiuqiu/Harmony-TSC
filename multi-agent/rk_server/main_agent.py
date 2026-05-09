import argparse
import json
import threading
from flask import Flask, Response, jsonify, request

from agent_tools import (
    DEFAULT_IMAGE_AGENT_URL,
    DEFAULT_LLM_API_URL,
    DEFAULT_LLM_MODEL_NAME,
    DEFAULT_SEGMENTATION_AGENT_URL,
    DEFAULT_TEXT_AGENT_URL,
    call_llm,
    call_mcp_tool,
    choose_user_text,
    extract_text_fields_from_form,
    get_sub_agent_profile,
    load_sub_agent_catalog,
    parse_json_object,
    post_multipart,
)


app = Flask(__name__)

lock = threading.Lock()
is_blocking = False
LLM_API_URL = DEFAULT_LLM_API_URL
LLM_MODEL_NAME = DEFAULT_LLM_MODEL_NAME
IMAGE_AGENT_URL = DEFAULT_IMAGE_AGENT_URL
TEXT_AGENT_URL = DEFAULT_TEXT_AGENT_URL
SEGMENTATION_AGENT_URL = DEFAULT_SEGMENTATION_AGENT_URL

SUB_AGENT_ENDPOINTS = {
    "image_agent": "/v1/sub-agents/image/execute",
    "text_agent": "/v1/sub-agents/text/execute",
    "segmentation_agent": "/v1/sub-agents/segmentation/execute",
}
SUPPORTED_SUB_AGENTS = set(SUB_AGENT_ENDPOINTS.keys())
DIRECT_CHAT_ROUTE = "chat"
UNSUPPORTED_TASK_ROUTE = "unsupported_task"
SUPPORTED_ROUTES = SUPPORTED_SUB_AGENTS | {DIRECT_CHAT_ROUTE, UNSUPPORTED_TASK_ROUTE}


def compact_sub_agent_catalog_for_selection(sub_agent_catalog):
    compact_catalog = {}
    for name, profile in sub_agent_catalog.items():
        if not isinstance(profile, dict):
            continue
        runtime = profile.get("runtime", {})
        if not isinstance(runtime, dict):
            runtime = {}
        compact_catalog[name] = {
            "supported_device_types": sorted(runtime.keys()),
            "supports_image_input": name in {"image_agent", "segmentation_agent"},
            "supports_text_input": True,
        }
    return compact_catalog


def compact_sub_agent_catalog_for_scheduling(sub_agent_catalog, sub_agent_name):
    profile = sub_agent_catalog.get(sub_agent_name, {})
    runtime = profile.get("runtime", {}) if isinstance(profile, dict) else {}
    compact_runtime = {}
    if isinstance(runtime, dict):
        for device_type, runtime_info in runtime.items():
            if not isinstance(runtime_info, dict):
                continue
            compact_runtime[device_type] = {
                "startup_timeout_sec": runtime_info.get("startup_timeout_sec"),
            }
    return {
        sub_agent_name: {
            "runtime": compact_runtime,
        }
    }


def compact_cluster_resources_for_scheduling(cluster_resources):
    compact_nodes = []
    for node in cluster_resources.get("result", []):
        if not isinstance(node, dict):
            continue
        compact_nodes.append(
            {
                "global_id": node.get("global_id"),
                "ip_address": node.get("ip_address"),
                "type": node.get("type"),
                "resource": node.get("resource", {}),
            }
        )
    return {
        "status": cluster_resources.get("status", "success"),
        "result": compact_nodes,
    }


def build_sub_agent_selection_messages(user_text, has_image, sub_agent_catalog):
    compact_catalog = compact_sub_agent_catalog_for_selection(sub_agent_catalog)
    return [
        {
            "role": "system",
            "content": (
                "You are a main multi-agent scheduler.\n"
                "Your first job is to decide whether the user request needs a sub agent or can be answered directly.\n"
                "Return JSON only with this schema:\n"
                "{\"route\":\"chat\",\"reason\":\"<short reason>\"}\n"
                "Rules:\n"
                "1. route must be one of: chat, unsupported_task, or one of the provided sub_agent_catalog keys.\n"
                "2. Choose chat for normal conversation, explanations, questions, greetings, or requests that do not need edge model execution.\n"
                "3. Choose text_agent only for supported text classification or text understanding model tasks.\n"
                "4. Choose segmentation_agent for image semantic segmentation requests when has_image is true.\n"
                "5. Choose image_agent for image detection, image classification, or image reasoning requests when has_image is true.\n"
                "6. Choose unsupported_task when the user asks for a model/tool task that no listed sub agent can handle, or when an image task lacks an image.\n"
                "7. Never output anything except a single JSON object."
            ),
        },
        {
            "role": "user",
            "content": (
                f"user_text:\n{user_text}\n\n"
                f"has_image:\n{json.dumps(has_image)}\n\n"
                f"sub_agent_catalog:\n{json.dumps(compact_catalog, ensure_ascii=False)}"
            ),
        },
    ]


def build_direct_chat_messages(user_text, reason=None):
    reason_text = reason or "direct chat"
    return [
        {
            "role": "system",
            "content": (
                "You are the main agent.\n"
                "Answer the user directly in concise natural Chinese.\n"
                "Do not mention sub agents, scheduling, JSON, node ids, URLs, or internal routing.\n"
                "If the user requested a task that the current system cannot execute, explain the limitation briefly and provide a helpful fallback answer."
            ),
        },
        {
            "role": "user",
            "content": f"user_text:\n{user_text}\n\nroute_reason:\n{reason_text}",
        },
    ]


def build_main_agent_messages(user_text, cluster_resources, sub_agent_catalog, sub_agent_name):
    compact_cluster_resources = compact_cluster_resources_for_scheduling(cluster_resources)
    compact_sub_agent_catalog = compact_sub_agent_catalog_for_scheduling(sub_agent_catalog, sub_agent_name)
    return [
        {
            "role": "system",
            "content": (
                "You are a main multi-agent scheduler.\n"
                f"Your job is to select which board should launch the {sub_agent_name}, based on cluster resources and sub agent startup overhead.\n"
                "Return JSON only with this schema:\n"
                "{\"target_global_id\":\"<uuid>\",\"reason\":\"<short reason>\"}\n"
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
                f"requested_sub_agent:\n{sub_agent_name}\n\n"
                f"cluster_resources:\n{json.dumps(compact_cluster_resources, ensure_ascii=False)}\n\n"
                f"sub_agent_catalog:\n{json.dumps(compact_sub_agent_catalog, ensure_ascii=False)}"
            ),
        },
    ]


def build_final_answer_messages(user_text, sub_agent_name, sub_agent_response):
    summarized_response = redact_large_binary_fields(sub_agent_response)
    compact_response = {
        "object": summarized_response.get("object"),
        "message": summarized_response.get("message"),
        "selected_execution": summarized_response.get("selected_execution"),
        "tool_result": summarized_response.get("tool_result"),
        "result_image": "<present>" if summarized_response.get("result_image") else None,
    }
    return [
        {
            "role": "system",
            "content": (
                "You are the main agent that returns the final answer to the user.\n"
                "Use the sub agent result as the source of truth.\n"
                "Return concise natural Chinese only.\n"
                "Do not include JSON, scheduling details, node ids, URLs, token usage, or debug fields.\n"
                "Summarize the actual task result for the user.\n"
                "When available, include the result, model name, and elapsed time."
            ),
        },
        {
            "role": "user",
            "content": (
                f"user_text:\n{user_text}\n\n"
                f"sub_agent:\n{sub_agent_name}\n\n"
                f"sub_agent_response:\n{json.dumps(compact_response, ensure_ascii=False)}"
            ),
        },
    ]


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


def make_final_answer(user_text, sub_agent_name, sub_agent_response):
    model_result = call_llm(
        build_final_answer_messages(user_text, sub_agent_name, sub_agent_response),
        llm_api_url=LLM_API_URL,
        model_name=LLM_MODEL_NAME,
    )
    content = model_result["content"].strip()
    if not content:
        content = sub_agent_response.get("message", "")
    return content, model_result


def make_direct_chat_answer(user_text, reason=None):
    model_result = call_llm(
        build_direct_chat_messages(user_text, reason),
        llm_api_url=LLM_API_URL,
        model_name=LLM_MODEL_NAME,
    )
    return model_result["content"].strip(), model_result


def parse_bool_form_value(value):
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def select_request_route(user_text, has_image, forced_sub_agent=None):
    if forced_sub_agent:
        if forced_sub_agent not in SUPPORTED_SUB_AGENTS:
            raise RuntimeError(f"unsupported sub_agent: {forced_sub_agent}")
        return {
            "route": forced_sub_agent,
            "sub_agent": forced_sub_agent,
            "reason": "sub_agent explicitly requested by client",
            "_usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
            },
        }

    sub_agent_catalog = load_sub_agent_catalog()
    model_result = call_llm(
        build_sub_agent_selection_messages(user_text, has_image, sub_agent_catalog),
        llm_api_url=LLM_API_URL,
        model_name=LLM_MODEL_NAME,
    )
    try:
        selection = parse_json_object(model_result["content"])
    except Exception:
        selection = {
            "route": DIRECT_CHAT_ROUTE,
            "reason": "route model did not return valid JSON; fallback to direct chat",
        }
    route = selection.get("route") or selection.get("sub_agent")
    if route not in SUPPORTED_ROUTES:
        route = DIRECT_CHAT_ROUTE
        selection["reason"] = f"unsupported route selected by model; fallback to direct chat"
    if route in {"image_agent", "segmentation_agent"} and not has_image:
        route = UNSUPPORTED_TASK_ROUTE
        selection["reason"] = f"{route} requires an uploaded image"
    selection["route"] = route
    if route in SUPPORTED_SUB_AGENTS:
        selection["sub_agent"] = route
    selection["_usage"] = {
        "prompt_tokens": model_result["prompt_tokens"],
        "completion_tokens": model_result["completion_tokens"],
    }
    return selection


def get_forced_sub_agent(text_fields):
    requested_sub_agent = (text_fields.get("sub_agent") or "").strip()
    if requested_sub_agent:
        if requested_sub_agent not in SUPPORTED_SUB_AGENTS:
            raise RuntimeError(f"unsupported sub_agent: {requested_sub_agent}")
        return requested_sub_agent
    return None


def filter_nodes_for_sub_agent(cluster_resources, sub_agent_name, excluded_global_ids=None):
    excluded_global_ids = set(excluded_global_ids or [])
    sub_agent_catalog = load_sub_agent_catalog()
    profile = sub_agent_catalog.get(sub_agent_name)
    if not isinstance(profile, dict):
        raise RuntimeError(f"sub agent profile not found: {sub_agent_name}")

    runtime = profile.get("runtime", {})
    if not isinstance(runtime, dict) or not runtime:
        raise RuntimeError(f"sub agent runtime config missing: {sub_agent_name}")

    supported_device_types = set(runtime.keys())
    filtered_nodes = [
        node
        for node in cluster_resources.get("result", [])
        if (
            isinstance(node, dict)
            and node.get("type") in supported_device_types
            and node.get("global_id") not in excluded_global_ids
        )
    ]

    if not filtered_nodes:
        raise RuntimeError(
            f"no registered nodes support sub agent {sub_agent_name}; supported device types: {sorted(supported_device_types)}"
        )

    return {
        "status": cluster_resources.get("status", "success"),
        "result": filtered_nodes,
    }


def select_target_node_for_sub_agent(user_text, cluster_resources, sub_agent_name, excluded_global_ids=None):
    sub_agent_catalog = load_sub_agent_catalog()
    filtered_cluster_resources = filter_nodes_for_sub_agent(cluster_resources, sub_agent_name, excluded_global_ids)
    model_result = call_llm(
        build_main_agent_messages(user_text, filtered_cluster_resources, sub_agent_catalog, sub_agent_name),
        llm_api_url=LLM_API_URL,
        model_name=LLM_MODEL_NAME,
    )
    selection = parse_json_object(model_result["content"])
    selection["sub_agent"] = sub_agent_name

    target_global_id = selection.get("target_global_id")
    if not target_global_id:
        raise RuntimeError("main agent selection missing target_global_id")

    for node in filtered_cluster_resources.get("result", []):
        if isinstance(node, dict) and node.get("global_id") == target_global_id:
            selection["_usage"] = {
                "prompt_tokens": model_result["prompt_tokens"],
                "completion_tokens": model_result["completion_tokens"],
            }
            selection["target_node"] = node
            return selection

    raise RuntimeError("main agent selected a target_global_id not present in cluster_resources")


def ensure_sub_agent_started(agent_name, target_global_id):
    start_result = json.loads(
        call_mcp_tool(
            "start_sub_agent",
            {
                "agent_name": agent_name,
                "target_global_id": target_global_id,
            },
            timeout=150,
        )
    )
    if start_result.get("status") != "success":
        raise RuntimeError(f"failed to start sub agent: {start_result}")
    return start_result


def resolve_sub_agent_url(agent_name, start_result):
    result = start_result.get("result", {})
    ip_address = result.get("ip_address")
    port = result.get("port")
    if ip_address and port:
        endpoint = SUB_AGENT_ENDPOINTS.get(agent_name)
        if not endpoint:
            raise RuntimeError(f"unknown sub agent endpoint mapping: {agent_name}")
        return f"http://{ip_address}:{port}{endpoint}"
    if agent_name == "image_agent":
        return IMAGE_AGENT_URL
    if agent_name == "text_agent":
        return TEXT_AGENT_URL
    if agent_name == "segmentation_agent":
        return SEGMENTATION_AGENT_URL
    raise RuntimeError(f"unknown sub agent fallback url: {agent_name}")


def invoke_sub_agent(selection, user_text, image_file, image_bytes=None):
    sub_agent_name = selection["sub_agent"]
    fields = {
        "user_text": user_text,
        "target_node": json.dumps(selection["target_node"], ensure_ascii=False),
        "sub_agent_profile": json.dumps(get_sub_agent_profile(sub_agent_name), ensure_ascii=False),
    }
    files = []

    if sub_agent_name in {"image_agent", "segmentation_agent"}:
        if image_file is None or not image_file.filename:
            raise RuntimeError(f"{sub_agent_name} requires an uploaded image")
        if image_bytes is None:
            image_bytes = image_file.read()
        files.append(
            {
                "field_name": "image",
                "filename": image_file.filename or "upload.bin",
                "content_type": image_file.content_type or "application/octet-stream",
                "content": image_bytes,
            }
        )
    elif sub_agent_name != "text_agent":
        raise RuntimeError(f"unsupported sub agent: {sub_agent_name}")

    start_result = ensure_sub_agent_started(sub_agent_name, selection["target_global_id"])
    sub_agent_url = resolve_sub_agent_url(sub_agent_name, start_result)
    sub_agent_response_raw = post_multipart(
        sub_agent_url,
        fields=fields,
        files=files,
        timeout=240,
    )
    sub_agent_response = json.loads(sub_agent_response_raw)
    if "error" in sub_agent_response:
        raise RuntimeError(sub_agent_response["error"].get("message", f"{sub_agent_name} returned an error"))
    return sub_agent_response, start_result, sub_agent_url


def invoke_sub_agent_with_failover(user_text, cluster_resources, sub_agent_name, image_file):
    excluded_global_ids = set()
    failures = []
    image_bytes = image_file.read() if image_file is not None and image_file.filename else None

    while True:
        selection = select_target_node_for_sub_agent(
            user_text,
            cluster_resources,
            sub_agent_name,
            excluded_global_ids,
        )
        try:
            sub_agent_response, start_result, sub_agent_url = invoke_sub_agent(
                selection,
                user_text,
                image_file,
                image_bytes,
            )
            return selection, sub_agent_response, start_result, sub_agent_url, failures
        except Exception as exc:
            failed_id = selection.get("target_global_id")
            failures.append({
                "target_global_id": failed_id,
                "reason": str(exc),
            })
            if failed_id:
                excluded_global_ids.add(failed_id)
            try:
                filter_nodes_for_sub_agent(cluster_resources, sub_agent_name, excluded_global_ids)
            except RuntimeError as no_candidates_exc:
                raise RuntimeError(
                    f"failed to start {sub_agent_name} on all candidate nodes: {failures}; {no_candidates_exc}"
                ) from exc


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
    text_fields = extract_text_fields_from_form(request.form)
    request_controls = dict(text_fields)
    request_controls.pop("sub_agent", None)
    debug_response = parse_bool_form_value(request_controls.pop("debug", None))
    if image_file is None and not request_controls:
        return jsonify({
            "error": {
                "message": "multipart form must include either an image field or a non-empty text field",
                "type": "invalid_request_error",
                "param": None,
                "code": None,
            }
        }), 400

    try:
        user_text = choose_user_text(request_controls)
        forced_sub_agent = get_forced_sub_agent(text_fields)
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
    try:
        is_blocking = True

        request_route = select_request_route(user_text, image_file is not None and bool(image_file.filename), forced_sub_agent)
        route = request_route["route"]
        if route in {DIRECT_CHAT_ROUTE, UNSUPPORTED_TASK_ROUTE}:
            final_message, final_usage = make_direct_chat_answer(user_text, request_route.get("reason", ""))
            response_payload = {
                "object": "multi_agent.chat",
                "message": final_message,
            }
            if debug_response:
                prompt_tokens = request_route["_usage"]["prompt_tokens"] + final_usage.get("prompt_tokens", 0)
                completion_tokens = request_route["_usage"]["completion_tokens"] + final_usage.get("completion_tokens", 0)
                response_payload.update({
                    "main_agent_selection": {
                        "route": route,
                        "reason": request_route.get("reason", ""),
                    },
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": prompt_tokens + completion_tokens,
                    },
                })
            return Response(json.dumps(response_payload, ensure_ascii=False), content_type="application/json")

        cluster_resources = json.loads(call_mcp_tool("get_cluster_resources"))
        sub_agent_name = request_route["sub_agent"]
        selection, sub_agent_response, start_result, sub_agent_url, startup_failures = invoke_sub_agent_with_failover(
            user_text,
            cluster_resources,
            sub_agent_name,
            image_file,
        )
        final_message, final_usage = make_final_answer(user_text, sub_agent_name, sub_agent_response)

        sub_agent_select_usage = request_route["_usage"]
        main_usage = selection["_usage"]
        sub_usage = sub_agent_response.get("usage", {})
        prompt_tokens = (
            sub_agent_select_usage["prompt_tokens"]
            + main_usage["prompt_tokens"]
            + sub_usage.get("prompt_tokens", 0)
            + final_usage.get("prompt_tokens", 0)
        )
        completion_tokens = (
            sub_agent_select_usage["completion_tokens"]
            + main_usage["completion_tokens"]
            + sub_usage.get("completion_tokens", 0)
            + final_usage.get("completion_tokens", 0)
        )

        response_payload = {
            "object": "multi_agent.chat",
            "message": final_message,
        }
        result_image = sub_agent_response.get("result_image")
        if result_image:
            response_payload["result_image"] = result_image

        if debug_response:
            response_payload.update({
                "main_agent_selection": {
                    "sub_agent": selection["sub_agent"],
                    "sub_agent_reason": request_route.get("reason", ""),
                    "target_global_id": selection["target_global_id"],
                    "reason": selection.get("reason", ""),
                    "target_node": selection["target_node"],
                },
                "sub_agent_startup": start_result,
                "sub_agent_startup_failures": startup_failures,
                "sub_agent_url": sub_agent_url,
                "sub_agent_result": sub_agent_response,
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
            })

        return Response(json.dumps(response_payload, ensure_ascii=False), content_type="application/json")
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
    parser.add_argument("--image_agent_url", default=DEFAULT_IMAGE_AGENT_URL)
    parser.add_argument("--text_agent_url", default=DEFAULT_TEXT_AGENT_URL)
    parser.add_argument("--segmentation_agent_url", default=DEFAULT_SEGMENTATION_AGENT_URL)
    args = parser.parse_args()

    LLM_API_URL = args.llm_api_url
    LLM_MODEL_NAME = args.llm_model_name
    IMAGE_AGENT_URL = args.image_agent_url
    TEXT_AGENT_URL = args.text_agent_url
    SEGMENTATION_AGENT_URL = args.segmentation_agent_url

    app.run(host=args.host, port=args.port, threaded=True, debug=False)
