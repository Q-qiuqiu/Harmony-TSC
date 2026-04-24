import ctypes
import sys
import os
import subprocess
import resource
import threading
import time
import argparse
import json
from flask import Flask, request, jsonify, Response, stream_with_context
import tiktoken

app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TOOLS_DIR = os.path.normpath(os.path.join(BASE_DIR, "..", "tools"))
DEFAULT_MCP_SERVER_PATH = os.path.join(TOOLS_DIR, "cluster_resources_mcp.py")
DEFAULT_MCP_TOOL_NAME = "get_cluster_resources"

# Set the dynamic library path
rkllm_lib = ctypes.CDLL('lib/librkllmrt.so')

# Define the structures from the library
RKLLM_Handle_t = ctypes.c_void_p
userdata = ctypes.c_void_p(None)

LLMCallState = ctypes.c_int
LLMCallState.RKLLM_RUN_NORMAL  = 0
LLMCallState.RKLLM_RUN_WAITING  = 1
LLMCallState.RKLLM_RUN_FINISH  = 2
LLMCallState.RKLLM_RUN_ERROR   = 3

RKLLMInputMode = ctypes.c_int
RKLLMInputMode.RKLLM_INPUT_PROMPT      = 0
RKLLMInputMode.RKLLM_INPUT_TOKEN       = 1
RKLLMInputMode.RKLLM_INPUT_EMBED       = 2
RKLLMInputMode.RKLLM_INPUT_MULTIMODAL  = 3

RKLLMInferMode = ctypes.c_int
RKLLMInferMode.RKLLM_INFER_GENERATE = 0
RKLLMInferMode.RKLLM_INFER_GET_LAST_HIDDEN_LAYER = 1
RKLLMInferMode.RKLLM_INFER_GET_LOGITS = 2
class RKLLMExtendParam(ctypes.Structure):
    _fields_ = [
        ("base_domain_id", ctypes.c_int32),
        ("embed_flash", ctypes.c_int8),
        ("enabled_cpus_num", ctypes.c_int8),
        ("enabled_cpus_mask", ctypes.c_uint32),
        ("reserved", ctypes.c_uint8 * 106)
    ]

class RKLLMParam(ctypes.Structure):
    _fields_ = [
        ("model_path", ctypes.c_char_p),
        ("max_context_len", ctypes.c_int32),
        ("max_new_tokens", ctypes.c_int32),
        ("top_k", ctypes.c_int32),
        ("n_keep", ctypes.c_int32),
        ("top_p", ctypes.c_float),
        ("temperature", ctypes.c_float),
        ("repeat_penalty", ctypes.c_float),
        ("frequency_penalty", ctypes.c_float),
        ("presence_penalty", ctypes.c_float),
        ("mirostat", ctypes.c_int32),
        ("mirostat_tau", ctypes.c_float),
        ("mirostat_eta", ctypes.c_float),
        ("skip_special_token", ctypes.c_bool),
        ("is_async", ctypes.c_bool),
        ("img_start", ctypes.c_char_p),
        ("img_end", ctypes.c_char_p),
        ("img_content", ctypes.c_char_p),
        ("extend_param", RKLLMExtendParam),
    ]

class RKLLMLoraAdapter(ctypes.Structure):
    _fields_ = [
        ("lora_adapter_path", ctypes.c_char_p),
        ("lora_adapter_name", ctypes.c_char_p),
        ("scale", ctypes.c_float)
    ]

class RKLLMEmbedInput(ctypes.Structure):
    _fields_ = [
        ("embed", ctypes.POINTER(ctypes.c_float)),
        ("n_tokens", ctypes.c_size_t)
    ]

class RKLLMTokenInput(ctypes.Structure):
    _fields_ = [
        ("input_ids", ctypes.POINTER(ctypes.c_int32)),
        ("n_tokens", ctypes.c_size_t)
    ]

class RKLLMMultiModelInput(ctypes.Structure):
    _fields_ = [
        ("prompt", ctypes.c_char_p),
        ("image_embed", ctypes.POINTER(ctypes.c_float)),
        ("n_image_tokens", ctypes.c_size_t),
        ("n_image", ctypes.c_size_t),
        ("image_width", ctypes.c_size_t),
        ("image_height", ctypes.c_size_t)
    ]

class RKLLMInputUnion(ctypes.Union):
    _fields_ = [
        ("prompt_input", ctypes.c_char_p),
        ("embed_input", RKLLMEmbedInput),
        ("token_input", RKLLMTokenInput),
        ("multimodal_input", RKLLMMultiModelInput)
    ]

class RKLLMInput(ctypes.Structure):
    _fields_ = [
        ("input_mode", ctypes.c_int),
        ("input_data", RKLLMInputUnion)
    ]

class RKLLMLoraParam(ctypes.Structure):
    _fields_ = [
        ("lora_adapter_name", ctypes.c_char_p)
    ]

class RKLLMPromptCacheParam(ctypes.Structure):
    _fields_ = [
        ("save_prompt_cache", ctypes.c_int),
        ("prompt_cache_path", ctypes.c_char_p)
    ]

class RKLLMInferParam(ctypes.Structure):
    _fields_ = [
        ("mode", RKLLMInferMode),
        ("lora_params", ctypes.POINTER(RKLLMLoraParam)),
        ("prompt_cache_params", ctypes.POINTER(RKLLMPromptCacheParam)),
        ("keep_history", ctypes.c_int)
    ]

class RKLLMResultLastHiddenLayer(ctypes.Structure):
    _fields_ = [
        ("hidden_states", ctypes.POINTER(ctypes.c_float)),
        ("embd_size", ctypes.c_int),
        ("num_tokens", ctypes.c_int)
    ]

class RKLLMResultLogits(ctypes.Structure):
    _fields_ = [
        ("logits", ctypes.POINTER(ctypes.c_float)),
        ("vocab_size", ctypes.c_int),
        ("num_tokens", ctypes.c_int)
    ]

class RKLLMResult(ctypes.Structure):
    _fields_ = [
        ("text", ctypes.c_char_p),
        ("token_id", ctypes.c_int),
        ("last_hidden_layer", RKLLMResultLastHiddenLayer),
        ("logits", RKLLMResultLogits)
    ]


# Create a lock to control multi-user access to the server.
lock = threading.Lock()

# Create a global variable to indicate whether the server is currently in a blocked state.
is_blocking = False

# Define global variables to store the callback function output for displaying in the Gradio interface
global_text = []
global_state = -1
split_byte_data = bytes(b"") # Used to store the segmented byte data

# Define the callback function
def callback_impl(result, userdata, state):
    global global_text, global_state, split_byte_data
    if state == LLMCallState.RKLLM_RUN_FINISH:
        global_state = state
        print("\n")
        sys.stdout.flush()
    elif state == LLMCallState.RKLLM_RUN_ERROR:
        global_state = state
        print("run error")
        sys.stdout.flush()
    elif state == LLMCallState.RKLLM_RUN_NORMAL:
        global_state = state
        try:
            global_text.append((split_byte_data + result.contents.text).decode('utf-8'))
            split_byte_data = bytes(b"")
        except:
            split_byte_data += result.contents.text
        sys.stdout.flush()

# Connect the callback function between the Python side and the C++ side
callback_type = ctypes.CFUNCTYPE(None, ctypes.POINTER(RKLLMResult), ctypes.c_void_p, ctypes.c_int)
callback = callback_type(callback_impl)

# Define the RKLLM class, which includes initialization, inference, and release operations for the RKLLM model in the dynamic library
class RKLLM(object):
    def __init__(self, model_path, lora_model_path = None, prompt_cache_path = None):
        rkllm_param = RKLLMParam()
        rkllm_param.model_path = bytes(model_path, 'utf-8')

        rkllm_param.max_context_len = 4096
        rkllm_param.max_new_tokens = -1
        rkllm_param.skip_special_token = True
        rkllm_param.n_keep = -1
        rkllm_param.top_k = 1
        rkllm_param.top_p = 0.9
        rkllm_param.temperature = 0.8
        rkllm_param.repeat_penalty = 1.1
        rkllm_param.frequency_penalty = 0.0
        rkllm_param.presence_penalty = 0.0

        rkllm_param.mirostat = 0
        rkllm_param.mirostat_tau = 5.0
        rkllm_param.mirostat_eta = 0.1

        rkllm_param.is_async = False

        rkllm_param.img_start = "".encode('utf-8')
        rkllm_param.img_end = "".encode('utf-8')
        rkllm_param.img_content = "".encode('utf-8')

        rkllm_param.extend_param.base_domain_id = 0
        rkllm_param.extend_param.enabled_cpus_num = 4
        rkllm_param.extend_param.enabled_cpus_mask = (1 << 4)|(1 << 5)|(1 << 6)|(1 << 7)

        self.handle = RKLLM_Handle_t()

        self.rkllm_init = rkllm_lib.rkllm_init
        self.rkllm_init.argtypes = [ctypes.POINTER(RKLLM_Handle_t), ctypes.POINTER(RKLLMParam), callback_type]
        self.rkllm_init.restype = ctypes.c_int
        self.rkllm_init(ctypes.byref(self.handle), ctypes.byref(rkllm_param), callback)

        self.rkllm_run = rkllm_lib.rkllm_run
        self.rkllm_run.argtypes = [RKLLM_Handle_t, ctypes.POINTER(RKLLMInput), ctypes.POINTER(RKLLMInferParam), ctypes.c_void_p]
        self.rkllm_run.restype = ctypes.c_int
        
        self.set_chat_template = rkllm_lib.rkllm_set_chat_template
        self.set_chat_template.argtypes = [RKLLM_Handle_t, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p]
        self.set_chat_template.restype = ctypes.c_int
        
        system_prompt = "<|im_start|>system You are a helpful assistant. <|im_end|>"
        prompt_prefix = "<|im_start|>user"
        prompt_postfix = "<|im_end|><|im_start|>assistant"
        # self.set_chat_template(self.handle, ctypes.c_char_p(system_prompt.encode('utf-8')), ctypes.c_char_p(prompt_prefix.encode('utf-8')), ctypes.c_char_p(prompt_postfix.encode('utf-8')))

        self.rkllm_destroy = rkllm_lib.rkllm_destroy
        self.rkllm_destroy.argtypes = [RKLLM_Handle_t]
        self.rkllm_destroy.restype = ctypes.c_int

        rkllm_lora_params = None
        if lora_model_path:
            lora_adapter_name = "test"
            lora_adapter = RKLLMLoraAdapter()
            ctypes.memset(ctypes.byref(lora_adapter), 0, ctypes.sizeof(RKLLMLoraAdapter))
            lora_adapter.lora_adapter_path = ctypes.c_char_p((lora_model_path).encode('utf-8'))
            lora_adapter.lora_adapter_name = ctypes.c_char_p((lora_adapter_name).encode('utf-8'))
            lora_adapter.scale = 1.0

            rkllm_load_lora = rkllm_lib.rkllm_load_lora
            rkllm_load_lora.argtypes = [RKLLM_Handle_t, ctypes.POINTER(RKLLMLoraAdapter)]
            rkllm_load_lora.restype = ctypes.c_int
            rkllm_load_lora(self.handle, ctypes.byref(lora_adapter))
            rkllm_lora_params = RKLLMLoraParam()
            rkllm_lora_params.lora_adapter_name = ctypes.c_char_p((lora_adapter_name).encode('utf-8'))
        
        self.rkllm_infer_params = RKLLMInferParam()
        ctypes.memset(ctypes.byref(self.rkllm_infer_params), 0, ctypes.sizeof(RKLLMInferParam))
        self.rkllm_infer_params.mode = RKLLMInferMode.RKLLM_INFER_GENERATE
        self.rkllm_infer_params.lora_params = ctypes.pointer(rkllm_lora_params) if rkllm_lora_params else None
        self.rkllm_infer_params.keep_history = 0

        self.prompt_cache_path = None
        if prompt_cache_path:
            self.prompt_cache_path = prompt_cache_path

            rkllm_load_prompt_cache = rkllm_lib.rkllm_load_prompt_cache
            rkllm_load_prompt_cache.argtypes = [RKLLM_Handle_t, ctypes.c_char_p]
            rkllm_load_prompt_cache.restype = ctypes.c_int
            rkllm_load_prompt_cache(self.handle, ctypes.c_char_p((prompt_cache_path).encode('utf-8')))

    def run(self, prompt):
        rkllm_input = RKLLMInput()
        rkllm_input.input_mode = RKLLMInputMode.RKLLM_INPUT_PROMPT
        rkllm_input.input_data.prompt_input = ctypes.c_char_p(prompt.encode('utf-8'))
        self.rkllm_run(self.handle, ctypes.byref(rkllm_input), ctypes.byref(self.rkllm_infer_params), None)
        return

    def release(self):
        self.rkllm_destroy(self.handle)

def num_tokens_from_string(string: str, encoding_name: str = "cl100k_base") -> int:
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def call_mcp_tool(server_path, tool_name, arguments=None, timeout=10):
    if arguments is None:
        arguments = {}

    process = subprocess.Popen(
        [sys.executable, server_path],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    requests = [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
        {"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {"name": tool_name, "arguments": arguments}},
    ]
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
        if not line:
            continue
        responses.append(json.loads(line))

    tool_response = None
    for item in responses:
        if item.get("id") == 2:
            tool_response = item
            break

    if tool_response is None:
        raise RuntimeError("MCP tool did not return a tools/call response")

    result = tool_response.get("result", {})
    if result.get("isError"):
        content = result.get("content", [])
        if content and isinstance(content[0], dict):
            raise RuntimeError(content[0].get("text", "MCP tool returned an error"))
        raise RuntimeError("MCP tool returned an error")

    content = result.get("content", [])
    if not content or not isinstance(content[0], dict):
        raise RuntimeError("MCP tool returned empty content")

    return content[0].get("text", "")

def list_tools():
    return [
        {
            "name": DEFAULT_MCP_TOOL_NAME,
            "description": "Fetch the latest cluster resource snapshot collected by the gateway.",
            "input_schema": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        }
    ]

def execute_tool(tool_name, arguments=None):
    if tool_name != DEFAULT_MCP_TOOL_NAME:
        raise RuntimeError(f"Unknown tool: {tool_name}")
    if not os.path.exists(DEFAULT_MCP_SERVER_PATH):
        raise RuntimeError(f"MCP server not found: {DEFAULT_MCP_SERVER_PATH}")
    return call_mcp_tool(DEFAULT_MCP_SERVER_PATH, tool_name, arguments=arguments or {})

def build_agent_system_prompt():
    tool_list_json = json.dumps(list_tools(), ensure_ascii=False)
    return (
        "You are a resource-aware cluster scheduling agent.\n"
        "Your job is to decide whether a user task needs runtime cluster resource data before answering.\n"
        "You have access to tools. First inspect the user request and decide whether calling a tool is necessary.\n"
        "Available tools are provided by the server through the list_tools function result below.\n"
        "list_tools result:\n"
        f"{tool_list_json}\n"
        "Rules:\n"
        "1. If you need tool data, respond with JSON only and no markdown:\n"
        "{\"action\":\"tool_call\",\"tool_name\":\"<tool name>\",\"arguments\":{}}\n"
        "2. If you can answer directly, respond with JSON only and no markdown:\n"
        "{\"action\":\"final\",\"content\":\"<your answer>\"}\n"
        "3. After a tool result is returned to you, use it and then respond with a final JSON object.\n"
        "4. Never invent tool results.\n"
        "5. Never output anything except a single JSON object."
    )

def normalize_messages(messages, enable_mcp_tools=False):
    normalized = list(messages)
    if enable_mcp_tools:
        return [{"role": "system", "content": build_agent_system_prompt()}] + normalized
    return normalized

def build_prompt_from_messages(messages):
    prompt = ""
    for message in messages:
        role = message.get('role', '')
        content = message.get('content', '')
        prompt += f"{role}: {content}\n"
    return prompt.strip()

def run_model_once(prompt):
    global global_text, global_state
    print("\n" + "="*50)
    print(f"用户请求: {prompt}")
    print("-" * 20)
    global_text = []
    global_state = -1

    rkllm_output = ""
    prompt_tokens = num_tokens_from_string(prompt)
    completion_tokens = 0
    infer_start_time = time.time()

    model_thread = threading.Thread(target=rkllm_model.run, args=(prompt,))
    model_thread.start()

    while True:
        while len(global_text) > 0:
            new_text = global_text.pop(0)
            rkllm_output += new_text
            completion_tokens += num_tokens_from_string(new_text)

        model_thread.join(timeout=0.005)
        if not model_thread.is_alive():
            break

    model_thread.join()

    infer_end_time = time.time()
    infer_duration = infer_end_time - infer_start_time
    tokens_per_sec = completion_tokens / infer_duration if infer_duration > 0 else 0


    print(f"模型回答: {rkllm_output}")
    print(f"推理耗时: {infer_duration:.3f} 秒")
    print(f"生成 token 数: {completion_tokens}")
    print(f"生成速度: {tokens_per_sec:.2f} tokens/sec")
    print("="*50 + "\n")
    sys.stdout.flush()

    return {
        "content": rkllm_output,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
    }

def parse_agent_json(text):
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if len(lines) >= 3:
            cleaned = "\n".join(lines[1:-1]).strip()
    return json.loads(cleaned)

def run_agent_with_tools(messages, max_steps=3):
    working_messages = normalize_messages(messages, enable_mcp_tools=True)
    total_prompt_tokens = 0
    total_completion_tokens = 0

    for _ in range(max_steps):
        prompt = build_prompt_from_messages(working_messages)
        model_result = run_model_once(prompt)
        total_prompt_tokens += model_result["prompt_tokens"]
        total_completion_tokens += model_result["completion_tokens"]

        try:
            agent_action = parse_agent_json(model_result["content"])
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Model did not return valid JSON for agent control: {exc}")

        action = agent_action.get("action")
        if action == "final":
            return {
                "content": agent_action.get("content", ""),
                "prompt_tokens": total_prompt_tokens,
                "completion_tokens": total_completion_tokens,
            }

        if action == "tool_call":
            tool_name = agent_action.get("tool_name")
            tool_arguments = agent_action.get("arguments", {})
            tool_result = execute_tool(tool_name, tool_arguments)
            working_messages.append({"role": "assistant", "content": model_result["content"]})
            working_messages.append({
                "role": "tool",
                "content": f"tool_name={tool_name}\nresult={tool_result}"
            })
            continue

        raise RuntimeError(f"Unknown agent action: {action}")

    raise RuntimeError("Agent exceeded max tool-calling steps without producing a final answer")

@app.route('/v1/chat/completions', methods=['POST'])
def chat_completions():
    global global_text, global_state
    global is_blocking

    if is_blocking or global_state == 0:
        return jsonify({
            "error": {
                "message": "RKLLM Server is busy! Please try again later.",
                "type": "server_error",
                "param": None,
                "code": None
            }
        }), 503

    lock.acquire()
    try:
        is_blocking = True

        data = request.json
        if not data or 'messages' not in data:
            return jsonify({
                "error": {
                    "message": "Invalid request",
                    "type": "invalid_request_error",
                    "param": None,
                    "code": None
                }
            }), 400

        global_text = []
        global_state = -1

        messages = data['messages']
        enable_mcp_tools = data.get('enable_mcp_tools', True)
        stream = data.get('stream', False)
        model = data.get('model', 'rkllm-default')
        if enable_mcp_tools and stream:
            return jsonify({
                "error": {
                    "message": "Streaming is not supported when MCP tool mode is enabled.",
                    "type": "invalid_request_error",
                    "param": None,
                    "code": None
                }
            }), 400

        prompt = build_prompt_from_messages(normalize_messages(messages))

        def generate():
            nonlocal prompt
            print("\n" + "="*50)
            print(f"用户请求: {prompt}")
            print("-" * 20)
            rkllm_output = ""
            prompt_tokens = num_tokens_from_string(prompt)
            completion_tokens = 0
            # ========== 推理计时开始 ==========
            infer_start_time = time.time()
            model_thread = threading.Thread(target=rkllm_model.run, args=(prompt,))
            model_thread.start()

            model_thread_finished = False
            while not model_thread_finished:
                while len(global_text) > 0:
                    new_text = global_text.pop(0)
                    rkllm_output += new_text
                    completion_tokens += num_tokens_from_string(new_text)

                    response = {
                        "id": f"chatcmpl-{time.time()}",
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": model,
                        "choices": [{
                            "index": 0,
                            "delta": {
                                "content": new_text
                            },
                            "finish_reason": None
                        }]
                    }

                    if stream:
                        yield f"data: {json.dumps(response)}\n\n"
                    time.sleep(0.005)

                model_thread.join(timeout=0.005)
                model_thread_finished = not model_thread.is_alive()

                model_thread.join() 
                # ========== 推理计时结束 ==========
                infer_end_time = time.time()
                infer_duration = infer_end_time - infer_start_time
                # 计算生成速度
                tokens_per_sec = completion_tokens / infer_duration if infer_duration > 0 else 0


                print(f"模型回答: {rkllm_output}")
                print(f"推理耗时: {infer_duration:.3f} 秒")
                print(f"生成 token 数: {completion_tokens}")
                print(f"生成速度: {tokens_per_sec:.2f} tokens/sec")
                print("="*50 + "\n")
                sys.stdout.flush() # 确保立即输出到控制台

            if stream:
                final_response = {
                    "id": f"chatcmpl-{time.time()}",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": model,
                    "choices": [{
                        "index": 0,
                        "delta": {},
                        "finish_reason": "stop"
                    }]
                }
                yield f"data: {json.dumps(final_response)}\n\n"
                yield "data: [DONE]\n\n"
            else:
                response = {
                    "id": f"chatcmpl-{time.time()}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": model,
                    "choices": [{
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": rkllm_output
                        },
                        "finish_reason": "stop"
                    }],
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": prompt_tokens + completion_tokens
                    }
                }
                yield json.dumps(response)

        if enable_mcp_tools:
            try:
                agent_result = run_agent_with_tools(messages)
            except Exception as exc:
                return jsonify({
                    "error": {
                        "message": f"Failed to run MCP tool agent loop: {str(exc)}",
                        "type": "server_error",
                        "param": None,
                        "code": None
                    }
                }), 500

            response = {
                "id": f"chatcmpl-{time.time()}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model,
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": agent_result["content"]
                    },
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": agent_result["prompt_tokens"],
                    "completion_tokens": agent_result["completion_tokens"],
                    "total_tokens": agent_result["prompt_tokens"] + agent_result["completion_tokens"]
                }
            }
            return Response(json.dumps(response), content_type='application/json')

        if stream:
            return Response(stream_with_context(generate()), content_type='text/event-stream')
        else:
            return Response(next(generate()), content_type='application/json')

    finally:
        lock.release()
        is_blocking = False

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--rkllm_model_path', type=str, required=True, help='Absolute path of the converted RKLLM model on the Linux board;')
    parser.add_argument('--target_platform', type=str, required=True, help='Target platform: e.g., rk3588/rk3576;')
    parser.add_argument('--lora_model_path', type=str, help='Absolute path of the lora_model on the Linux board;')
    parser.add_argument('--prompt_cache_path', type=str, help='Absolute path of the prompt_cache file on the Linux board;')
    args = parser.parse_args()

    if not os.path.exists(args.rkllm_model_path):
        print("Error: Please provide the correct rkllm model path, and ensure it is the absolute path on the board.")
        sys.stdout.flush()
        exit()

    if not (args.target_platform in ["rk3588", "rk3576"]):
        print("Error: Please specify the correct target platform: rk3588/rk3576.")
        sys.stdout.flush()
        exit()

    if args.lora_model_path:
        if not os.path.exists(args.lora_model_path):
            print("Error: Please provide the correct lora_model path, and advise it is the absolute path on the board.")
            sys.stdout.flush()
            exit()

    if args.prompt_cache_path:
        if not os.path.exists(args.prompt_cache_path):
            print("Error: Please provide the correct prompt_cache_file path, and advise it is the absolute path on the board.")
            sys.stdout.flush()
            exit()

    # Fix frequency
    command = "sudo bash fix_freq_{}.sh".format(args.target_platform)
    subprocess.run(command, shell=True)

    # Set resource limit
    resource.setrlimit(resource.RLIMIT_NOFILE, (102400, 102400))

    load_start_time = time.time()
    # Initialize RKLLM model
    print("=========init....===========")
    sys.stdout.flush()
    model_path = args.rkllm_model_path
    rkllm_model = RKLLM(model_path, args.lora_model_path, args.prompt_cache_path)
    load_end_time = time.time()
    load_duration = load_end_time - load_start_time
    print("RKLLM Model has been initialized successfully！")
    print(f"模型加载耗时: {load_duration:.3f} 秒 ({load_duration*1000:.1f} ms)")
    print("==============================")
    sys.stdout.flush()

    # Start the Flask application.
    app.run(host='0.0.0.0', port=8081, threaded=True, debug=False)

    print("====================")
    print("RKLLM model inference completed, releasing RKLLM model resources...")
    rkllm_model.release()
    print("====================")
