import base64
import json
import mimetypes
import os
import urllib.error
import urllib.parse
import urllib.request
import uuid
from typing import Any


TASK_TYPE_TO_TASK_ID = {
    "YoloV5": "YoloV5",
    "MobileNet": "MobileNet",
    "ResNet50": "ResNet50",
    "Bert": "Bert",
    "deeplabv3": "deeplabv3",
}


def _gateway_host() -> str:
    return os.environ.get("GATEWAY_HOST", "127.0.0.1")


def _gateway_port() -> str:
    return os.environ.get("GATEWAY_PORT", "6666")


def _gateway_url(path: str) -> str:
    return f"http://{_gateway_host()}:{_gateway_port()}{path}"


def encode_multipart_formdata_from_bytes(file_field_name: str, filename: str, file_bytes: bytes) -> tuple[bytes, str]:
    boundary = f"----edge-cluster-{uuid.uuid4().hex}"
    content_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"

    body = bytearray()
    body.extend(f"--{boundary}\r\n".encode("utf-8"))
    body.extend(
        (
            f'Content-Disposition: form-data; name="{file_field_name}"; '
            f'filename="{filename}"\r\n'
        ).encode("utf-8")
    )
    body.extend(f"Content-Type: {content_type}\r\n\r\n".encode("utf-8"))
    body.extend(file_bytes)
    body.extend(f"\r\n--{boundary}--\r\n".encode("utf-8"))
    return bytes(body), f"multipart/form-data; boundary={boundary}"


def encode_multipart_formdata(file_field_name: str, file_path: str) -> tuple[bytes, str]:
    filename = os.path.basename(file_path)
    with open(file_path, "rb") as file_obj:
        file_bytes = file_obj.read()
    return encode_multipart_formdata_from_bytes(file_field_name, filename, file_bytes)


def run_vision_task_on_node(arguments: dict[str, Any]) -> dict[str, Any]:
    task_type = arguments.get("task_type")
    target_global_id = arguments.get("target_global_id")
    image_path = arguments.get("image_path")
    image_b64 = arguments.get("image_b64")
    image_name = arguments.get("image_name", "upload.bin")
    real_url = arguments.get("real_url", "predict")
    file_field_name = arguments.get("file_field_name", "image")

    if task_type not in TASK_TYPE_TO_TASK_ID:
        raise RuntimeError(f"unsupported task_type: {task_type}")
    if not target_global_id:
        raise RuntimeError("target_global_id is required")
    if image_b64:
        image_path = None
    if not image_path and not image_b64:
        raise RuntimeError("image_path or image_b64 is required")
    if image_path and not os.path.exists(image_path):
        raise RuntimeError(f"image_path does not exist: {image_path}")

    path = os.environ.get("GATEWAY_QUEST_ON_NODE_PATH", "/quest_on_node")
    query = urllib.parse.urlencode(
        {
            "taskid": TASK_TYPE_TO_TASK_ID[task_type],
            "target_global_id": target_global_id,
            "real_url": real_url,
        }
    )
    url = f"{_gateway_url(path)}?{query}"

    if image_b64:
        try:
            image_bytes = base64.b64decode(image_b64)
        except Exception as exc:
            raise RuntimeError(f"invalid image_b64: {exc}") from exc
        body, content_type = encode_multipart_formdata_from_bytes(file_field_name, image_name, image_bytes)
    else:
        body, content_type = encode_multipart_formdata(file_field_name, image_path)
    request = urllib.request.Request(url, data=body, method="POST")
    request.add_header("Content-Type", content_type)
    request.add_header("Content-Length", str(len(body)))

    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            response_body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"gateway returned HTTP {exc.code} for {url}: {error_body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"failed to reach gateway at {url}: {exc.reason}") from exc

    try:
        return json.loads(response_body)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"gateway returned invalid JSON: {exc}") from exc
