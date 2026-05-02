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


def _normalize_latin1_image_payload(payload: dict[str, Any]) -> dict[str, Any]:
    image_value = payload.get("image")
    if not isinstance(image_value, str):
        return payload

    normalized = dict(payload)
    try:
        image_bytes = image_value.encode("latin1")
    except UnicodeEncodeError:
        image_bytes = image_value.encode("utf-8", errors="surrogatepass")
    normalized["image_b64"] = base64.b64encode(image_bytes).decode("ascii")
    normalized.setdefault("content_type", "image/jpeg")
    normalized.pop("image", None)
    return normalized


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


def encode_multipart_formdata_with_fields(
    fields: dict[str, str],
    file_field_name: str | None = None,
    filename: str | None = None,
    file_bytes: bytes | None = None,
) -> tuple[bytes, str]:
    boundary = f"----edge-cluster-{uuid.uuid4().hex}"
    body = bytearray()

    for key, value in fields.items():
        body.extend(f"--{boundary}\r\n".encode("utf-8"))
        body.extend(f'Content-Disposition: form-data; name="{key}"\r\n\r\n'.encode("utf-8"))
        body.extend(str(value).encode("utf-8"))
        body.extend(b"\r\n")

    if file_field_name is not None and filename is not None and file_bytes is not None:
        content_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"
        body.extend(f"--{boundary}\r\n".encode("utf-8"))
        body.extend(
            (
                f'Content-Disposition: form-data; name="{file_field_name}"; '
                f'filename="{filename}"\r\n'
            ).encode("utf-8")
        )
        body.extend(f"Content-Type: {content_type}\r\n\r\n".encode("utf-8"))
        body.extend(file_bytes)
        body.extend(b"\r\n")

    body.extend(f"--{boundary}--\r\n".encode("utf-8"))
    return bytes(body), f"multipart/form-data; boundary={boundary}"


def encode_multipart_formdata(file_field_name: str, file_path: str) -> tuple[bytes, str]:
    filename = os.path.basename(file_path)
    with open(file_path, "rb") as file_obj:
        file_bytes = file_obj.read()
    return encode_multipart_formdata_from_bytes(file_field_name, filename, file_bytes)


def run_task_on_node(arguments: dict[str, Any]) -> dict[str, Any]:
    task_type = arguments.get("task_type")
    target_global_id = arguments.get("target_global_id")
    input_path = arguments.get("input_path")
    input_b64 = arguments.get("input_b64")
    input_name = arguments.get("input_name", "upload.bin")
    real_url = arguments.get("real_url", "predict")
    file_field_name = arguments.get("file_field_name", "file")
    form_fields = arguments.get("form_fields", {})

    if task_type not in TASK_TYPE_TO_TASK_ID:
        raise RuntimeError(f"unsupported task_type: {task_type}")
    if not target_global_id:
        raise RuntimeError("target_global_id is required")
    if form_fields and not isinstance(form_fields, dict):
        raise RuntimeError("form_fields must be an object when provided")
    if input_b64:
        input_path = None
    if not input_path and not input_b64 and not form_fields:
        raise RuntimeError("input_path, input_b64, or form_fields is required")
    if input_path and not os.path.exists(input_path):
        raise RuntimeError(f"input_path does not exist: {input_path}")

    path = os.environ.get("GATEWAY_QUEST_ON_NODE_PATH", "/quest_on_node")
    query = urllib.parse.urlencode(
        {
            "taskid": TASK_TYPE_TO_TASK_ID[task_type],
            "target_global_id": target_global_id,
            "real_url": real_url,
        }
    )
    url = f"{_gateway_url(path)}?{query}"

    if input_b64:
        try:
            input_bytes = base64.b64decode(input_b64)
        except Exception as exc:
            raise RuntimeError(f"invalid input_b64: {exc}") from exc
        body, content_type = encode_multipart_formdata_with_fields(
            {str(key): str(value) for key, value in form_fields.items()},
            file_field_name=file_field_name,
            filename=input_name,
            file_bytes=input_bytes,
        )
    else:
        if input_path:
            body, content_type = encode_multipart_formdata_with_fields(
                {str(key): str(value) for key, value in form_fields.items()},
                file_field_name=file_field_name,
                filename=os.path.basename(input_path),
                file_bytes=open(input_path, "rb").read(),
            )
        else:
            body, content_type = encode_multipart_formdata_with_fields(
                {str(key): str(value) for key, value in form_fields.items()}
            )
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
        payload = json.loads(response_body)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"gateway returned invalid JSON: {exc}") from exc
    if isinstance(payload, dict) and task_type == "deeplabv3":
        return _normalize_latin1_image_payload(payload)
    return payload


def run_vision_task_on_node(arguments: dict[str, Any]) -> dict[str, Any]:
    return run_task_on_node(
        {
            "task_type": arguments.get("task_type"),
            "target_global_id": arguments.get("target_global_id"),
            "input_path": arguments.get("image_path"),
            "input_b64": arguments.get("image_b64"),
            "input_name": arguments.get("image_name", "upload.bin"),
            "real_url": arguments.get("real_url", "predict"),
            "file_field_name": arguments.get("file_field_name", "image"),
        }
    )
