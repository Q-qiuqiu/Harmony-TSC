import json
import os
from typing import Any


TASK_TYPE_TO_TASK_ID = {
    "YoloV5": "YoloV5",
    "MobileNet": "MobileNet",
    "ResNet50": "ResNet50",
    "Bert": "Bert",
    "deeplabv3": "deeplabv3",
}


def _default_static_info_path() -> str:
    return os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "config_files", "static_info.json"))


def fetch_task_catalog(available_device_types: list[str] | None = None) -> dict[str, Any]:
    static_info_path = os.environ.get("STATIC_INFO_PATH", _default_static_info_path())
    try:
        with open(static_info_path, "r", encoding="utf-8") as file_obj:
            payload = json.load(file_obj)
    except OSError as exc:
        raise RuntimeError(f"failed to read static info from {static_info_path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"invalid JSON in static info {static_info_path}: {exc}") from exc

    allowed_device_types = set(available_device_types or [])
    use_filter = bool(allowed_device_types)
    result: list[dict[str, Any]] = []

    for task_name, device_entries in payload.items():
        if task_name not in TASK_TYPE_TO_TASK_ID:
            continue

        for device_type, config in device_entries.items():
            if use_filter and device_type not in allowed_device_types:
                continue
            task_overhead = config.get("taskOverhead", {})
            result.append(
                {
                    "device_type": device_type,
                    "model_name": task_name,
                    "task_type": task_name,
                    "overhead": {
                    "proc_time": task_overhead.get("proc_time"),
                    "mem_usage": task_overhead.get("mem_usage"),
                    "cpu_usage": task_overhead.get("cpu_usage"),
                    "xpu_usage": task_overhead.get("xpu_usage"),
                    },
                }
            )

    return {"status": "success", "result": result}
