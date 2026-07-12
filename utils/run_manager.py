from __future__ import annotations

import os
from dataclasses import asdict, is_dataclass
from datetime import datetime
from typing import Any, Dict, Optional

from utils.io import ensure_dir, save_json


def now_str() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def sanitize_tag(text: str) -> str:
    text = str(text).strip().replace(" ", "_")
    bad_chars = ['/', '\\', ':', '*', '?', '"', '<', '>', '|']
    for ch in bad_chars:
        text = text.replace(ch, "-")
    return text


def build_run_name(prefix: str, method: str, tag: Optional[str] = None) -> str:
    parts = [now_str(), sanitize_tag(prefix), sanitize_tag(method)]
    if tag:
        parts.append(sanitize_tag(tag))
    return "_".join(parts)


def build_run_dirs(root_dir: str, run_name: str) -> Dict[str, str]:
    run_dir = os.path.join(root_dir, run_name)
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    plot_dir = os.path.join(run_dir, "plots")
    log_dir = os.path.join(run_dir, "logs")

    ensure_dir(run_dir)
    ensure_dir(ckpt_dir)
    ensure_dir(plot_dir)
    ensure_dir(log_dir)

    return {
        "run_dir": run_dir,
        "ckpt_dir": ckpt_dir,
        "plot_dir": plot_dir,
        "log_dir": log_dir,
    }


def _to_serializable(obj: Any) -> Any:
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(x) for x in obj]
    return obj


def save_manifest(
    run_dir: str,
    run_type: str,
    run_name: str,
    note: str = "",
    **kwargs: Any,
) -> None:
    manifest = {
        "run_type": run_type,
        "run_name": run_name,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "note": note,
    }
    for k, v in kwargs.items():
        manifest[k] = _to_serializable(v)

    save_json(manifest, os.path.join(run_dir, "manifest.json"))