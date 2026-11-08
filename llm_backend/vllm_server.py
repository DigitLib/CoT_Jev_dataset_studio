"""
vLLM Process & Server Manager for ClearCoT Studio.

Starts, monitors and stops a local vLLM OpenAI-compatible server from Python or Streamlit.

Models live in `vllm_model/<name>/` (config.VLLM_MODELS_DIR, override with DTSTUDIO_VLLM_MODELS_DIR): any
sub-folder with a `config.json` is offered. An optional `vllm_model/<name>/studio_vllm.json` gives launch
defaults for that model, e.g. {"trust_remote_code": true, "max_model_len": 4096, "gpu_memory_utilization": 0.6,
"quantization": null, "extra_args": []}. The server is always started with `--served-model-name <name>`, so the
API (and every dataset's provenance) shows the short name, never a local absolute path.
"""

import json
import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

from config import BASE_DIR, DATA_DIR, DEFAULT_VLLM_MODEL, VLLM_MODELS_DIR

logger = logging.getLogger(__name__)

PID_FILE = DATA_DIR / "vllm_server.pid"
LOG_FILE = DATA_DIR / "vllm_server.log"
SIDECAR_NAME = "studio_vllm.json"
DEFAULT_HF_MODEL = "Sagicc/Occam-2B-CCoT"
SIDECAR_KEYS = {"trust_remote_code", "max_model_len", "gpu_memory_utilization", "quantization", "extra_args",
                "served_model_name"}
# Older layout: model folders directly in the Studio root. Still discovered until they are moved to vllm_model/.
LEGACY_MODEL_DIRS = ("Occam-2B-CCoT", "Spark-X2.6-4B", "Spark-X2.6-4B-vllm")
LOCAL_OCCAM_DIR = BASE_DIR / "Occam-2B-CCoT"  # kept for backward compatibility


def _read_sidecar(model_dir: Path) -> Dict[str, Any]:
    p = model_dir / SIDECAR_NAME
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return {k: v for k, v in data.items() if k in SIDECAR_KEYS} if isinstance(data, dict) else {}
    except Exception as e:
        logger.warning(f"Ignoring unreadable {p}: {e}")
        return {}


def _needs_remote_code(model_dir: Path) -> bool:
    """A checkpoint that ships its own modeling code (auto_map in config.json) needs --trust-remote-code."""
    try:
        cfg = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
        return bool(cfg.get("auto_map")) or any(model_dir.glob("modeling_*.py"))
    except Exception:
        return False


def discover_local_models(models_dir: Optional[Path] = None, include_legacy: bool = True) -> List[Dict[str, Any]]:
    """
    Local vLLM models: [{name, path, settings, legacy}] for every folder with a config.json in `models_dir`
    (default config.VLLM_MODELS_DIR), plus the old root-level folders when include_legacy is True.
    `settings` merges the studio_vllm.json sidecar over detected defaults (trust_remote_code).
    """
    models_dir = Path(models_dir) if models_dir is not None else Path(VLLM_MODELS_DIR)
    found: Dict[str, Dict[str, Any]] = {}
    candidates: List[tuple] = []
    if models_dir.is_dir():
        candidates += [(d, False) for d in sorted(models_dir.iterdir())]
    if include_legacy:
        candidates += [(BASE_DIR / n, True) for n in LEGACY_MODEL_DIRS]
    for d, legacy in candidates:
        try:
            if not d.is_dir() or not (d / "config.json").exists() or d.name in found:
                continue
        except OSError:
            continue
        settings = {"trust_remote_code": _needs_remote_code(d)}
        settings.update(_read_sidecar(d))
        found[d.name] = {"name": d.name, "path": str(d), "settings": settings, "legacy": legacy}
    return list(found.values())


def resolve_model(model: Optional[str], models_dir: Optional[Path] = None) -> Dict[str, Any]:
    """
    Turn a model reference into {name, path, settings}: a discovered folder name, a filesystem path, or a
    Hugging Face id (returned unchanged, served under its own id).
    """
    local = {m["name"]: m for m in discover_local_models(models_dir)}
    if model and model in local:
        return local[model]
    if model and Path(model).expanduser().is_dir() and (Path(model).expanduser() / "config.json").exists():
        d = Path(model).expanduser().resolve()
        settings = {"trust_remote_code": _needs_remote_code(d)}
        settings.update(_read_sidecar(d))
        return {"name": d.name, "path": str(d), "settings": settings, "legacy": False}
    if model and model != DEFAULT_HF_MODEL:
        return {"name": model, "path": model, "settings": {}, "legacy": False}
    # Default: the configured local model, then any local model, then the Hugging Face id
    if DEFAULT_VLLM_MODEL in local:
        return local[DEFAULT_VLLM_MODEL]
    if model is None and local:
        return next(iter(local.values()))
    return {"name": DEFAULT_HF_MODEL, "path": DEFAULT_HF_MODEL, "settings": {}, "legacy": False}


def get_default_model() -> str:
    """Path of the default local model (config.DEFAULT_VLLM_MODEL), else the Hugging Face id."""
    return resolve_model(None)["path"]


def is_vllm_installed() -> bool:
    """Check if the vllm Python package is installed in the current environment."""
    try:
        import importlib.util
        return importlib.util.find_spec("vllm") is not None
    except Exception:
        return False


def is_vllm_running(base_url: str = "http://localhost:8000/v1") -> bool:
    """Check if the vLLM server is responding to health or model queries."""
    health_url = base_url.replace("/v1", "").rstrip("/") + "/health"
    try:
        with httpx.Client(timeout=1.5) as client:
            resp = client.get(health_url)
            if resp.status_code == 200:
                return True
            models_resp = client.get(f"{base_url.rstrip('/')}/models")
            return models_resp.status_code == 200
    except Exception:
        return False


def _is_vllm_process(pid: int) -> bool:
    """True only if `pid` is a live vLLM server process (guards against a stale PID file naming another process)."""
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    cmdline = Path(f"/proc/{pid}/cmdline")
    if cmdline.exists():
        try:
            return b"vllm" in cmdline.read_bytes()
        except OSError:
            return False
    try:  # macOS / Windows: no /proc; fall back to `ps`
        out = subprocess.run(["ps", "-p", str(pid), "-o", "command="], capture_output=True, text=True, timeout=3)
        return "vllm" in out.stdout
    except Exception:
        return False


def get_active_vllm_pid() -> Optional[int]:
    """PID of the vLLM server started by the Studio (from data/vllm_server.pid), if it is still running."""
    if PID_FILE.exists():
        try:
            pid = int(PID_FILE.read_text().strip())
        except ValueError:
            PID_FILE.unlink(missing_ok=True)
            return None
        if _is_vllm_process(pid):
            return pid
        PID_FILE.unlink(missing_ok=True)
    return None


def build_vllm_command(
    model_info: Dict[str, Any],
    port: int = 8000,
    host: str = "127.0.0.1",
    gpu_memory_utilization: float = 0.60,
    max_model_len: int = 4096,
    quantization: Optional[str] = None,
    trust_remote_code: bool = False,
    extra_args: Optional[List[str]] = None,
) -> List[str]:
    cmd = [
        sys.executable,
        "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_info["path"],
        "--served-model-name", model_info.get("settings", {}).get("served_model_name") or model_info["name"],
        "--host", host,
        "--port", str(port),
        "--gpu-memory-utilization", str(gpu_memory_utilization),
        "--max-model-len", str(max_model_len),
        "--enforce-eager",
        "--no-enable-log-requests",
    ]
    if trust_remote_code:
        cmd.append("--trust-remote-code")
    if quantization and str(quantization).lower() != "none":
        cmd.extend(["--quantization", str(quantization).lower()])
    if extra_args:
        cmd.extend(extra_args)
    return cmd


def start_vllm_server(
    model: Optional[str] = None,
    port: int = 8000,
    host: str = "127.0.0.1",
    gpu_memory_utilization: Optional[float] = None,
    max_model_len: Optional[int] = None,
    quantization: Optional[str] = None,
    trust_remote_code: Optional[bool] = None,
    extra_args: Optional[List[str]] = None,
    log_file: Optional[str] = None,
) -> int:
    """
    Launch the vLLM OpenAI API server as a background process and return its PID.

    `model` is a folder name from vllm_model/, a path, or a Hugging Face id (None: the default local model).
    Arguments left as None take the model's studio_vllm.json value, then the built-in default
    (4096 context, 60% VRAM). host defaults to 127.0.0.1; pass "0.0.0.0" only to expose the server on the network.
    """
    info = resolve_model(model)
    s = info.get("settings", {})
    if is_vllm_running(f"http://localhost:{port}/v1"):
        logger.info(f"vLLM server is already running on port {port}.")
        return get_active_vllm_pid() or 0

    cmd = build_vllm_command(
        info, port=port, host=host,
        gpu_memory_utilization=gpu_memory_utilization if gpu_memory_utilization is not None
        else float(s.get("gpu_memory_utilization", 0.60)),
        max_model_len=max_model_len if max_model_len is not None else int(s.get("max_model_len", 4096)),
        quantization=quantization if quantization is not None else s.get("quantization"),
        trust_remote_code=trust_remote_code if trust_remote_code is not None else bool(s.get("trust_remote_code")),
        extra_args=(list(s.get("extra_args") or []) + list(extra_args or [])) or None,
    )
    logger.info(f"Starting vLLM server: {' '.join(cmd)}")

    env = os.environ.copy()
    env["VLLM_USE_FLASHINFER_SAMPLER"] = "0"

    log_path = Path(log_file) if log_file else LOG_FILE
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as log_handle:  # the child keeps its own copy of the handle
        proc = subprocess.Popen(
            cmd,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            env=env,
            preexec_fn=os.setsid if hasattr(os, "setsid") else None,
        )
    PID_FILE.parent.mkdir(parents=True, exist_ok=True)
    PID_FILE.write_text(str(proc.pid))
    return proc.pid


def stop_vllm_server(port: int = 8000) -> bool:
    """Terminate the Studio-started vLLM server and its process tree to reclaim GPU VRAM."""
    pid = get_active_vllm_pid()  # verified to be a vLLM process
    stopped = False

    if pid:
        try:
            if hasattr(os, "killpg"):
                os.killpg(os.getpgid(pid), signal.SIGTERM)
            else:
                os.kill(pid, signal.SIGTERM)
            stopped = True
        except ProcessLookupError:
            pass
        except Exception as e:
            logger.warning(f"Error terminating vLLM process {pid}: {e}")

    PID_FILE.unlink(missing_ok=True)
    return stopped


def wait_for_vllm_ready(base_url: str = "http://localhost:8000/v1", timeout_seconds: int = 90) -> bool:
    """Poll the vLLM server until it passes health check or timeout expires."""
    start_time = time.time()
    while time.time() - start_time < timeout_seconds:
        if is_vllm_running(base_url):
            return True
        time.sleep(2.0)
    return False
