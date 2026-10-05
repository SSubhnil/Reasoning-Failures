from __future__ import annotations

import socket
import subprocess
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

import httpx


@dataclass(frozen=True, slots=True)
class ServerSettings:
    model: str
    server_python: str
    port: int
    max_model_len: int
    max_num_seqs: int
    gpu_memory_utilization: float
    tensor_parallel_size: int
    startup_timeout_s: float


def server_command(settings: ServerSettings) -> list[str]:
    return [
        settings.server_python, "-m", "vllm.entrypoints.openai.api_server",
        "--model", settings.model,
        "--served-model-name", settings.model,
        "--port", str(settings.port),
        "--dtype", "bfloat16",
        "--max-model-len", str(settings.max_model_len),
        "--max-num-seqs", str(settings.max_num_seqs),
        "--gpu-memory-utilization", str(settings.gpu_memory_utilization),
        "--tensor-parallel-size", str(settings.tensor_parallel_size),
        "--enable-prefix-caching",
    ]


def raise_if_stopped(process: subprocess.Popen, deadline: float) -> None:
    if process.poll() is not None:
        raise RuntimeError(f"vLLM server exited with code {process.returncode}")
    if time.monotonic() > deadline:
        raise TimeoutError("vLLM server did not become healthy in time")


def port_is_open(port: int) -> bool:
    with socket.socket() as sock:
        return sock.connect_ex(("127.0.0.1", port)) == 0


def wait_until_healthy(settings: ServerSettings, process: subprocess.Popen) -> str:
    """Block until the server answers /health and return its base URL."""
    deadline = time.monotonic() + settings.startup_timeout_s
    base_url = f"http://127.0.0.1:{settings.port}"
    while not port_is_open(settings.port):
        raise_if_stopped(process, deadline)
        time.sleep(2.0)
    while httpx.get(f"{base_url}/health", timeout=10.0).status_code != 200:
        raise_if_stopped(process, deadline)
        time.sleep(2.0)
    return base_url


@contextmanager
def running_server(settings: ServerSettings) -> Iterator[str]:
    """Start the vLLM server, yield its base URL, and stop it on exit."""
    process = subprocess.Popen(server_command(settings))
    try:
        yield wait_until_healthy(settings, process)
    finally:
        process.terminate()
        process.wait(timeout=120)
