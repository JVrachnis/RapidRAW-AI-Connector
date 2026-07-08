"""Gateway-side client for the persistent mask worker (Phase 2).

Kept separate from capabilities/mask.py so tests can monkeypatch a single seam
(`call_worker`) to exercise the eligible/healthy, worker-error, and off paths
without a real worker or GPU. All functions here are best-effort: any failure
returns/raises in a way that mask.handle() treats as "fall back to subprocess".
"""
from __future__ import annotations

import logging
import os
import subprocess
import time
from pathlib import Path

import aiohttp

logger = logging.getLogger("MaskWorker")


def is_worker_eligible(params: dict) -> bool:
    """Eligible = the resident worker can replicate this job faithfully:
    backend sam3, non-agentic, and a mode that routes through mask_c2f.py's SAM3
    branch (prompt/box drive it via --backend sam3; paint only when the caller
    explicitly asks for sam3). Points (SAM2) and agentic stay subprocess."""
    if params.get("agentic"):
        return False
    if params.get("backend") != "sam3":
        return False
    return params.get("mode") in ("prompt", "box", "paint")


async def probe_health(url: str, timeout: float = 1.0) -> bool:
    """True iff GET {url}/health returns ok within `timeout`."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url.rstrip("/") + "/health",
                                   timeout=aiohttp.ClientTimeout(total=timeout)) as resp:
                if resp.status != 200:
                    return False
                data = await resp.json()
                return bool(data.get("ok"))
    except Exception:
        return False


def spawn_worker(settings, log_path: Path) -> "subprocess.Popen | None":
    """Spawn the worker detached in the ComfyUI venv, stdout/stderr -> log_path.
    Carries CUDA prefs from the gateway's env. Returns the Popen or None on
    failure (caller then falls back)."""
    py = settings.GATEWAY_COMFY_VENV_PY
    script = str(Path(__file__).parent / "mask_worker.py")
    host, port = _host_port(settings.GATEWAY_WORKER_URL)
    cmd = [py, script, "--host", host, "--port", str(port),
           "--tools-dir", settings.GATEWAY_TOOLS_DIR]
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        logf = open(log_path, "ab")
        env = dict(os.environ)
        proc = subprocess.Popen(
            cmd, stdout=logf, stderr=logf, stdin=subprocess.DEVNULL,
            start_new_session=True, env=env)
        logger.info("spawned mask worker pid=%s -> %s", proc.pid, log_path)
        return proc
    except Exception as e:
        logger.warning("failed to spawn mask worker: %s", e)
        return None


async def wait_healthy(url: str, deadline_s: float = 20.0, interval: float = 0.5) -> bool:
    """Poll /health until healthy or deadline. Used right after spawn."""
    end = time.perf_counter() + deadline_s
    while time.perf_counter() < end:
        if await probe_health(url, timeout=1.0):
            return True
        import asyncio
        await asyncio.sleep(interval)
    return False


async def call_worker(url: str, payload: dict, timeout: float) -> dict:
    """POST {url}/mask with `payload`; return the parsed JSON. Raises on any
    transport error or non-2xx / {ok:false} response so the caller falls back."""
    async with aiohttp.ClientSession() as session:
        async with session.post(url.rstrip("/") + "/mask", json=payload,
                                timeout=aiohttp.ClientTimeout(total=timeout)) as resp:
            data = await resp.json()
            if resp.status != 200 or not data.get("ok"):
                raise RuntimeError(f"worker error {resp.status}: {str(data)[:300]}")
            return data


def _host_port(worker_url: str) -> tuple[str, int]:
    from urllib.parse import urlparse
    u = urlparse(worker_url)
    return (u.hostname or "127.0.0.1", u.port or 5101)


# Module-level handle to the worker we spawned, so we don't spawn duplicates
# within a single gateway process lifetime.
_spawned_proc: "subprocess.Popen | None" = None


async def ensure_worker(settings, cache_dir: Path) -> bool:
    """Return True if a healthy worker is available (probing, and spawning if
    needed). Never raises; on any failure returns False and the caller uses the
    subprocess path."""
    global _spawned_proc
    url = settings.GATEWAY_WORKER_URL
    if await probe_health(url, timeout=1.0):
        return True
    # Not up: spawn (unless we already have a live child that just isn't ready).
    if _spawned_proc is None or _spawned_proc.poll() is not None:
        _spawned_proc = spawn_worker(settings, cache_dir / "worker.log")
        if _spawned_proc is None:
            return False
    return await wait_healthy(url, deadline_s=20.0)
