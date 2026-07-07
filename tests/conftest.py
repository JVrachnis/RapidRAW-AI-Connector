import sys
from pathlib import Path

# Add parent directory to path so gateway module is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from fastapi.testclient import TestClient
from engine import Settings
from gateway import registry
from gateway.app import create_app
from gateway.registry import Capability

@pytest.fixture
def clean_registry():
    saved = dict(registry.REGISTRY)
    registry.REGISTRY.clear()
    yield
    registry.REGISTRY.clear()
    registry.REGISTRY.update(saved)

@pytest.fixture
def client(tmp_path, clean_registry, monkeypatch):
    async def echo(ctx):
        return {"echo": ctx.params, "kind": ctx.source.kind if ctx.source else None,
                "rrdata": ctx.source.rrdata if ctx.source else None}
    registry.register(Capability(
        id="echo", title="Echo",
        params_schema={"type": "object", "required": ["msg"],
                       "properties": {"msg": {"type": "string"}}},
        handler=echo))
    settings = Settings(CACHE_DIR=tmp_path, GATEWAY_JOB_TIMEOUT_S=5)
    app = create_app(settings, load_caps=False)
    with TestClient(app) as c:
        yield c

def poll_done(client, job_id, tries=200):
    import time
    for _ in range(tries):
        j = client.get(f"/jobs/{job_id}").json()
        if j["status"] in ("done", "error", "cancelled"):
            return j
        time.sleep(0.02)
    raise TimeoutError(j)
