"""Run ON INFERNO with ComfyUI up:  COMFY_PORT=8188 .venv/bin/pytest -m integration -v
Fixture image: any JPEG with a clear single subject at tests/integration/fixtures/subject.jpg
(not committed to git)."""
import base64
import io
import time
import pytest
from pathlib import Path
from PIL import Image
from fastapi.testclient import TestClient
from engine import Settings
from gateway.app import create_app

FIXTURE = Path(__file__).parent / "fixtures" / "subject.jpg"
pytestmark = pytest.mark.integration

@pytest.fixture(scope="module")
def client(tmp_path_factory):
    settings = Settings(CACHE_DIR=tmp_path_factory.mktemp("cache"),
                        GATEWAY_JOB_TIMEOUT_S=600)
    with TestClient(create_app(settings)) as c:
        yield c

def run_mask(client, params):
    sid = client.post("/sources", files={"file": ("subject.jpg", FIXTURE.read_bytes(),
                       "image/jpeg")}).json()["source_id"]
    job = client.post("/jobs/mask", json={"source_id": sid, "params": params}).json()
    for _ in range(600):
        j = client.get(f"/jobs/{job['job_id']}").json()
        if j["status"] in ("done", "error"):
            break
        time.sleep(1)
    assert j["status"] == "done", j.get("error")
    return j["result"]

@pytest.mark.skipif(not FIXTURE.exists(), reason="fixture image missing")
def test_prompt_mode(client):
    r = run_mask(client, {"mode": "prompt", "query": "the main subject."})
    img = Image.open(io.BytesIO(base64.b64decode(r["mask_png_b64"])))
    src = Image.open(FIXTURE)
    assert img.size == src.size
    hist = img.convert("L").histogram()
    assert sum(hist[16:]) > 100          # mask is not empty
    assert isinstance(r["labels"], list)

@pytest.mark.skipif(not FIXTURE.exists(), reason="fixture image missing")
def test_preset_subject(client):
    r = run_mask(client, {"mode": "preset", "preset": "subject"})
    assert r["alignment"] == "exact"

@pytest.mark.skipif(not FIXTURE.exists(), reason="fixture image missing")
def test_points_mode(client):
    src = Image.open(FIXTURE)
    cx, cy = src.size[0] // 2, src.size[1] // 2
    r = run_mask(client, {"mode": "points", "points": [[cx, cy, 1]]})
    assert r["width"] == src.size[0]
