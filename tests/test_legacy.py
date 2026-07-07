import base64
import io
import pytest
from unittest.mock import AsyncMock, patch
from fastapi.testclient import TestClient
from PIL import Image
from engine import Settings
from gateway.app import create_app


def png_bytes(size=(8, 6), color=(255, 255, 255)):
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, "PNG")
    return buf.getvalue()


@pytest.fixture
def client(tmp_path):
    settings = Settings(CACHE_DIR=tmp_path, GATEWAY_JOB_TIMEOUT_S=10)
    app = create_app(settings, load_caps=True)   # real capabilities incl. inpaint
    with TestClient(app) as c:
        yield c


def test_upload_source_legacy_shape(client):
    r = client.post("/upload_source",
                    files={"file": ("source.jpg", png_bytes(), "image/jpeg")},
                    data={"source_id": "abc123"})
    assert r.status_code == 200
    assert r.json()["status"] == "cached"


def test_inpaint_unknown_source_404(client):
    r = client.post("/inpaint", json={
        "source_id": "missing", "prompt": "x",
        "mask_image_base64": base64.b64encode(png_bytes()).decode()})
    assert r.status_code == 404


def test_inpaint_roundtrip_with_mocked_comfy(client):
    client.post("/upload_source",
                files={"file": ("source.jpg", png_bytes((64, 48)), "image/jpeg")},
                data={"source_id": "legacy1"})
    fake_result = png_bytes((64, 48), (10, 20, 30))
    with patch("capabilities.inpaint.ComfyClient") as MockClient:
        MockClient.return_value.execute = AsyncMock(return_value=fake_result)
        mask = png_bytes((64, 48), (255, 255, 255))
        r = client.post("/inpaint", json={
            "source_id": "legacy1", "prompt": "a cat",
            "mask_image_base64": base64.b64encode(mask).decode()})
    assert r.status_code == 200
    body = r.json()
    assert set(body.keys()) == {"x", "y", "width", "height", "color", "mask"}
    assert isinstance(body["x"], int) and isinstance(body["y"], int)
    assert body["width"] > 0 and body["height"] > 0
    # color/mask are base64-encoded PNGs
    decoded_color = base64.b64decode(body["color"])
    decoded_mask = base64.b64decode(body["mask"])
    Image.open(io.BytesIO(decoded_color)).verify()
    Image.open(io.BytesIO(decoded_mask)).verify()


def test_inpaint_error_bodies_are_single_line(client):
    client.post("/upload_source",
                files={"file": ("source.jpg", png_bytes(), "image/jpeg")},
                data={"source_id": "errsrc"})
    with patch("capabilities.inpaint.ComfyClient") as MockClient:
        MockClient.return_value.execute = AsyncMock(side_effect=ConnectionError("comfy is down"))
        r = client.post("/inpaint", json={
            "source_id": "errsrc", "prompt": "x",
            "mask_image_base64": base64.b64encode(png_bytes()).decode()})
    assert r.status_code == 502
    assert "\n" not in r.json()["detail"]
    assert r.json()["detail"].startswith("ComfyUI Unavailable:")
