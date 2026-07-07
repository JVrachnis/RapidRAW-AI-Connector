import io
from PIL import Image
from tests.conftest import poll_done

def png_bytes():
    buf = io.BytesIO()
    Image.new("RGB", (8, 6), (9, 9, 9)).save(buf, "PNG")
    return buf.getvalue()

def upload(client, **extra):
    files = {"file": ("img.png", png_bytes(), "image/png")}
    return client.post("/sources", files=files, data=extra)

def test_capabilities_lists_echo(client):
    caps = client.get("/capabilities").json()
    assert [c["id"] for c in caps] == ["echo"]
    assert caps[0]["params_schema"]["required"] == ["msg"]

def test_source_upload_and_dedup(client):
    r1 = upload(client).json()
    r2 = upload(client).json()
    assert r1["source_id"] == r2["source_id"]
    assert r1["kind"] == "std" and r1["width"] == 8

def test_source_with_sidecars_reaches_handler(client):
    r = upload(client, exif='{"ISO":800}', rrdata='{"rating":5}').json()
    job = client.post("/jobs/echo", json={"source_id": r["source_id"],
                                          "params": {"msg": "hi"}}).json()
    done = poll_done(client, job["job_id"])
    assert done["result"]["rrdata"] == {"rating": 5}

def test_job_lifecycle(client):
    sid = upload(client).json()["source_id"]
    resp = client.post("/jobs/echo", json={"source_id": sid, "params": {"msg": "yo"}})
    assert resp.status_code == 202
    body = resp.json()
    assert body["status"] == "queued" and "queue_position" in body
    done = poll_done(client, body["job_id"])
    assert done["result"]["echo"] == {"msg": "yo"}

def test_params_schema_422(client):
    sid = upload(client).json()["source_id"]
    resp = client.post("/jobs/echo", json={"source_id": sid, "params": {"msg": 7}})
    assert resp.status_code == 422

def test_unknown_capability_404(client):
    sid = upload(client).json()["source_id"]
    assert client.post("/jobs/nope", json={"source_id": sid, "params": {}}).status_code == 404

def test_unknown_source_404(client):
    r = client.post("/jobs/echo", json={"source_id": "f" * 64, "params": {"msg": "x"}})
    assert r.status_code == 404

def test_queue_lists_jobs(client):
    sid = upload(client).json()["source_id"]
    job = client.post("/jobs/echo", json={"source_id": sid, "params": {"msg": "q"}}).json()
    poll_done(client, job["job_id"])
    ids = [j["job_id"] for j in client.get("/queue").json()]
    assert job["job_id"] in ids

def test_cancel_unknown_404(client):
    assert client.delete("/jobs/" + "0" * 32).status_code == 404

def test_health_reports_capabilities(client):
    h = client.get("/health").json()
    assert "echo" in h["capabilities"]
    assert "queue_depth" in h

def test_app_capability_view_survives_registry_mutation(tmp_path, clean_registry):
    """Each app must see the registry as it was at create_app() time, matching
    the handler snapshot its queue took — not the live global."""
    from fastapi.testclient import TestClient
    from engine import Settings
    from gateway import registry
    from gateway.app import create_app
    from gateway.registry import Capability

    async def echo(ctx):
        return {"echo": ctx.params}

    schema = {"type": "object", "required": ["msg"],
              "properties": {"msg": {"type": "string"}}}
    registry.register(Capability(id="echo", title="Echo",
                                 params_schema=schema, handler=echo))
    app1 = create_app(Settings(CACHE_DIR=tmp_path / "a", GATEWAY_JOB_TIMEOUT_S=5),
                      load_caps=False)

    registry.REGISTRY.clear()
    registry.register(Capability(id="other", title="Other",
                                 params_schema={"type": "object"}, handler=echo))
    app2 = create_app(Settings(CACHE_DIR=tmp_path / "b", GATEWAY_JOB_TIMEOUT_S=5),
                      load_caps=False)

    with TestClient(app1) as c1:
        assert [c["id"] for c in c1.get("/capabilities").json()] == ["echo"]
        assert c1.get("/health").json()["capabilities"] == ["echo"]
        sid = upload(c1).json()["source_id"]
        resp = c1.post("/jobs/echo", json={"source_id": sid, "params": {"msg": "hi"}})
        assert resp.status_code == 202
        assert poll_done(c1, resp.json()["job_id"])["result"]["echo"] == {"msg": "hi"}
        assert c1.post("/jobs/other", json={"source_id": sid, "params": {}}).status_code == 404

    with TestClient(app2) as c2:
        assert [c["id"] for c in c2.get("/capabilities").json()] == ["other"]
