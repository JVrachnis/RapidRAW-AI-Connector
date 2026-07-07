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
