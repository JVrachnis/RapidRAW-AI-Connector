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


# --- Exact-params result cache (nudge-cache Change 2) -----------------------
#
# A retry with identical capability+source_id+params should short-circuit to
# the existing completed job instead of paying pipeline work again.

def test_identical_job_resubmit_returns_cached_same_job_id(client):
    sid = upload(client).json()["source_id"]
    body = {"source_id": sid, "params": {"msg": "hi"}}
    first = client.post("/jobs/echo", json=body).json()
    poll_done(client, first["job_id"])

    second_resp = client.post("/jobs/echo", json=body)
    assert second_resp.status_code == 202
    second = second_resp.json()
    assert second["cached"] is True
    assert second["job_id"] == first["job_id"]
    assert second["status"] == "done"

    got = client.get(f"/jobs/{second['job_id']}").json()
    assert got["status"] == "done"
    assert got["result"]["echo"] == {"msg": "hi"}


def test_key_order_does_not_defeat_cache(client):
    sid = upload(client).json()["source_id"]
    first = client.post("/jobs/echo", json={"source_id": sid,
                                            "params": {"msg": "hi"}}).json()
    poll_done(client, first["job_id"])

    # Same params, but constructed with different key insertion order --
    # cache lookup must compare parsed params, not raw JSON strings.
    reordered_params = {}
    reordered_params["msg"] = "hi"
    second = client.post("/jobs/echo", json={"source_id": sid,
                                             "params": reordered_params}).json()
    assert second.get("cached") is True
    assert second["job_id"] == first["job_id"]


def test_different_params_not_cached(client):
    sid = upload(client).json()["source_id"]
    first = client.post("/jobs/echo", json={"source_id": sid,
                                            "params": {"msg": "hi"}}).json()
    poll_done(client, first["job_id"])

    second = client.post("/jobs/echo", json={"source_id": sid,
                                             "params": {"msg": "bye"}}).json()
    assert not second.get("cached")
    assert second["job_id"] != first["job_id"]


def test_no_cache_flag_skips_lookup(client):
    sid = upload(client).json()["source_id"]
    first = client.post("/jobs/echo", json={"source_id": sid,
                                            "params": {"msg": "hi"}}).json()
    poll_done(client, first["job_id"])

    second = client.post("/jobs/echo", json={"source_id": sid,
                                             "params": {"msg": "hi"},
                                             "no_cache": True}).json()
    assert not second.get("cached")
    assert second["job_id"] != first["job_id"]
    poll_done(client, second["job_id"])


def test_cancelled_job_never_returned_as_cache_hit(client):
    # echo runs essentially instantly, so racing client.delete() against the
    # worker to actually catch it 'queued' is flaky by construction; assert
    # the cache-lookup contract directly by seeding a 'cancelled' row with the
    # queue's own db, exactly as gateway/routes.py's SELECT would see it after
    # a real cancel-while-queued.
    sid = upload(client).json()["source_id"]
    db = client.app.state.queue.db
    from gateway.db import dumps, now
    db.execute(
        "INSERT INTO jobs(id,capability,source_id,params,priority,status,created,finished,error)"
        " VALUES(?,?,?,?,?,?,?,?,?)",
        ("cancelled-job-1", "echo", sid, dumps({"msg": "cancel-me"}), "interactive",
         "cancelled", now(), now(), dumps({"kind": "cancelled", "detail": "test"})))

    second = client.post("/jobs/echo", json={"source_id": sid,
                                             "params": {"msg": "cancel-me"}}).json()
    assert not second.get("cached")
    assert second["job_id"] != "cancelled-job-1"


def test_error_job_never_returned_as_cache_hit(client, monkeypatch):
    import gateway.registry as registry
    from gateway.registry import Capability

    async def boom(ctx):
        raise RuntimeError("boom")
    registry.REGISTRY["echo"] = Capability(
        id="echo", title="Echo",
        params_schema={"type": "object", "required": ["msg"],
                       "properties": {"msg": {"type": "string"}}},
        handler=boom)
    client.app.state.queue.handlers["echo"] = boom

    sid = upload(client).json()["source_id"]
    first = client.post("/jobs/echo", json={"source_id": sid,
                                            "params": {"msg": "will-error"}}).json()
    poll_done(client, first["job_id"])

    async def echo(ctx):
        return {"echo": ctx.params}
    client.app.state.queue.handlers["echo"] = echo
    second = client.post("/jobs/echo", json={"source_id": sid,
                                             "params": {"msg": "will-error"}}).json()
    assert not second.get("cached")
    assert second["job_id"] != first["job_id"]
