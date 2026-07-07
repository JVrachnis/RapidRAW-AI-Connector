import asyncio
import pytest
from gateway.db import Db
from gateway.jobs import JobQueue, JobContext

@pytest.fixture
def db(tmp_path):
    return Db(tmp_path / "g.sqlite3")

def make_queue(db, tmp_path, handlers, timeout_s=5, ttl_hours=24):
    return JobQueue(db=db, workdir_root=tmp_path / "jobs", handlers=handlers,
                    job_timeout_s=timeout_s, result_ttl_hours=ttl_hours)

async def wait_status(q, job_id, statuses, timeout=5.0):
    for _ in range(int(timeout / 0.02)):
        j = q.get(job_id)
        if j["status"] in statuses:
            return j
        await asyncio.sleep(0.02)
    raise TimeoutError(q.get(job_id))

@pytest.mark.asyncio
async def test_job_runs_and_stores_result(db, tmp_path):
    async def ok(ctx: JobContext):
        ctx.set_progress(0.5)
        return {"answer": ctx.params["x"] * 2}
    q = make_queue(db, tmp_path, {"double": ok})
    await q.start()
    j = q.submit("double", "src1", {"x": 21}, "interactive")
    done = await wait_status(q, j["job_id"], {"done"})
    assert done["result"] == {"answer": 42}
    await q.stop()

@pytest.mark.asyncio
async def test_interactive_beats_batch(db, tmp_path):
    order = []
    gate = asyncio.Event()
    async def slow(ctx):
        order.append(ctx.params["tag"])
        if ctx.params["tag"] == "first":
            await gate.wait()
        return {}
    q = make_queue(db, tmp_path, {"c": slow})
    await q.start()
    q.submit("c", "s", {"tag": "first"}, "interactive")     # occupies worker
    q.submit("c", "s", {"tag": "batch"}, "batch")
    q.submit("c", "s", {"tag": "inter"}, "interactive")     # must run before batch
    await asyncio.sleep(0.1)
    gate.set()
    for _ in range(200):
        if len(order) == 3:
            break
        await asyncio.sleep(0.02)
    assert order == ["first", "inter", "batch"]
    await q.stop()

@pytest.mark.asyncio
async def test_handler_error_recorded(db, tmp_path):
    async def boom(ctx):
        raise RuntimeError("kapow")
    q = make_queue(db, tmp_path, {"b": boom})
    await q.start()
    j = q.submit("b", "s", {}, "interactive")
    done = await wait_status(q, j["job_id"], {"error"})
    assert "kapow" in done["error"]["detail"]
    await q.stop()

@pytest.mark.asyncio
async def test_timeout(db, tmp_path):
    async def sleepy(ctx):
        await asyncio.sleep(10)
        return {}
    q = make_queue(db, tmp_path, {"s": sleepy}, timeout_s=0.1)
    await q.start()
    j = q.submit("s", "s", {}, "interactive")
    done = await wait_status(q, j["job_id"], {"error"})
    assert done["error"]["kind"] == "timeout"
    await q.stop()

@pytest.mark.asyncio
async def test_cancel_queued(db, tmp_path):
    gate = asyncio.Event()
    async def block(ctx):
        await gate.wait()
        return {}
    q = make_queue(db, tmp_path, {"c": block})
    await q.start()
    q.submit("c", "s", {}, "interactive")
    j2 = q.submit("c", "s", {}, "interactive")
    assert q.cancel(j2["job_id"]) is True
    assert q.get(j2["job_id"])["status"] == "cancelled"
    gate.set()
    await q.stop()

@pytest.mark.asyncio
async def test_startup_recovery_requeues_running(db, tmp_path):
    db.execute("INSERT INTO jobs(id,capability,source_id,params,priority,status,created)"
               " VALUES('j1','x','s','{}','interactive','running',1.0)")
    async def ok(ctx):
        return {"ok": 1}
    q = make_queue(db, tmp_path, {"x": ok})
    await q.start()   # recovery happens on start
    done = await wait_status(q, "j1", {"done"})
    assert done["result"] == {"ok": 1}
    await q.stop()
