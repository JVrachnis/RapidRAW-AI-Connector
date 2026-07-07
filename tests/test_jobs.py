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
        # Row may already be gone if aggressive pruning (e.g. ttl_hours=0) raced
        # ahead of this poll; treat that as "reached a terminal state".
        if j is None or j["status"] in statuses:
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
async def test_stop_terminates_running_handler_without_orphan(db, tmp_path):
    started = asyncio.Event()
    finished = []
    async def hang(ctx):
        started.set()
        await asyncio.sleep(30)
        finished.append(ctx.job_id)   # must never run after stop()
        return {"late": True}
    q = make_queue(db, tmp_path, {"h": hang})
    await q.start()
    j = q.submit("h", "s", {}, "interactive")
    await asyncio.wait_for(started.wait(), 2)
    handler_task = q._running_task
    tasks_before = asyncio.all_tasks()
    await q.stop()
    await asyncio.sleep(0.1)          # give an orphan a chance to misbehave
    # The in-flight handler task must be terminated (cancelled/done), not left
    # running in the background as an orphan.
    assert handler_task.done(), "handler task must not survive stop()"
    orphans = [t for t in asyncio.all_tasks() - tasks_before if not t.done()]
    assert orphans == [], f"orphan tasks still running after stop(): {orphans}"
    assert finished == []             # no orphan completed
    row = q.get(j["job_id"])
    assert row["status"] == "running"  # left for crash recovery
    # restart requeues and completes nothing (handler hangs), so just verify requeue:
    q2 = make_queue(db, tmp_path, {"h": hang})
    await q2.start()
    for _ in range(100):
        st = q2.get(j["job_id"])["status"]
        if st == "running":
            break
        await asyncio.sleep(0.02)
    assert q2.get(j["job_id"])["status"] in ("queued", "running")
    await q2.stop()

@pytest.mark.asyncio
async def test_done_events_do_not_accumulate(db, tmp_path):
    async def ok(ctx):
        return {}
    q = make_queue(db, tmp_path, {"k": ok}, ttl_hours=0)  # everything prunable immediately
    await q.start()
    ids = [q.submit("k", "s", {}, "interactive")["job_id"] for _ in range(5)]
    for jid in ids:
        await wait_status(q, jid, {"done"})
    q._prune()
    assert len(q._done_events) == 0
    await q.stop()

@pytest.mark.asyncio
async def test_cancelled_by_user_set_drains_on_timeout(db, tmp_path):
    async def sleepy(ctx):
        await asyncio.sleep(10)
        return {}
    q = make_queue(db, tmp_path, {"s": sleepy}, timeout_s=0.15)
    await q.start()
    j = q.submit("s", "s", {}, "interactive")
    for _ in range(200):
        if q.get(j["job_id"])["status"] == "running":
            break
        await asyncio.sleep(0.01)
    q._cancelled_by_user.add(j["job_id"])  # simulate cancel losing the race
    done = await wait_status(q, j["job_id"], {"error", "cancelled"})
    assert q._cancelled_by_user == set()
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

@pytest.mark.asyncio
async def test_workdir_removed_after_job_completes(db, tmp_path):
    async def writes_file(ctx):
        (ctx.workdir / "scratch.bin").write_bytes(b"x" * 10)
        return {}
    q = make_queue(db, tmp_path, {"w": writes_file})
    await q.start()
    j = q.submit("w", "s", {}, "interactive")
    await wait_status(q, j["job_id"], {"done"})
    await asyncio.sleep(0.05)  # let the finally block run
    assert not (tmp_path / "jobs" / j["job_id"]).exists()
    await q.stop()

@pytest.mark.asyncio
async def test_prune_sweeps_orphan_workdirs(db, tmp_path):
    async def ok(ctx):
        return {}
    q = make_queue(db, tmp_path, {"k": ok})
    orphan = tmp_path / "jobs" / "deadbeefdeadbeef"
    orphan.mkdir(parents=True)
    (orphan / "junk.png").write_bytes(b"j")
    await q.start()
    q._prune()
    assert not orphan.exists()
    await q.stop()
