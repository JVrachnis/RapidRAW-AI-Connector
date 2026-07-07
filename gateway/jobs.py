import asyncio
import os
import signal
import traceback
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Awaitable, Callable, Optional
from .db import Db, now, dumps, loads

@dataclass
class JobContext:
    job_id: str
    source_id: str
    params: dict
    workdir: Path
    source: object = None          # SourceRecord, attached by app wiring; None in queue unit tests
    settings: object = None        # engine.Settings, attached by app wiring
    _queue: "JobQueue" = None
    _proc: Optional[asyncio.subprocess.Process] = None

    def set_progress(self, fraction: float) -> None:
        self._queue._set_progress(self.job_id, fraction)

    async def run_tool(self, cmd: list[str], timeout: Optional[float] = None) -> tuple[int, str, str]:
        """Run a subprocess in its own process group so cancel/timeout can kill the tree.
        Returns (returncode, stdout, stderr)."""
        proc = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            preexec_fn=os.setsid)
        self._proc = proc
        try:
            out, err = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except ProcessLookupError:
                pass
            raise
        finally:
            self._proc = None
        return proc.returncode, out.decode(errors="replace"), err.decode(errors="replace")

class JobQueue:
    def __init__(self, db: Db, workdir_root: Path,
                 handlers: dict[str, Callable[[JobContext], Awaitable[dict]]],
                 job_timeout_s: float, result_ttl_hours: float,
                 make_context: Optional[Callable[[JobContext], JobContext]] = None):
        self.db = db
        self.workdir_root = workdir_root
        self.handlers = handlers
        self.job_timeout_s = job_timeout_s
        self.result_ttl_hours = result_ttl_hours
        self.make_context = make_context or (lambda ctx: ctx)
        self._wake = asyncio.Event()
        self._task: Optional[asyncio.Task] = None
        self._running_ctx: Optional[JobContext] = None
        self._running_task: Optional[asyncio.Task] = None
        self._cancelled_by_user: set[str] = set()
        self._done_events: dict[str, asyncio.Event] = {}

    async def start(self):
        # crash recovery: anything 'running' at last shutdown goes back to queued
        self.db.execute("UPDATE jobs SET status='queued', started=NULL WHERE status='running'")
        self._task = asyncio.create_task(self._worker())
        self._wake.set()

    async def stop(self):
        # Capture the in-flight handler task BEFORE touching the worker loop:
        # the worker's own CancelledError handling clears self._running_task
        # in its `finally` block, so this reference would be lost otherwise.
        handler_task = self._running_task
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        # Terminate the handler task too, so no orphan keeps running after
        # stop() returns. This must NOT go through the user-cancel path (the
        # DB row must remain 'running' for crash recovery on the next start()).
        if handler_task and not handler_task.done():
            handler_task.cancel()
            try:
                await handler_task
            except BaseException:
                pass

    def submit(self, capability: str, source_id: str, params: dict, priority: str) -> dict:
        job_id = uuid.uuid4().hex
        self.db.execute(
            "INSERT INTO jobs(id,capability,source_id,params,priority,status,created)"
            " VALUES(?,?,?,?,?,'queued',?)",
            (job_id, capability, source_id, dumps(params), priority, now()))
        self._done_events[job_id] = asyncio.Event()
        self._wake.set()
        return {"job_id": job_id, "status": "queued",
                "queue_position": self.queue_position(job_id)}

    def queue_position(self, job_id: str) -> int:
        rows = self.db.query(
            "SELECT id FROM jobs WHERE status='queued'"
            " ORDER BY CASE priority WHEN 'interactive' THEN 0 ELSE 1 END, created")
        for i, r in enumerate(rows):
            if r["id"] == job_id:
                return i
        return 0

    def get(self, job_id: str) -> Optional[dict]:
        r = self.db.query_one("SELECT * FROM jobs WHERE id=?", (job_id,))
        if r is None:
            return None
        d = {"job_id": r["id"], "capability": r["capability"], "status": r["status"],
             "progress": r["progress"], "result": loads(r["result"]), "error": loads(r["error"])}
        if r["status"] == "queued":
            d["queue_position"] = self.queue_position(r["id"])
        return d

    def list_jobs(self) -> list[dict]:
        return [self.get(r["id"]) for r in
                self.db.query("SELECT id FROM jobs ORDER BY created DESC LIMIT 200")]

    def cancel(self, job_id: str) -> bool:
        r = self.db.query_one("SELECT status FROM jobs WHERE id=?", (job_id,))
        if r is None:
            return False
        if r["status"] == "queued":
            self._finish(job_id, "cancelled", error={"kind": "cancelled", "detail": "cancelled while queued"})
            return True
        if r["status"] == "running" and self._running_ctx and self._running_ctx.job_id == job_id:
            self._cancelled_by_user.add(job_id)
            self._running_task.cancel()
            return True
        return False

    async def wait(self, job_id: str, timeout: float) -> dict:
        ev = self._done_events.get(job_id)
        if ev:
            await asyncio.wait_for(ev.wait(), timeout)
        return self.get(job_id)

    def depth(self) -> int:
        return self.db.query_one("SELECT COUNT(*) AS n FROM jobs WHERE status IN ('queued','running')")["n"]

    def _set_progress(self, job_id: str, fraction: float):
        self.db.execute("UPDATE jobs SET progress=? WHERE id=?", (fraction, job_id))

    def _finish(self, job_id: str, status: str, result: dict | None = None, error: dict | None = None):
        self.db.execute("UPDATE jobs SET status=?, finished=?, result=?, error=? WHERE id=?",
                        (status, now(), dumps(result) if result else None,
                         dumps(error) if error else None, job_id))
        ev = self._done_events.get(job_id)
        if ev:
            ev.set()

    def _next_queued(self):
        return self.db.query_one(
            "SELECT * FROM jobs WHERE status='queued'"
            " ORDER BY CASE priority WHEN 'interactive' THEN 0 ELSE 1 END, created LIMIT 1")

    def _prune(self):
        cutoff = now() - self.result_ttl_hours * 3600
        self.db.execute("DELETE FROM jobs WHERE status IN ('done','error','cancelled') AND finished < ?",
                        (cutoff,))
        # Sweep _done_events for entries whose job row is gone. We only drop
        # entries whose event is already set (i.e. the job reached a terminal
        # state and _finish() ran) so we never strand a waiter that is
        # currently blocked in wait() on an unset event.
        dead = [jid for jid, ev in self._done_events.items()
                if ev.is_set() and self.db.query_one("SELECT 1 FROM jobs WHERE id=?", (jid,)) is None]
        for jid in dead:
            del self._done_events[jid]

    async def _worker(self):
        while True:
            row = self._next_queued()
            if row is None:
                self._wake.clear()
                self._prune()
                try:
                    await asyncio.wait_for(self._wake.wait(), timeout=30)
                except asyncio.TimeoutError:
                    pass
                continue
            job_id = row["id"]
            handler = self.handlers.get(row["capability"])
            if handler is None:
                self._finish(job_id, "error",
                             error={"kind": "unknown_capability", "detail": row["capability"]})
                continue
            self.db.execute("UPDATE jobs SET status='running', started=? WHERE id=?", (now(), job_id))
            workdir = self.workdir_root / job_id
            workdir.mkdir(parents=True, exist_ok=True)
            ctx = self.make_context(JobContext(
                job_id=job_id, source_id=row["source_id"], params=loads(row["params"]),
                workdir=workdir, _queue=self))
            self._running_ctx = ctx
            self._running_task = asyncio.create_task(handler(ctx))
            try:
                result = await asyncio.wait_for(asyncio.shield(self._running_task),
                                                timeout=self.job_timeout_s)
                self._finish(job_id, "done", result=result)
            except asyncio.TimeoutError:
                self._running_task.cancel()
                try:
                    await self._running_task
                except (asyncio.CancelledError, Exception):
                    pass
                self._cancelled_by_user.discard(job_id)  # a racing user-cancel lost to the timeout
                self._finish(job_id, "error", error={"kind": "timeout",
                             "detail": f"exceeded {self.job_timeout_s}s"})
            except asyncio.CancelledError:
                if job_id in self._cancelled_by_user:
                    self._cancelled_by_user.discard(job_id)
                    try:
                        await self._running_task
                    except (asyncio.CancelledError, Exception):
                        pass
                    self._finish(job_id, "cancelled",
                                 error={"kind": "cancelled", "detail": "cancelled while running"})
                else:
                    raise  # queue itself is being stopped
            except Exception as e:
                self._cancelled_by_user.discard(job_id)  # a racing user-cancel lost to this error
                self._finish(job_id, "error", error={
                    "kind": getattr(e, "kind", "handler_error"),
                    "detail": f"{e}\n{traceback.format_exc(limit=5)}"})
            finally:
                self._running_ctx = None
                self._running_task = None
