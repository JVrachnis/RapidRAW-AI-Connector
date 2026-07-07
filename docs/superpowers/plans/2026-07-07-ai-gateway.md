# rr-ai-gateway Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the RapidRAW AI Connector into a general AI-operations gateway: capability registry, priority job queue with SQLite journal, content-addressed source cache (TIFF/RAW/std + EXIF + rrdata), a common job API, a v1 `mask` capability wrapping the proven scripts on inferno, and back-compat `/inpaint`.

**Architecture:** FastAPI app built by a `create_app()` factory. Singletons (settings, SQLite `Db`, `SourceStore`, `JobQueue`) live on `app.state`. Capabilities are modules under `capabilities/` that self-register into a global registry at import; handlers are async functions receiving a `JobContext` (source record, params, workdir, cancellable subprocess runner). Legacy endpoints are thin adapters over the same machinery.

**Tech Stack:** Python 3.10+, FastAPI, uvicorn, sqlite3 (stdlib), blake3, jsonschema, Pillow, numpy, aiohttp (existing ComfyClient), pytest + fastapi TestClient.

**Spec:** `docs/superpowers/specs/2026-07-07-ai-gateway-design.md`
**Repo/branch:** `~/Apps/RapidRawFork/rr-ai-gateway`, branch `feat/ai-gateway`

## File Structure

```
main.py                          (rewired: logging + create_app())
engine.py                        (existing; Settings gains GATEWAY_* fields; ComfyClient/ImageProcessor/build_workflow unchanged)
gateway/__init__.py
gateway/hashing.py               content_id() + kind detection
gateway/db.py                    thread-safe sqlite wrapper + schema
gateway/store.py                 SourceStore (content-addressed, sidecars, eviction, 410 semantics)
gateway/registry.py              Capability dataclass + registry + discovery
gateway/jobs.py                  Job model, JobContext, JobQueue + worker
gateway/auth.py                  optional bearer-token dependency
gateway/routes.py                /capabilities /sources /jobs/{cap} /jobs/{id} /queue /health
gateway/legacy.py                /upload_source /inpaint adapters
gateway/app.py                   create_app() factory
capabilities/__init__.py
capabilities/inpaint.py          existing inpaint flow as a capability
capabilities/mask.py             mask capability (shells to inferno scripts)
capabilities/mask_tools/mask_points.py   SAM2 point-prompt script (runs in ComfyUI venv on inferno)
deploy/rr-ai-gateway.service     systemd user unit
tests/conftest.py, tests/test_*.py
requirements.txt (+ blake3, jsonschema), requirements-dev.txt
```

All commands below run from the repo root. Python for tests: `python -m venv .venv && .venv/bin/pip install -r requirements.txt -r requirements-dev.txt`, then `PY=.venv/bin/python`.

---

### Task 1: Test scaffolding + content hashing

**Files:**
- Create: `requirements-dev.txt`, `pytest.ini`, `gateway/__init__.py`, `gateway/hashing.py`
- Modify: `requirements.txt`
- Test: `tests/test_hashing.py`

- [ ] **Step 1: Add dependencies**

`requirements.txt` — append:
```
blake3
jsonschema
```

`requirements-dev.txt`:
```
pytest
httpx
```

`pytest.ini`:
```ini
[pytest]
testpaths = tests
markers =
    integration: requires inferno (GPU, ComfyUI, masking scripts); deselected by default
addopts = -m "not integration"
```

Run: `python -m venv .venv && .venv/bin/pip install -r requirements.txt -r requirements-dev.txt`

- [ ] **Step 2: Write the failing test**

`tests/test_hashing.py`:
```python
from gateway.hashing import content_id, detect_kind

def test_content_id_is_stable_and_content_addressed():
    a = content_id(b"hello")
    assert a == content_id(b"hello")
    assert a != content_id(b"hello!")
    assert len(a) == 64 and all(c in "0123456789abcdef" for c in a)

def test_detect_kind():
    assert detect_kind("shot.ARW") == "raw"
    assert detect_kind("x.dng") == "raw"
    assert detect_kind("linear.tiff") == "tiff"
    assert detect_kind("linear.TIF") == "tiff"
    assert detect_kind("img.jpg") == "std"
    assert detect_kind("noext") == "std"
```

- [ ] **Step 3: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_hashing.py -v` — Expected: FAIL (`ModuleNotFoundError: gateway`)

- [ ] **Step 4: Implement**

`gateway/__init__.py`: empty file.

`gateway/hashing.py`:
```python
import os
import blake3

RAW_EXTS = {".arw", ".cr2", ".cr3", ".nef", ".dng", ".raf", ".orf", ".rw2", ".pef", ".srw"}
TIFF_EXTS = {".tif", ".tiff"}

def content_id(data: bytes) -> str:
    return blake3.blake3(data).hexdigest()

def detect_kind(filename: str) -> str:
    ext = os.path.splitext(filename)[1].lower()
    if ext in RAW_EXTS:
        return "raw"
    if ext in TIFF_EXTS:
        return "tiff"
    return "std"
```

- [ ] **Step 5: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_hashing.py -v` — Expected: 2 PASS

- [ ] **Step 6: Commit**

```bash
git add requirements.txt requirements-dev.txt pytest.ini gateway tests/test_hashing.py
git commit -m "feat(gateway): scaffolding, content hashing, kind detection"
```

---

### Task 2: Settings extension

**Files:**
- Modify: `engine.py:22-30` (Settings class)
- Test: `tests/test_settings.py`

- [ ] **Step 1: Write the failing test**

`tests/test_settings.py`:
```python
from engine import Settings

def test_gateway_settings_defaults(tmp_path, monkeypatch):
    monkeypatch.setenv("CACHE_DIR", str(tmp_path))
    s = Settings()
    assert s.GATEWAY_TOKEN is None
    assert s.GATEWAY_CACHE_MAX_GB == 20
    assert s.GATEWAY_RESULT_TTL_HOURS == 24
    assert s.GATEWAY_JOB_TIMEOUT_S == 600
    assert s.GATEWAY_TOOLS_DIR.endswith("comfy")
    assert str(s.gateway_db_path).endswith("gateway.sqlite3")

def test_gateway_settings_env_override(tmp_path, monkeypatch):
    monkeypatch.setenv("GATEWAY_TOKEN", "sekrit")
    monkeypatch.setenv("GATEWAY_JOB_TIMEOUT_S", "5")
    monkeypatch.setenv("GATEWAY_DB_PATH", str(tmp_path / "j.db"))
    s = Settings()
    assert s.GATEWAY_TOKEN == "sekrit"
    assert s.GATEWAY_JOB_TIMEOUT_S == 5
    assert s.gateway_db_path == tmp_path / "j.db"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_settings.py -v` — Expected: FAIL (`AttributeError: GATEWAY_TOKEN`)

- [ ] **Step 3: Implement**

In `engine.py`, inside `class Settings(BaseSettings)` after `MAX_CACHE_SIZE_MB: int = 2048` add:
```python
    GATEWAY_TOKEN: Optional[str] = None
    GATEWAY_TOOLS_DIR: str = os.path.expanduser("~/comfy")
    GATEWAY_CACHE_MAX_GB: int = 20
    GATEWAY_RESULT_TTL_HOURS: int = 24
    GATEWAY_JOB_TIMEOUT_S: int = 600
    GATEWAY_DB_PATH: Optional[Path] = None
    GATEWAY_COMFY_VENV_PY: str = os.path.expanduser("~/comfy/ComfyUI/.venv/bin/python")
    GATEWAY_RAWTOOLS_PY: str = os.path.expanduser("~/rawtools/bin/python")

    @property
    def gateway_db_path(self) -> Path:
        return self.GATEWAY_DB_PATH or (self.CACHE_DIR / "gateway.sqlite3")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_settings.py -v` — Expected: 2 PASS

- [ ] **Step 5: Commit**

```bash
git add engine.py tests/test_settings.py
git commit -m "feat(gateway): GATEWAY_* settings"
```

---

### Task 3: Db wrapper + SourceStore

**Files:**
- Create: `gateway/db.py`, `gateway/store.py`
- Test: `tests/test_store.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_store.py`:
```python
import io
import pytest
from PIL import Image
from gateway.db import Db
from gateway.store import SourceStore, SourceEvicted

def make_store(tmp_path, max_bytes=10**9):
    db = Db(tmp_path / "g.sqlite3")
    return SourceStore(db=db, root=tmp_path / "sources", max_bytes=max_bytes)

def png_bytes(w=8, h=6, color=(255, 0, 0)):
    buf = io.BytesIO()
    Image.new("RGB", (w, h), color).save(buf, "PNG")
    return buf.getvalue()

def test_add_returns_record_and_dedups(tmp_path):
    st = make_store(tmp_path)
    r1 = st.add(png_bytes(), "a.png")
    r2 = st.add(png_bytes(), "b.png")   # same bytes -> same id
    assert r1["source_id"] == r2["source_id"]
    assert r1["kind"] == "std"
    assert (r1["width"], r1["height"]) == (8, 6)

def test_sidecars_and_get(tmp_path):
    st = make_store(tmp_path)
    r = st.add(png_bytes(), "a.png", exif={"ISO": 100}, rrdata={"rating": 3})
    rec = st.get(r["source_id"])
    assert rec.exif == {"ISO": 100}
    assert rec.rrdata == {"rating": 3}
    assert rec.path.exists()
    assert rec.kind == "std"

def test_raw_kind_no_dims_uses_client_dims(tmp_path):
    st = make_store(tmp_path)
    r = st.add(b"\x00" * 100, "shot.ARW", client_width=8640, client_height=5760)
    assert r["kind"] == "raw"
    rec = st.get(r["source_id"])
    assert (rec.width, rec.height) == (8640, 5760)

def test_unknown_id_returns_none(tmp_path):
    st = make_store(tmp_path)
    assert st.get("deadbeef" * 8) is None

def test_eviction_raises_gone_on_get(tmp_path):
    st = make_store(tmp_path, max_bytes=300)  # tiny cap
    r1 = st.add(png_bytes(color=(1, 2, 3)), "a.png")
    st.add(png_bytes(16, 12, (4, 5, 6)), "b.png")  # bigger, evicts a
    with pytest.raises(SourceEvicted):
        st.get(r1["source_id"])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_store.py -v` — Expected: FAIL (import errors)

- [ ] **Step 3: Implement Db**

`gateway/db.py`:
```python
import json
import sqlite3
import threading
import time
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS sources(
  id TEXT PRIMARY KEY, path TEXT NOT NULL, kind TEXT NOT NULL,
  width INTEGER, height INTEGER, exif TEXT, rrdata TEXT,
  size INTEGER NOT NULL, created REAL NOT NULL, last_used REAL NOT NULL,
  evicted INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS jobs(
  id TEXT PRIMARY KEY, capability TEXT NOT NULL, source_id TEXT NOT NULL,
  params TEXT NOT NULL, priority TEXT NOT NULL, status TEXT NOT NULL,
  created REAL NOT NULL, started REAL, finished REAL,
  progress REAL, result TEXT, error TEXT
);
CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status, priority, created);
"""

class Db:
    """Tiny thread-safe sqlite wrapper. All operations are short; a single
    lock keeps things simple (the gateway is one process, low QPS)."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock:
            self._conn.executescript(SCHEMA)
            self._conn.commit()

    def execute(self, sql: str, params: tuple = ()):
        with self._lock:
            cur = self._conn.execute(sql, params)
            self._conn.commit()
            return cur

    def query(self, sql: str, params: tuple = ()) -> list[sqlite3.Row]:
        with self._lock:
            return self._conn.execute(sql, params).fetchall()

    def query_one(self, sql: str, params: tuple = ()):
        rows = self.query(sql, params)
        return rows[0] if rows else None

def now() -> float:
    return time.time()

def dumps(obj) -> str:
    return json.dumps(obj, separators=(",", ":"))

def loads(s):
    return json.loads(s) if s else None
```

- [ ] **Step 4: Implement SourceStore**

`gateway/store.py`:
```python
import io
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from PIL import Image
from .db import Db, now, dumps, loads
from .hashing import content_id, detect_kind

class SourceEvicted(Exception):
    """Source was known but its bytes were evicted; client must re-upload (HTTP 410)."""

@dataclass
class SourceRecord:
    source_id: str
    path: Path
    kind: str
    width: Optional[int]
    height: Optional[int]
    exif: Optional[dict]
    rrdata: Optional[dict]

class SourceStore:
    def __init__(self, db: Db, root: Path, max_bytes: int):
        self.db = db
        self.root = root
        self.max_bytes = max_bytes
        self.root.mkdir(parents=True, exist_ok=True)

    def add(self, content: bytes, filename: str, exif: dict | None = None,
            rrdata: dict | None = None, client_width: int | None = None,
            client_height: int | None = None) -> dict:
        sid = content_id(content)
        kind = detect_kind(filename)
        row = self.db.query_one("SELECT * FROM sources WHERE id=?", (sid,))
        if row and not row["evicted"] and Path(row["path"]).exists():
            # dedup hit; refresh sidecars if newly provided
            self.db.execute(
                "UPDATE sources SET last_used=?, exif=COALESCE(?,exif), rrdata=COALESCE(?,rrdata) WHERE id=?",
                (now(), dumps(exif) if exif else None, dumps(rrdata) if rrdata else None, sid))
            return {"source_id": sid, "kind": row["kind"],
                    "width": row["width"], "height": row["height"]}

        ext = os.path.splitext(filename)[1].lower() or ".bin"
        path = self.root / f"{sid}{ext}"
        path.write_bytes(content)

        width, height = client_width, client_height
        if kind != "raw":
            try:
                with Image.open(io.BytesIO(content)) as im:
                    width, height = im.size
            except Exception:
                pass

        self.db.execute(
            "INSERT OR REPLACE INTO sources(id,path,kind,width,height,exif,rrdata,size,created,last_used,evicted)"
            " VALUES(?,?,?,?,?,?,?,?,?,?,0)",
            (sid, str(path), kind, width, height,
             dumps(exif) if exif else None, dumps(rrdata) if rrdata else None,
             len(content), now(), now()))
        self._enforce_limit()
        return {"source_id": sid, "kind": kind, "width": width, "height": height}

    def get(self, source_id: str) -> Optional[SourceRecord]:
        row = self.db.query_one("SELECT * FROM sources WHERE id=?", (source_id,))
        if row is None:
            return None
        if row["evicted"] or not Path(row["path"]).exists():
            raise SourceEvicted(source_id)
        self.db.execute("UPDATE sources SET last_used=? WHERE id=?", (now(), source_id))
        return SourceRecord(source_id=source_id, path=Path(row["path"]), kind=row["kind"],
                            width=row["width"], height=row["height"],
                            exif=loads(row["exif"]), rrdata=loads(row["rrdata"]))

    def _enforce_limit(self):
        rows = self.db.query(
            "SELECT id,path,size FROM sources WHERE evicted=0 ORDER BY last_used ASC")
        total = sum(r["size"] for r in rows)
        for r in rows:
            if total <= self.max_bytes:
                break
            try:
                Path(r["path"]).unlink(missing_ok=True)
            finally:
                self.db.execute("UPDATE sources SET evicted=1 WHERE id=?", (r["id"],))
                total -= r["size"]
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_store.py -v` — Expected: 5 PASS

Note: `test_eviction_raises_gone_on_get` needs the *first* file evicted, not the second. Eviction is LRU by `last_used`; the second `add` happens later so the first is oldest. If the combined size of both PNGs is under 300 bytes, raise the image sizes until eviction triggers (PNG of 8×6 is ~80 bytes; 16×12 ~120; adjust `max_bytes=150` if needed to force eviction of exactly one).

- [ ] **Step 6: Commit**

```bash
git add gateway/db.py gateway/store.py tests/test_store.py
git commit -m "feat(gateway): sqlite db + content-addressed source store with sidecars and eviction"
```

---

### Task 4: Capability registry

**Files:**
- Create: `gateway/registry.py`, `capabilities/__init__.py`
- Test: `tests/test_registry.py`

- [ ] **Step 1: Write the failing test**

`tests/test_registry.py`:
```python
import pytest
from gateway import registry

@pytest.fixture(autouse=True)
def clean_registry():
    saved = dict(registry.REGISTRY)
    registry.REGISTRY.clear()
    yield
    registry.REGISTRY.clear()
    registry.REGISTRY.update(saved)

async def dummy_handler(ctx):
    return {"ok": True}

def test_register_and_describe():
    cap = registry.Capability(
        id="echo", title="Echo", params_schema={"type": "object"}, handler=dummy_handler)
    registry.register(cap)
    assert registry.get("echo") is cap
    desc = registry.describe()
    assert desc == [{"id": "echo", "title": "Echo",
                     "params_schema": {"type": "object"}, "modes": None, "presets": None}]

def test_duplicate_id_rejected():
    cap = registry.Capability(id="x", title="X", params_schema={}, handler=dummy_handler)
    registry.register(cap)
    with pytest.raises(ValueError):
        registry.register(cap)

def test_validate_params_raises_on_schema_violation():
    cap = registry.Capability(
        id="v", title="V",
        params_schema={"type": "object", "required": ["mode"],
                       "properties": {"mode": {"enum": ["a", "b"]}}},
        handler=dummy_handler)
    registry.register(cap)
    registry.validate_params("v", {"mode": "a"})   # ok
    with pytest.raises(registry.ParamsInvalid):
        registry.validate_params("v", {"mode": "zzz"})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_registry.py -v` — Expected: FAIL (import error)

- [ ] **Step 3: Implement**

`capabilities/__init__.py`: empty file.

`gateway/registry.py`:
```python
import importlib
import pkgutil
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Optional
import jsonschema

class ParamsInvalid(Exception):
    pass

@dataclass
class Capability:
    id: str
    title: str
    params_schema: dict
    handler: Callable[["JobContext"], Awaitable[dict]]  # noqa: F821 (defined in gateway.jobs)
    modes: Optional[list] = None
    presets: Optional[list] = None

REGISTRY: dict[str, Capability] = {}

def register(cap: Capability) -> None:
    if cap.id in REGISTRY:
        raise ValueError(f"capability {cap.id!r} already registered")
    REGISTRY[cap.id] = cap

def get(cap_id: str) -> Optional[Capability]:
    return REGISTRY.get(cap_id)

def describe() -> list[dict]:
    return [{"id": c.id, "title": c.title, "params_schema": c.params_schema,
             "modes": c.modes, "presets": c.presets} for c in REGISTRY.values()]

def validate_params(cap_id: str, params: dict) -> None:
    cap = REGISTRY[cap_id]
    try:
        jsonschema.validate(params, cap.params_schema)
    except jsonschema.ValidationError as e:
        raise ParamsInvalid(e.message) from e

def load_capabilities() -> None:
    """Import every module in the capabilities package; modules call register() at import."""
    import capabilities
    for m in pkgutil.iter_modules(capabilities.__path__):
        if not m.name.startswith("_"):
            importlib.import_module(f"capabilities.{m.name}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_registry.py -v` — Expected: 3 PASS

- [ ] **Step 5: Commit**

```bash
git add gateway/registry.py capabilities/__init__.py tests/test_registry.py
git commit -m "feat(gateway): capability registry with jsonschema param validation"
```

---

### Task 5: Job queue + worker

**Files:**
- Create: `gateway/jobs.py`
- Test: `tests/test_jobs.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_jobs.py`:
```python
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
```

Also add `pytest-asyncio` to `requirements-dev.txt` and `asyncio_mode = auto` under `[pytest]` in `pytest.ini` (then `pip install -r requirements-dev.txt` again). With `asyncio_mode = auto`, the `@pytest.mark.asyncio` decorators are optional but harmless.

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_jobs.py -v` — Expected: FAIL (import error)

- [ ] **Step 3: Implement**

`gateway/jobs.py`:
```python
import asyncio
import os
import signal
import traceback
import uuid
from dataclasses import dataclass, field
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
        self._done_events: dict[str, asyncio.Event] = {}

    async def start(self):
        # crash recovery: anything 'running' at last shutdown goes back to queued
        self.db.execute("UPDATE jobs SET status='queued', started=NULL WHERE status='running'")
        self._task = asyncio.create_task(self._worker())
        self._wake.set()

    async def stop(self):
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
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
                self._finish(job_id, "error", error={"kind": "timeout",
                             "detail": f"exceeded {self.job_timeout_s}s"})
            except asyncio.CancelledError:
                if self._running_task.cancelled() or not self._running_task.done():
                    # cancel() was called on the running job
                    self._finish(job_id, "cancelled",
                                 error={"kind": "cancelled", "detail": "cancelled while running"})
                    continue
                raise  # queue itself is being stopped
            except Exception as e:
                self._finish(job_id, "error", error={
                    "kind": getattr(e, "kind", "handler_error"),
                    "detail": f"{e}\n{traceback.format_exc(limit=5)}"})
            finally:
                self._running_ctx = None
                self._running_task = None
```

Note on cancellation semantics: `cancel()` on a running job cancels `_running_task`; `asyncio.shield` makes `wait_for` see a `CancelledError` from the inner task without the worker loop itself dying. Verify with the tests; if the interplay proves finicky, the fallback is a plain `try: await self._running_task` and checking a `cancelled_flag` set by `cancel()` — tests define the contract, implementation may adapt.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_jobs.py -v` — Expected: 6 PASS

- [ ] **Step 5: Commit**

```bash
git add gateway/jobs.py tests/test_jobs.py requirements-dev.txt pytest.ini
git commit -m "feat(gateway): priority job queue with sqlite journal, timeout, cancel, recovery"
```

---

### Task 6: App factory, auth, and the common job API

**Files:**
- Create: `gateway/auth.py`, `gateway/routes.py`, `gateway/app.py`
- Test: `tests/conftest.py`, `tests/test_api.py`

- [ ] **Step 1: Write the failing tests**

`tests/conftest.py`:
```python
import json
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
```

`tests/test_api.py`:
```python
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
```

Auth test, `tests/test_auth.py`:
```python
import pytest
from fastapi.testclient import TestClient
from engine import Settings
from gateway.app import create_app

@pytest.fixture
def auth_client(tmp_path, monkeypatch):
    settings = Settings(CACHE_DIR=tmp_path, GATEWAY_TOKEN="sekrit")
    app = create_app(settings, load_caps=False)
    with TestClient(app) as c:
        yield c

def test_rejects_missing_token(auth_client):
    assert auth_client.get("/capabilities").status_code == 401

def test_accepts_bearer_token(auth_client):
    r = auth_client.get("/capabilities", headers={"Authorization": "Bearer sekrit"})
    assert r.status_code == 200

def test_health_is_open(auth_client):
    assert auth_client.get("/health").status_code == 200
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_api.py tests/test_auth.py -v` — Expected: FAIL (import errors)

- [ ] **Step 3: Implement auth**

`gateway/auth.py`:
```python
from fastapi import HTTPException, Request

def make_auth_dependency(token: str | None):
    async def check(request: Request):
        if token is None:
            return
        got = request.headers.get("authorization", "")
        if got != f"Bearer {token}":
            raise HTTPException(401, "invalid or missing bearer token")
    return check
```

- [ ] **Step 4: Implement routes**

`gateway/routes.py`:
```python
import json
from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from . import registry
from .registry import ParamsInvalid
from .store import SourceEvicted

router = APIRouter()

@router.get("/capabilities")
async def capabilities():
    return registry.describe()

@router.post("/sources")
async def add_source(request: Request, file: UploadFile = File(...),
                     exif: str | None = Form(None), rrdata: str | None = Form(None),
                     client_width: int | None = Form(None),
                     client_height: int | None = Form(None)):
    content = await file.read()
    try:
        exif_obj = json.loads(exif) if exif else None
        rrdata_obj = json.loads(rrdata) if rrdata else None
    except json.JSONDecodeError as e:
        raise HTTPException(422, f"sidecar is not valid JSON: {e}")
    return request.app.state.store.add(
        content, file.filename or "upload.bin", exif=exif_obj, rrdata=rrdata_obj,
        client_width=client_width, client_height=client_height)

@router.post("/jobs/{capability}", status_code=202)
async def submit_job(capability: str, request: Request, body: dict):
    if registry.get(capability) is None:
        raise HTTPException(404, f"unknown capability {capability!r}")
    source_id = body.get("source_id")
    params = body.get("params", {})
    priority = body.get("priority", "interactive")
    if priority not in ("interactive", "batch"):
        raise HTTPException(422, "priority must be 'interactive' or 'batch'")
    try:
        registry.validate_params(capability, params)
    except ParamsInvalid as e:
        raise HTTPException(422, str(e))
    try:
        if request.app.state.store.get(source_id) is None:
            raise HTTPException(404, "unknown source_id")
    except SourceEvicted:
        raise HTTPException(410, "source evicted from cache; re-upload")
    return request.app.state.queue.submit(capability, source_id, params, priority)

@router.get("/jobs/{job_id}")
async def job_status(job_id: str, request: Request):
    j = request.app.state.queue.get(job_id)
    if j is None:
        raise HTTPException(404, "unknown job")
    return j

@router.delete("/jobs/{job_id}")
async def cancel_job(job_id: str, request: Request):
    q = request.app.state.queue
    if q.get(job_id) is None:
        raise HTTPException(404, "unknown job")
    return {"cancelled": q.cancel(job_id)}

@router.get("/queue")
async def queue_list(request: Request):
    return request.app.state.queue.list_jobs()
```

- [ ] **Step 5: Implement app factory**

`gateway/app.py`:
```python
import logging
from contextlib import asynccontextmanager
from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from engine import ComfyClient, Settings
from . import registry
from .auth import make_auth_dependency
from .db import Db
from .jobs import JobContext, JobQueue
from .routes import router
from .store import SourceEvicted, SourceStore

logger = logging.getLogger("API")

def create_app(settings: Settings, load_caps: bool = True) -> FastAPI:
    if load_caps:
        registry.load_capabilities()

    db = Db(settings.gateway_db_path)
    store = SourceStore(db=db, root=settings.CACHE_DIR / "sources",
                        max_bytes=settings.GATEWAY_CACHE_MAX_GB * 1024 ** 3)

    def make_context(ctx: JobContext) -> JobContext:
        try:
            ctx.source = store.get(ctx.source_id)
        except SourceEvicted:
            ctx.source = None
        ctx.settings = settings
        return ctx

    queue = JobQueue(
        db=db, workdir_root=settings.CACHE_DIR / "jobs",
        handlers={c.id: c.handler for c in registry.REGISTRY.values()},
        job_timeout_s=settings.GATEWAY_JOB_TIMEOUT_S,
        result_ttl_hours=settings.GATEWAY_RESULT_TTL_HOURS,
        make_context=make_context)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logger.info(f"rr-ai-gateway starting; capabilities: {list(registry.REGISTRY)}")
        await queue.start()
        yield
        await queue.stop()

    auth = make_auth_dependency(settings.GATEWAY_TOKEN)
    app = FastAPI(title="rr-ai-gateway", lifespan=lifespan)
    app.add_middleware(CORSMiddleware, allow_origins=["*"],
                       allow_methods=["*"], allow_headers=["*"])
    app.state.settings = settings
    app.state.store = store
    app.state.queue = queue
    app.include_router(router, dependencies=[Depends(auth)])

    @app.get("/health")
    async def health():
        comfy_up = await ComfyClient.check_health()
        return {"status": "ok" if comfy_up else "degraded",
                "comfy_url": settings.comfy_url, "connected": comfy_up,
                "capabilities": list(registry.REGISTRY),
                "queue_depth": queue.depth()}

    return app
```

Note: `queue.handlers` is captured at creation from the registry — in tests capabilities are registered before `create_app`, in production `load_capabilities()` runs first. `/health` is registered outside the auth-protected router on purpose (stock RapidRAW probes it without a token).

`ComfyClient.check_health()` in `engine.py` — confirm it is a `@staticmethod`/classmethod usable as `ComfyClient.check_health()` (it is called that way in the existing `main.py:79`); if it's an instance method, instantiate: `await ComfyClient().check_health()`.

- [ ] **Step 6: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_api.py tests/test_auth.py -v` — Expected: all PASS
Then full suite: `.venv/bin/pytest -v` — Expected: all PASS

- [ ] **Step 7: Commit**

```bash
git add gateway/auth.py gateway/routes.py gateway/app.py tests/conftest.py tests/test_api.py tests/test_auth.py
git commit -m "feat(gateway): app factory, bearer auth, common job API"
```

---

### Task 7: Inpaint capability + legacy adapters

**Files:**
- Create: `capabilities/inpaint.py`, `gateway/legacy.py`
- Modify: `gateway/app.py` (include legacy router)
- Test: `tests/test_legacy.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_legacy.py`:
```python
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
    assert "patch_data_base64" in body or "x" in body  # keys per ImageProcessor.crop_and_pack
```

Before finalizing the last assertion, read `ImageProcessor.crop_and_pack` in `engine.py` and assert the actual keys it returns (open the file; it returns the response dict sent to RapidRAW — use its real key names).

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_legacy.py -v` — Expected: FAIL (routes don't exist / capability missing)

- [ ] **Step 3: Implement the inpaint capability**

`capabilities/inpaint.py` (logic ported from the current `main.py` `/inpaint` body — keep behavior identical):
```python
import base64
import time
import aiofiles
from engine import ComfyClient, ImageProcessor, build_workflow
from gateway.registry import Capability, register

PARAMS_SCHEMA = {
    "type": "object",
    "required": ["prompt", "mask_image_base64"],
    "properties": {
        "prompt": {"type": "string"},
        "negative_prompt": {"type": "string",
                            "default": "blur, low quality, distortion, watermark"},
        "mask_image_base64": {"type": "string"},
        "seed": {"type": "integer"},
    },
}

class ComfyDown(Exception):
    kind = "comfyui_down"

async def handle(ctx) -> dict:
    if ctx.source is None:
        raise FileNotFoundError("source not available")
    mask_bytes = base64.b64decode(ctx.params["mask_image_base64"])
    processed = ImageProcessor.process_mask_for_comfyui(mask_bytes)
    mask_path = ctx.workdir / "mask.png"
    async with aiofiles.open(mask_path, "wb") as f:
        await f.write(processed)
    seed = ctx.params.get("seed") or int(time.time())
    workflow = build_workflow(
        str(ctx.source.path.absolute()), str(mask_path.absolute()),
        ctx.params["prompt"],
        ctx.params.get("negative_prompt", "blur, low quality, distortion, watermark"),
        seed)
    try:
        result_bytes = await ComfyClient().execute(workflow)
    except ConnectionError as e:
        raise ComfyDown(str(e)) from e
    return ImageProcessor.crop_and_pack(result_bytes, mask_bytes)

register(Capability(id="inpaint", title="Generative inpaint (ComfyUI)",
                    params_schema=PARAMS_SCHEMA, handler=handle))
```

Check `engine.py` for how `ComfyClient` is instantiated in current `main.py` (`client = ComfyClient()` then `await client.execute(workflow)`) — mirror exactly.

- [ ] **Step 4: Implement legacy adapters**

`gateway/legacy.py`:
```python
"""Byte-compatible endpoints for stock RapidRAW: /upload_source and /inpaint.
They map legacy source_ids (client-chosen strings) onto the content store via an
alias table, enqueue an interactive job, and wait for it server-side."""
from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from pydantic import BaseModel

router = APIRouter()

class InpaintPayload(BaseModel):
    source_id: str
    prompt: str
    negative_prompt: str = "blur, low quality, distortion, watermark"
    mask_image_base64: str
    seed: int = 0

# legacy client ids -> content-store ids (in-memory; legacy flow re-uploads freely)
_ALIASES: dict[str, str] = {}

@router.post("/upload_source")
async def upload_source(request: Request, file: UploadFile = File(...),
                        source_id: str = Form(...)):
    content = await file.read()
    rec = request.app.state.store.add(content, file.filename or "source.jpg")
    _ALIASES[source_id] = rec["source_id"]
    return {"status": "cached", "path": rec["source_id"]}

@router.post("/inpaint")
async def inpaint(request: Request, req: InpaintPayload):
    sid = _ALIASES.get(req.source_id)
    if sid is None:
        raise HTTPException(404, "Source ID not found. Upload required.")
    q = request.app.state.queue
    job = q.submit("inpaint", sid, {
        "prompt": req.prompt, "negative_prompt": req.negative_prompt,
        "mask_image_base64": req.mask_image_base64, "seed": req.seed}, "interactive")
    done = await q.wait(job["job_id"], timeout=request.app.state.settings.GATEWAY_JOB_TIMEOUT_S + 5)
    if done["status"] == "done":
        return done["result"]
    err = done.get("error") or {}
    if err.get("kind") == "comfyui_down":
        raise HTTPException(502, f"ComfyUI Unavailable: {err.get('detail')}")
    raise HTTPException(500, f"Processing error: {err.get('detail')}")
```

In `gateway/app.py`, after `app.include_router(router, ...)` add:
```python
    from .legacy import router as legacy_router
    app.include_router(legacy_router, dependencies=[Depends(auth)])
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_legacy.py -v` — Expected: 3 PASS. Full suite: `.venv/bin/pytest -v` — all PASS.

- [ ] **Step 6: Commit**

```bash
git add capabilities/inpaint.py gateway/legacy.py gateway/app.py tests/test_legacy.py
git commit -m "feat(gateway): inpaint as capability + byte-compatible legacy endpoints"
```

---

### Task 8: Rewire main.py

**Files:**
- Modify: `main.py` (replace app construction; keep logging config)

- [ ] **Step 1: Rewrite main.py**

Keep lines 1–75 (imports may shrink; the `LOGGING_CONFIG` dict and `EndpointFilter` stay). Delete the old lifespan, app, payload model, and the three route functions. The tail becomes:

```python
from engine import config
from gateway.app import create_app

app = create_app(config)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=config.HOST, port=config.PORT, log_config=LOGGING_CONFIG)
```

Remove now-unused imports (`uuid`, `base64`, `asyncio`, `aiofiles`, `UploadFile`, `File`, `Form`, `HTTPException`, `BaseModel`, `cache`, `ComfyClient`, `ImageProcessor`, `build_workflow`, `save_inputs_for_debug`, `CORSMiddleware`, `asynccontextmanager`). The `SourceCache` class in `engine.py` and `save_inputs_for_debug` stay untouched (unused by the gateway; deleting is unnecessary churn against upstream).

- [ ] **Step 2: Smoke test**

Run: `.venv/bin/python -c "import main; print(main.app.title)"` — Expected: `rr-ai-gateway`
Run: `.venv/bin/pytest -v` — Expected: all PASS

- [ ] **Step 3: Commit**

```bash
git add main.py
git commit -m "refactor: main.py builds the app via gateway.create_app"
```

---

### Task 9: Mask capability

**Files:**
- Create: `capabilities/mask.py`
- Test: `tests/test_mask_capability.py`

The handler shells to the inferno scripts. Unit tests never run the real tools: they monkeypatch `ctx.run_tool` with a recorder that also writes the expected output file, so we can assert exact command lines and result packing.

- [ ] **Step 1: Write the failing tests**

`tests/test_mask_capability.py`:
```python
import base64
import io
import json
import numpy as np
import pytest
from pathlib import Path
from PIL import Image
from engine import Settings
from gateway.db import Db
from gateway.jobs import JobContext
from gateway.store import SourceRecord
from capabilities import mask as M

def gray_png(path: Path, w=8, h=6, val=200):
    Image.new("L", (w, h), val).save(path)

def make_ctx(tmp_path, params, kind="tiff", exif=None, rrdata=None, dims=(8, 6)):
    src_path = tmp_path / ("src.tiff" if kind == "tiff" else "src.arw")
    if kind == "tiff":
        Image.new("RGB", dims, (50, 60, 70)).save(src_path, "TIFF")
    else:
        src_path.write_bytes(b"\x00" * 64)
    workdir = tmp_path / "job"
    workdir.mkdir()
    ctx = JobContext(job_id="j1", source_id="s1", params=params, workdir=workdir)
    ctx.source = SourceRecord(source_id="s1", path=src_path, kind=kind,
                              width=dims[0], height=dims[1], exif=exif, rrdata=rrdata)
    ctx.settings = Settings(CACHE_DIR=tmp_path)
    return ctx

class ToolRecorder:
    """Fake ctx.run_tool: records commands, writes the --out mask file."""
    def __init__(self, mask_val=200):
        self.calls = []
        self.mask_val = mask_val
    async def __call__(self, cmd, timeout=None):
        self.calls.append(cmd)
        if "--out" in cmd:
            out = Path(cmd[cmd.index("--out") + 1])
            gray_png(out, val=self.mask_val)
        if cmd[1].endswith("raw_develop.py"):
            outdir = Path(cmd[2])
            outdir.mkdir(parents=True, exist_ok=True)
            for ev in ("-2", "0", "2", "4"):
                Image.new("RGB", (8, 6)).save(outdir / f"ev{ev}.png")
            (outdir / "exif.json").write_text("{}")
        return 0, "detected: person 0.91\n", ""

@pytest.mark.asyncio
async def test_prompt_mode_calls_mask_hq(tmp_path):
    ctx = make_ctx(tmp_path, {"mode": "prompt", "query": "the dog."})
    rec = ToolRecorder(); ctx.run_tool = rec
    result = await M.handle(ctx)
    cmd = rec.calls[0]
    assert cmd[1].endswith("mask_hq.py")
    assert "--query" in cmd and cmd[cmd.index("--query") + 1] == "the dog."
    img = Image.open(io.BytesIO(base64.b64decode(result["mask_png_b64"])))
    assert img.size == (8, 6) and img.mode == "L"
    assert result["alignment"] == "exact"
    assert result["labels"] == ["person"]

@pytest.mark.asyncio
async def test_prompt_agentic_calls_mask_agentic(tmp_path):
    ctx = make_ctx(tmp_path, {"mode": "prompt", "query": "gear", "agentic": True,
                              "backend": "sam3"})
    rec = ToolRecorder(); ctx.run_tool = rec
    await M.handle(ctx)
    cmd = rec.calls[0]
    assert cmd[1].endswith("mask_agentic.py")
    assert cmd[cmd.index("--target") + 1] == "gear"
    assert cmd[cmd.index("--backend") + 1] == "sam3"

@pytest.mark.asyncio
async def test_paint_mode_passes_roi(tmp_path):
    roi = io.BytesIO(); Image.new("L", (8, 6), 255).save(roi, "PNG")
    ctx = make_ctx(tmp_path, {"mode": "paint",
                              "roi_mask_b64": base64.b64encode(roi.getvalue()).decode()})
    rec = ToolRecorder(); ctx.run_tool = rec
    await M.handle(ctx)
    cmd = rec.calls[0]
    assert cmd[1].endswith("mask_c2f.py")
    roi_arg = cmd[cmd.index("--roi") + 1]
    assert Path(roi_arg).exists()

@pytest.mark.asyncio
async def test_points_mode_calls_points_tool(tmp_path):
    ctx = make_ctx(tmp_path, {"mode": "points", "points": [[4, 3, 1], [1, 1, 0]]})
    rec = ToolRecorder(); ctx.run_tool = rec
    await M.handle(ctx)
    cmd = rec.calls[0]
    assert cmd[1].endswith("mask_points.py")
    pts = json.loads(cmd[cmd.index("--points") + 1])
    assert pts == [[4, 3, 1], [1, 1, 0]]

@pytest.mark.asyncio
async def test_preset_subject(tmp_path):
    ctx = make_ctx(tmp_path, {"mode": "preset", "preset": "subject"})
    rec = ToolRecorder(); ctx.run_tool = rec
    await M.handle(ctx)
    assert rec.calls[0][1].endswith("mask_hq.py")

@pytest.mark.asyncio
async def test_raw_source_develops_first_and_flags_alignment(tmp_path):
    ctx = make_ctx(tmp_path, {"mode": "prompt", "query": "x."}, kind="raw",
                   dims=(8, 6))
    rec = ToolRecorder(); ctx.run_tool = rec
    result = await M.handle(ctx)
    assert rec.calls[0][1].endswith("raw_develop.py")
    assert rec.calls[1][1].endswith(("mask_hq.py",))
    assert result["alignment"] == "best_effort"

@pytest.mark.asyncio
async def test_rrdata_tags_extend_query(tmp_path):
    ctx = make_ctx(tmp_path, {"mode": "prompt", "query": "the dog."},
                   rrdata={"tags": ["beach", "sunset"]})
    rec = ToolRecorder(); ctx.run_tool = rec
    await M.handle(ctx)
    q = rec.calls[0][rec.calls[0].index("--query") + 1]
    assert q == "the dog."   # tags only seed AGENTIC target expansion, not plain queries

def test_parse_labels_tolerant():
    assert M.parse_labels("detected: person 0.91\ndetected: dog 0.5\n") == ["person", "dog"]
    assert M.parse_labels("no matches here") == []

def test_reconcile_mask_center_crops_and_pads():
    m = np.full((10, 12), 255, np.uint8)
    out = M.reconcile_mask(m, target_w=8, target_h=6)
    assert out.shape == (6, 8)
    out2 = M.reconcile_mask(m, target_w=14, target_h=12)
    assert out2.shape == (12, 14)
    assert out2[0, 0] == 0  # padded border
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_mask_capability.py -v` — Expected: FAIL (module missing)

- [ ] **Step 3: Implement**

`capabilities/mask.py`:
```python
"""Mask capability: high-quality AI masking by shelling into the proven tools
living in GATEWAY_TOOLS_DIR (mask_hq.py, mask_agentic.py, mask_c2f.py,
raw_develop.py) plus the bundled mask_tools/mask_points.py."""
import base64
import io
import json
import os
import re
import time
import numpy as np
from pathlib import Path
from PIL import Image
from gateway.registry import Capability, register

PARAMS_SCHEMA = {
    "type": "object",
    "required": ["mode"],
    "properties": {
        "mode": {"enum": ["prompt", "points", "paint", "preset"]},
        "query": {"type": "string"},
        "points": {"type": "array",
                   "items": {"type": "array", "minItems": 3, "maxItems": 3,
                             "items": {"type": "number"}}},
        "roi_mask_b64": {"type": "string"},
        "preset": {"enum": ["subject", "sky", "foreground"]},
        "agentic": {"type": "boolean", "default": False},
        "backend": {"enum": ["sam2", "sam3"], "default": "sam2"},
        "matte": {"type": "boolean", "default": True},
        "ev_stack": {},
    },
    "allOf": [
        {"if": {"properties": {"mode": {"const": "prompt"}}},
         "then": {"required": ["mode", "query"]}},
        {"if": {"properties": {"mode": {"const": "points"}}},
         "then": {"required": ["mode", "points"]}},
        {"if": {"properties": {"mode": {"const": "paint"}}},
         "then": {"required": ["mode", "roi_mask_b64"]}},
        {"if": {"properties": {"mode": {"const": "preset"}}},
         "then": {"required": ["mode", "preset"]}},
    ],
}

LABEL_RE = re.compile(r"detected:\s*([a-zA-Z][\w \-]*?)\s+[\d.]+", re.MULTILINE)

def parse_labels(stdout: str) -> list[str]:
    return LABEL_RE.findall(stdout)

def reconcile_mask(mask: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    """Center-crop or zero-pad a mask to the client's stated dimensions (raw mode)."""
    h, w = mask.shape
    # crop
    if w > target_w:
        x0 = (w - target_w) // 2
        mask = mask[:, x0:x0 + target_w]
    if h > target_h:
        y0 = (h - target_h) // 2
        mask = mask[y0:y0 + target_h, :]
    # pad
    h, w = mask.shape
    if w < target_w or h < target_h:
        out = np.zeros((target_h, target_w), mask.dtype)
        y0 = (target_h - h) // 2
        x0 = (target_w - w) // 2
        out[y0:y0 + h, x0:x0 + w] = mask
        mask = out
    return mask

def _tool(settings, name: str) -> str:
    return str(Path(settings.GATEWAY_TOOLS_DIR) / name)

def _bundled(name: str) -> str:
    return str(Path(__file__).parent / "mask_tools" / name)

async def _develop_raw(ctx) -> Path:
    """raw_develop.py -> pick the 0EV frame as the working image."""
    outdir = ctx.workdir / "developed"
    cmd = [ctx.settings.GATEWAY_RAWTOOLS_PY, _tool(ctx.settings, "raw_develop.py"),
           str(ctx.source.path), str(outdir)]
    code, out, err = await ctx.run_tool(cmd)
    if code != 0:
        raise RuntimeError(f"raw_develop failed: {err[-800:]}")
    candidates = sorted(outdir.glob("*0*.png")) or sorted(outdir.glob("*.png"))
    if not candidates:
        raise RuntimeError("raw_develop produced no frames")
    return candidates[0]

async def handle(ctx) -> dict:
    t0 = time.perf_counter()
    if ctx.source is None:
        raise FileNotFoundError("source not available")
    p = ctx.params
    settings = ctx.settings
    py = settings.GATEWAY_COMFY_VENV_PY
    alignment = "exact"

    image_path = ctx.source.path
    if ctx.source.kind == "raw":
        image_path = await _develop_raw(ctx)
        alignment = "best_effort"

    out_path = ctx.workdir / "mask.png"
    mode = p["mode"]
    backend = p.get("backend", "sam2")

    if mode == "prompt" and p.get("agentic"):
        target = p["query"]
        rr = ctx.source.rrdata or {}
        tags = rr.get("tags") or []
        if tags:
            target = f"{target} (photo context: {', '.join(tags)})"
        cmd = [py, _tool(settings, "mask_agentic.py"), str(image_path),
               "--target", target, "--backend", backend, "--out", str(out_path)]
    elif mode == "prompt":
        cmd = [py, _tool(settings, "mask_hq.py"), str(image_path),
               "--query", p["query"], "--out", str(out_path)]
        if not p.get("matte", True):
            pass  # mask_hq always mattes; acceptable v1
    elif mode == "points":
        cmd = [py, _bundled("mask_points.py"), str(image_path),
               "--points", json.dumps(p["points"]), "--backend", backend,
               "--tools-dir", settings.GATEWAY_TOOLS_DIR, "--out", str(out_path)]
    elif mode == "paint":
        roi_path = ctx.workdir / "roi.png"
        roi_path.write_bytes(base64.b64decode(p["roi_mask_b64"]))
        cmd = [py, _tool(settings, "mask_c2f.py"), str(image_path),
               "--query", p.get("query") or "the main subject.",
               "--roi", str(roi_path), "--out", str(out_path)]
    else:  # preset
        preset = p["preset"]
        if preset == "subject":
            cmd = [py, _tool(settings, "mask_hq.py"), str(image_path),
                   "--query", "the main subject.", "--main-subject", "--out", str(out_path)]
        elif preset == "sky":
            cmd = [py, _tool(settings, "mask_hq.py"), str(image_path),
                   "--query", "sky.", "--out", str(out_path)]
        else:  # foreground
            cmd = [py, _tool(settings, "mask_hq.py"), str(image_path),
                   "--no-sam", "--query", "foreground.", "--out", str(out_path)]

    code, stdout, stderr = await ctx.run_tool(cmd)
    if code != 0:
        kind = "comfyui_down" if "ComfyUI" in stderr or "8188" in stderr else "tool_error"
        e = RuntimeError(f"mask tool failed: {stderr[-800:]}")
        e.kind = kind
        raise e

    mask = np.array(Image.open(out_path).convert("L"))
    tw, th = ctx.source.width, ctx.source.height
    if tw and th and (mask.shape[1], mask.shape[0]) != (tw, th):
        if ctx.source.kind == "raw":
            # resize to the developed base first if tool downscaled, then reconcile margins
            if abs(mask.shape[1] - tw) > 64 or abs(mask.shape[0] - th) > 64:
                mask = np.array(Image.fromarray(mask).resize((tw, th), Image.BILINEAR))
            mask = reconcile_mask(mask, tw, th)
        else:
            mask = np.array(Image.fromarray(mask).resize((tw, th), Image.BILINEAR))

    buf = io.BytesIO()
    Image.fromarray(mask).save(buf, "PNG")
    return {
        "mask_png_b64": base64.b64encode(buf.getvalue()).decode(),
        "width": int(mask.shape[1]), "height": int(mask.shape[0]),
        "labels": parse_labels(stdout),
        "alignment": alignment,
        "timings": {"total_s": round(time.perf_counter() - t0, 2)},
    }

register(Capability(
    id="mask", title="AI masking (GroundedSAM/SAM2/BiRefNet/ViTMatte)",
    params_schema=PARAMS_SCHEMA, handler=handle,
    modes=["prompt", "points", "paint", "preset"],
    presets=["subject", "sky", "foreground"]))
```

Implementation notes for the executor:
- `LABEL_RE` is best-effort: run `mask_hq.py` once on inferno, look at its real stdout, and adjust the regex to match its detection lines (spec allows `labels: []` when nothing parses — never fail the job on label parsing).
- `ev_stack` param is accepted by the schema but v1 wiring is: raw sources always get the default `raw_develop.py` stack; `--det-images` wiring into `mask_c2f.py` is a follow-up noted in the README (do not silently pretend it works).
- The `test_rrdata_tags_extend_query` test pins the decision that plain (non-agentic) queries are NOT mutated by tags — only the agentic target gets photo context.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_mask_capability.py -v` — Expected: 10 PASS. Full suite: all PASS.

- [ ] **Step 5: Commit**

```bash
git add capabilities/mask.py tests/test_mask_capability.py
git commit -m "feat(gateway): mask capability wrapping inferno masking tools"
```

---

### Task 10: mask_points.py tool

**Files:**
- Create: `capabilities/mask_tools/__init__.py` (empty), `capabilities/mask_tools/mask_points.py`

No local unit test (needs GPU + SAM2); covered by the integration suite (Task 11). Code review + `python -m py_compile` is the local gate.

- [ ] **Step 1: Implement**

`capabilities/mask_tools/mask_points.py`:
```python
#!/usr/bin/env python3
"""mask_points.py - SAM2/SAM3 point-prompt masking for the rr-ai-gateway.
Runs in the ComfyUI venv on inferno; imports the SAM2 predictor plumbing from
grounded_sam.py in --tools-dir (default ~/comfy).

Usage:
  mask_points.py IMAGE --points '[[x,y,1],[x,y,0]]' --backend sam2 --out mask.png
Point labels: 1 = foreground, 0 = background. Coordinates in image pixels.
"""
import argparse, json, os, sys
import numpy as np
from PIL import Image

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--points", required=True, help="JSON [[x,y,label],...]")
    ap.add_argument("--backend", choices=["sam2", "sam3"], default="sam2")
    ap.add_argument("--tools-dir", default=os.path.expanduser("~/comfy"))
    ap.add_argument("--out", default="mask.png")
    args = ap.parse_args()

    sys.path.insert(0, args.tools_dir)
    import grounded_sam as GS  # provides the cached SAM2 predictor plumbing

    image = Image.open(args.image).convert("RGB")
    pts = json.loads(args.points)
    coords = np.array([[p[0], p[1]] for p in pts], dtype=np.float32)
    labels = np.array([int(p[2]) for p in pts], dtype=np.int32)

    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    pred = SAM2ImagePredictor(build_sam2(GS.SAM2_CFG, GS.SAM2_CKPT, device=dev))
    pred.set_image(np.array(image))
    with torch.no_grad():
        masks, scores, _ = pred.predict(point_coords=coords, point_labels=labels,
                                        multimask_output=True)
    best = int(np.argmax(scores))
    mask = (masks[best].astype(np.float32) * 255).astype(np.uint8)
    Image.fromarray(mask, mode="L").save(args.out)
    print(f"detected: point-selection {float(scores[best]):.2f}")

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Compile check**

Run: `.venv/bin/python -m py_compile capabilities/mask_tools/mask_points.py` — Expected: exit 0

- [ ] **Step 3: Commit**

```bash
git add capabilities/mask_tools
git commit -m "feat(gateway): SAM2 point-prompt tool for the mask capability"
```

---

### Task 11: Integration tests (run on inferno)

**Files:**
- Create: `tests/integration/test_mask_integration.py`, `tests/integration/README.md`

- [ ] **Step 1: Write the integration tests**

`tests/integration/test_mask_integration.py`:
```python
"""Run ON INFERNO with ComfyUI up:  .venv/bin/pytest -m integration -v
Fixture image: put any JPEG with a clear single subject at tests/integration/fixtures/subject.jpg
(e.g. scp one of the sample shots; not committed to git)."""
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
```

`tests/integration/README.md`:
```markdown
# Integration tests

Run on inferno (GPU + ComfyUI + ~/comfy scripts required):

    rsync -a --exclude .venv --exclude cache ./ inferno:~/rr-ai-gateway/
    ssh inferno 'cd ~/rr-ai-gateway && python -m venv .venv && \
        .venv/bin/pip install -r requirements.txt -r requirements-dev.txt && \
        mkdir -p tests/integration/fixtures && \
        cp <some-subject-photo>.jpg tests/integration/fixtures/subject.jpg && \
        .venv/bin/pytest -m integration -v'

ComfyUI must be listening (COMFY_HOST/COMFY_PORT env if not 127.0.0.1:5545 —
on inferno ComfyUI runs on 8188, so: COMFY_PORT=8188).
```

- [ ] **Step 2: Verify collection locally (not execution)**

Run: `.venv/bin/pytest -m integration --collect-only -q` — Expected: 3 tests collected (skipped locally by the default `-m "not integration"` addopts when running the normal suite; `--collect-only -m integration` just proves they exist).
Run: `.venv/bin/pytest -v` — Expected: integration tests NOT selected, all others PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/integration
git commit -m "test(gateway): integration suite for the mask capability (runs on inferno)"
```

- [ ] **Step 4: Execute on inferno**

Follow `tests/integration/README.md` verbatim (rsync, venv, fixture, `COMFY_PORT=8188 .venv/bin/pytest -m integration -v`). Expected: 3 PASS. Fix the `LABEL_RE` regex in `capabilities/mask.py` here if real `mask_hq.py` stdout doesn't match (see Task 9 note), commit any fix:

```bash
git add capabilities/mask.py && git commit -m "fix(gateway): match real mask_hq stdout for label parsing"
```

---

### Task 12: Deployment unit + docs

**Files:**
- Create: `deploy/rr-ai-gateway.service`
- Modify: `README.md`

- [ ] **Step 1: systemd unit**

`deploy/rr-ai-gateway.service`:
```ini
[Unit]
Description=rr-ai-gateway (AI operations gateway for RapidRAW & friends)
After=network-online.target

[Service]
WorkingDirectory=%h/rr-ai-gateway
Environment=COMFY_HOST=127.0.0.1
Environment=COMFY_PORT=8188
Environment=CACHE_DIR=%h/rr-ai-gateway/cache
# Environment=GATEWAY_TOKEN=change-me
ExecStart=%h/rr-ai-gateway/.venv/bin/python main.py
Restart=on-failure
RestartSec=3

[Install]
WantedBy=default.target
```

- [ ] **Step 2: README section**

Append to `README.md`:
```markdown
## rr-ai-gateway (this fork)

This fork extends the connector into a general AI-operations gateway:

- `GET /capabilities` — discover capabilities + param schemas
- `POST /sources` — upload image (TIFF/RAW/JPEG) + optional `exif`/`rrdata` JSON once (content-addressed)
- `POST /jobs/{capability}` — enqueue; `GET /jobs/{id}` — poll; `DELETE /jobs/{id}` — cancel; `GET /queue` — list
- Capabilities v1: `mask` (GroundedSAM/SAM2/BiRefNet/ViTMatte via the tools in `GATEWAY_TOOLS_DIR`), `inpaint`
- Legacy `/upload_source` + `/inpaint` kept byte-compatible for stock RapidRAW

Config env vars: `GATEWAY_TOKEN`, `GATEWAY_TOOLS_DIR` (default `~/comfy`),
`GATEWAY_CACHE_MAX_GB` (20), `GATEWAY_RESULT_TTL_HOURS` (24),
`GATEWAY_JOB_TIMEOUT_S` (600), `GATEWAY_DB_PATH`,
`GATEWAY_COMFY_VENV_PY`, `GATEWAY_RAWTOOLS_PY`.

Known v1 limits: `ev_stack` param accepted but the multi-EV `--det-images`
wiring into mask_c2f is not connected yet; labels parsing is best-effort.

Deploy on inferno:
    rsync -a --exclude .venv --exclude cache ./ inferno:~/rr-ai-gateway/
    ssh inferno 'cd ~/rr-ai-gateway && python -m venv .venv && .venv/bin/pip install -r requirements.txt'
    scp deploy/rr-ai-gateway.service inferno:~/.config/systemd/user/
    ssh inferno 'systemctl --user daemon-reload && systemctl --user enable --now rr-ai-gateway'
```

- [ ] **Step 3: Full suite + commit**

Run: `.venv/bin/pytest -v` — Expected: all PASS

```bash
git add deploy README.md
git commit -m "feat(gateway): systemd unit + gateway docs"
```

---

## Self-review checklist (run after all tasks)

1. Spec coverage: capabilities registry ✓ (T4), queue+priorities+journal+recovery ✓ (T5), source cache+sidecars+eviction+410 ✓ (T3/T6), common API ✓ (T6), auth ✓ (T6), legacy adapters ✓ (T7), mask modes ✓ (T9/T10), raw develop + alignment ✓ (T9), rrdata signals ✓ (T9, tags→agentic only; EV wiring documented as v1 limit), health extension ✓ (T6), deployment ✓ (T12), integration ✓ (T11).
2. `pytest -v` green locally; `pytest -m integration` green on inferno.
3. Legacy compatibility manually verified with stock RapidRAW pointed at the gateway (settings → Self-Hosted backend → inferno:5000) before switching off the old connector instance.
