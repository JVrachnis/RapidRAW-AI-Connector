import json
from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from . import registry
from .db import loads
from .registry import ParamsInvalid
from .store import SourceEvicted

router = APIRouter()

@router.get("/capabilities")
async def capabilities(request: Request):
    return registry.describe(request.app.state.capabilities)

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

def _find_cached_job(request: Request, capability: str, source_id: str, params: dict):
    """Look for an existing completed job with the same capability+source_id
    and semantically-equal params (nudge-cache Change 2). Exact retries --
    e.g. a client re-polling after a dropped response, or genuinely repeating
    the same edit -- get an instant result instead of paying for the whole
    pipeline again. Rows are pruned by the existing result TTL, so staleness
    is bounded; source_id is content-addressed, so the image itself cannot
    have changed underneath a hit.

    Compares parsed params (json.loads), not raw strings: dict key order is
    not guaranteed to match between two logically-identical requests."""
    db = request.app.state.queue.db
    rows = db.query(
        "SELECT id, params FROM jobs WHERE capability=? AND source_id=? AND status='done'"
        " ORDER BY created DESC",
        (capability, source_id))
    for row in rows:
        if loads(row["params"]) == params:
            return row["id"]
    return None


@router.post("/jobs/{capability}", status_code=202)
async def submit_job(capability: str, request: Request, body: dict):
    caps = request.app.state.capabilities
    if capability not in caps:
        raise HTTPException(404, f"unknown capability {capability!r}")
    source_id = body.get("source_id")
    params = body.get("params", {})
    priority = body.get("priority", "interactive")
    no_cache = body.get("no_cache", False)
    if priority not in ("interactive", "batch"):
        raise HTTPException(422, "priority must be 'interactive' or 'batch'")
    try:
        registry.validate_params(capability, params, caps)
    except ParamsInvalid as e:
        raise HTTPException(422, str(e))
    try:
        if request.app.state.store.get(source_id) is None:
            raise HTTPException(404, "unknown source_id")
    except SourceEvicted:
        raise HTTPException(410, "source evicted from cache; re-upload")
    if not no_cache:
        cached_id = _find_cached_job(request, capability, source_id, params)
        if cached_id is not None:
            return {"job_id": cached_id, "status": "done", "cached": True}
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
