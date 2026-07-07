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
