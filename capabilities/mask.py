"""Mask capability: high-quality AI masking by shelling into the proven tools
living in GATEWAY_TOOLS_DIR (mask_hq.py, mask_agentic.py, mask_c2f.py,
raw_develop.py) plus the bundled mask_tools/mask_points.py."""
import base64
import io
import json
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

# Real tool stdout (verified on inferno against mask_hq.py / grounded_sam.py):
#   "   detected: the main subject(0.61)"
#   "   detected: person(0.91), dog(0.55)"
#   "[understanding] detected 3 objects: person(0.91), dog(0.55), cat(0.30)"
#   "   detected: none"
# i.e. a "detected[...]:" prefix (optionally "N objects") followed by zero or
# more comma-separated "label(score)" pairs, or the literal "none".
LABEL_LINE_RE = re.compile(r"detected\b[^:]*:\s*(.+)", re.IGNORECASE)
LABEL_ITEM_RE = re.compile(r"([a-zA-Z][\w \-]*?)\(\s*[\d.]+\s*\)")
EV_FRAME_RE = re.compile(r"_EV([+-]?[0-9.]+)\.(jpg|jpeg|png)$", re.IGNORECASE)


def pick_base_frame(outdir: Path) -> Path:
    """Pick the EV closest to 0 from raw_develop.py's output
    ({base}_EV{ev:+g}.jpg), ignoring the _stack montage."""
    best = None
    for f in sorted(outdir.iterdir()):
        m = EV_FRAME_RE.search(f.name)
        if not m:
            continue
        ev = abs(float(m.group(1)))
        if best is None or ev < best[0]:
            best = (ev, f)
    if best is None:
        raise RuntimeError(f"raw_develop produced no EV frames in {outdir}")
    return best[1]


def parse_labels(stdout: str) -> list[str]:
    """Best-effort extraction of 'detected: label(score), ...' lines from tool
    stdout. Never raises; an empty list is a valid (if uninformative) result."""
    labels = []
    for line in stdout.splitlines():
        m = LABEL_LINE_RE.search(line)
        if not m:
            continue
        labels.extend(label.strip() for label in LABEL_ITEM_RE.findall(m.group(1)))
    return labels


def reconcile_mask(mask: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    """Center-crop or zero-pad a mask to the client's stated dimensions (raw mode)."""
    h, w = mask.shape
    if w > target_w:
        x0 = (w - target_w) // 2
        mask = mask[:, x0:x0 + target_w]
    if h > target_h:
        y0 = (h - target_h) // 2
        mask = mask[y0:y0 + target_h, :]
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
    return pick_base_frame(outdir)


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
    elif mode == "points":
        # Points arrive in client (RapidRAW-decode) pixel space; for raw
        # sources the developed frame may differ by a few pixels per edge,
        # so edge clicks can drift by the half-margin -- spec-sanctioned
        # best_effort; the TIFF payload path gives exact coords instead.
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
