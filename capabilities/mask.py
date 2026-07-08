"""Mask capability: high-quality AI masking by shelling into the proven tools
living in GATEWAY_TOOLS_DIR (mask_hq.py, mask_agentic.py, mask_c2f.py,
raw_develop.py) plus the bundled mask_tools/mask_points.py."""
import base64
import io
import json
import logging
import re
import shutil
import subprocess
import time
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw
from gateway.registry import Capability, register

logger = logging.getLogger("Mask")

PARAMS_SCHEMA = {
    "type": "object",
    "required": ["mode"],
    "properties": {
        "mode": {"enum": ["prompt", "points", "paint", "preset", "box"]},
        "query": {"type": "string"},
        "points": {"type": "array",
                   "items": {"type": "array", "minItems": 3, "maxItems": 3,
                             "items": {"type": "number"}}},
        "box": {"type": "array", "minItems": 4, "maxItems": 4,
                "items": {"type": "number"}},
        "roi_mask_b64": {"type": "string"},
        "preset": {"enum": ["subject", "sky", "foreground"]},
        "agentic": {"type": "boolean", "default": False},
        "backend": {"enum": ["sam2", "sam3"], "default": "sam2"},
        "sam3_multirep": {"type": "boolean", "default": False},
        "matte": {"type": "boolean", "default": True},
        "carve": {"type": "boolean", "default": False},
        "agentic_mode": {"enum": ["precise", "removal"], "default": "precise"},
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
        {"if": {"properties": {"mode": {"const": "box"}}},
         "then": {"required": ["mode", "box"]}},
    ],
}

# Real tool stdout (verified on inferno against mask_hq.py / grounded_sam.py):
#   "   detected: the main subject(0.61)"
#   "   detected: person(0.91), dog(0.55)"
#   "[understanding] detected 3 objects: person(0.91), dog(0.55), cat(0.30)"
#   "   detected: none"
# i.e. a "detected[...]:" prefix (optionally "N objects") followed by zero or
# more comma-separated "label(score)" pairs, or the literal "none".
#
# Two other tools use different formats:
#   mask_points.py:  "detected: point-selection 0.87"  -- bare "label score",
#                     no parens, single pair, no comma-separation.
#   mask_c2f.py (sam3 path): "[sam3] 2 instance(s) from concepts "
#                     "['the main subject', 'bicycle']" -- a python-repr-ish
#                     list of quoted concept names, no scores on that line.
LABEL_LINE_RE = re.compile(r"detected\b[^:]*:\s*(.+)", re.IGNORECASE)
LABEL_ITEM_RE = re.compile(r"([a-zA-Z][\w \-]*?)\(\s*[\d.]+\s*\)")
LABEL_BARE_ITEM_RE = re.compile(r"^([a-zA-Z][\w \-]*?)\s+[\d.]+\s*$")
SAM3_CONCEPTS_LINE_RE = re.compile(r"from concepts\s*\[(.*)\]", re.IGNORECASE)
SAM3_CONCEPT_ITEM_RE = re.compile(r"""['"]([^'"]+)['"]""")
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
    """Best-effort extraction of labels from tool stdout: 'detected:
    label(score), ...' (mask_hq/grounded_sam), 'detected: label score' (bare,
    single-pair -- mask_points.py), and "from concepts ['a', 'b']" (sam3
    concept segmentation via mask_c2f.py). Never raises; an empty list is a
    valid (if uninformative) result."""
    labels = []
    try:
        for line in stdout.splitlines():
            m = LABEL_LINE_RE.search(line)
            if m:
                rest = m.group(1)
                items = LABEL_ITEM_RE.findall(rest)
                if items:
                    labels.extend(label.strip() for label in items)
                else:
                    bare = LABEL_BARE_ITEM_RE.match(rest.strip())
                    if bare:
                        labels.append(bare.group(1).strip())
                continue
            m = SAM3_CONCEPTS_LINE_RE.search(line)
            if m:
                labels.extend(SAM3_CONCEPT_ITEM_RE.findall(m.group(1)))
    except Exception:
        return labels
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


def parse_gpu_free(nvidia_smi_output: str) -> list[tuple[int, int]]:
    """Parse `nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits`
    output into [(index, free_mb)]. Tolerant: skip malformed lines."""
    out = []
    for line in nvidia_smi_output.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 2:
            continue
        try:
            out.append((int(parts[0]), int(parts[1])))
        except ValueError:
            continue
    return out


def pick_cuda_device(min_free_mb: int = 3000) -> "str | None":
    """Return the index (as str) of the GPU with the most free VRAM, or None if
    nvidia-smi is unavailable / no GPU has at least min_free_mb free (caller then
    leaves the environment untouched)."""
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5)
        if proc.returncode != 0:
            return None
        gpus = parse_gpu_free(proc.stdout)
    except Exception:
        return None
    if not gpus:
        return None
    best_idx, best_free = max(gpus, key=lambda g: g[1])
    if best_free < min_free_mb:
        return None
    return str(best_idx)


async def _develop_raw(ctx, env: "dict | None" = None) -> Path:
    """raw_develop.py -> pick the 0EV frame as the working image."""
    outdir = ctx.workdir / "developed"
    cmd = [ctx.settings.GATEWAY_RAWTOOLS_PY, _tool(ctx.settings, "raw_develop.py"),
           str(ctx.source.path), str(outdir)]
    code, out, err = await ctx.run_tool(cmd, env=env)
    if code != 0:
        raise RuntimeError(f"raw_develop failed: {err[-800:]}")
    return pick_base_frame(outdir)


def _depth_cache_path(ctx) -> "Path | None":
    """Per-source depth-map cache sidecar path: depth depends ONLY on the
    source image bytes, never on job params (box/points/query), so a nudged
    job re-using the same ctx.source can skip regenerating it entirely.
    Returns None when ctx.source is unavailable (defensive; handle() already
    requires ctx.source, but keeps this helper safe to call standalone)."""
    if ctx.source is None:
        return None
    return Path(str(ctx.source.path) + ".depth.png")


async def _make_depth(ctx, image_path: Path, env: "dict | None" = None) -> "Path | None":
    """Run the bundled Depth Pro tool to produce a near=bright depth map for
    the carve path (biggest lattice-quality win per the bench: BMX spokes fill
    0.878->0.486). Best-effort: on failure, log-and-continue -- carve should
    still run with its own texture fallback rather than failing the job.

    Cached beside the source file (<source path>.depth.png): depth depends
    only on the source image, so a second carve job on the same ctx.source
    (e.g. a nudged box/points retry) reuses it instead of paying the ~10-15s
    model load + inference again."""
    cache_path = _depth_cache_path(ctx)
    if cache_path is not None and cache_path.exists():
        return cache_path

    py = ctx.settings.GATEWAY_COMFY_VENV_PY
    depth_path = ctx.workdir / "depth.png"
    cmd = [py, _bundled("make_depth.py"), str(image_path),
           "--tools-dir", ctx.settings.GATEWAY_TOOLS_DIR, "--out", str(depth_path)]
    code, out, err = await ctx.run_tool(cmd, env=env)
    if code != 0:
        logger.warning("make_depth failed, continuing without depth map: %s", err[-400:])
        return None

    if cache_path is not None:
        try:
            shutil.copyfile(depth_path, cache_path)
        except OSError:
            pass  # best-effort: cache miss next time is fine, job must not fail

    return depth_path


async def classify_tool_failure(stderr: str) -> str:
    """Classify a failed mask-tool subprocess's stderr into a user-facing
    error kind. OOM is checked first since it is unambiguous from the text
    alone; everything else is disambiguated with an actual health probe
    rather than string-matching "ComfyUI"/"8188" -- those strings appear in
    every tool traceback via the interpreter path
    (~/comfy/ComfyUI/.venv/...), which previously caused GPU OOMs and other
    unrelated failures to be misreported as "ComfyUI not running"."""
    if "OutOfMemoryError" in stderr or "out of memory" in stderr.lower():
        return "gpu_oom"
    from engine import ComfyClient
    if not await ComfyClient.check_health():
        return "comfyui_down"
    return "tool_error"


def box_to_roi_png(box: "list | tuple", width: int, height: int) -> bytes:
    """Render a box selection [x0, y0, x1, y1] as a coarse ROI mask: a white
    filled rectangle on black, uint8 L-mode PNG at the source dims. The box
    is clamped to the image bounds so out-of-range client coordinates never
    raise or wrap. This is how box selections become a REGION HINT for the
    full SAM3/agentic pipeline (mask_c2f.py / mask_agentic.py --roi), instead
    of a bare SAM2 box prompt."""
    x0, y0, x1, y1 = box
    x0 = max(0, min(int(round(x0)), width))
    x1 = max(0, min(int(round(x1)), width))
    y0 = max(0, min(int(round(y0)), height))
    y1 = max(0, min(int(round(y1)), height))
    x0, x1 = sorted((x0, x1))
    y0, y1 = sorted((y0, y1))
    img = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(img)
    if x1 > x0 and y1 > y0:
        draw.rectangle([x0, y0, x1 - 1, y1 - 1], fill=255)
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def _apply_sam3_carve_flags(cmd: list, carve: bool, depth_used: bool, depth_path) -> None:
    """Append --sam3-carve/--depth-map to a mask_c2f.py command, shared by
    the direct sam3 prompt branch and the sam3-backed preset branch."""
    if carve:
        cmd.append("--sam3-carve")
    if depth_used:
        cmd += ["--depth-map", str(depth_path)]


PRESET_QUERIES = {
    "subject": "the main subject.",
    "sky": "sky.",
    "foreground": "foreground.",
}


def _build_agentic_cmd(py, settings, image_path, out_path, target, backend,
                        carve, depth_used, depth_path, agentic_mode, tool_env,
                        roi_path=None):
    """Shared mask_agentic.py invocation plumbing, used by both prompt+agentic
    and box+agentic (the box case additionally restricts the search via
    --roi). Returns (cmd, tool_env) since agentic runs need the LLM/VLM env
    merged in on top of any GPU pin."""
    cmd = [py, _tool(settings, "mask_agentic.py"), str(image_path),
           "--target", target, "--backend", backend, "--out", str(out_path)]
    if roi_path is not None:
        cmd += ["--roi", str(roi_path)]
    if carve:
        cmd.append("--carve")
    if depth_used:
        cmd += ["--depth-map", str(depth_path)]
    cmd += ["--mode", agentic_mode]
    agentic_env = {
        "LLM_URL": settings.GATEWAY_LLM_URL,
        "INTENT_LLM": settings.GATEWAY_INTENT_LLM,
        "VLM_URL": settings.GATEWAY_VLM_URL,
        "VLM_MODEL": settings.GATEWAY_VLM_MODEL,
    }
    tool_env = {**agentic_env, **(tool_env or {})}
    return cmd, tool_env


async def handle(ctx) -> dict:
    t0 = time.perf_counter()
    if ctx.source is None:
        raise FileNotFoundError("source not available")
    p = ctx.params
    settings = ctx.settings
    py = settings.GATEWAY_COMFY_VENV_PY
    alignment = "exact"

    mode = p["mode"]
    backend = p.get("backend", "sam2")
    is_multirep = (mode in ("prompt", "box") and backend == "sam3"
                   and p.get("sam3_multirep"))

    # Dynamic per-job GPU selection: script-side tools (SAM2/SAM3/GroundingDINO/
    # ViTMatte) default to cuda:0, which piles onto whichever card ComfyUI is
    # using. Pin the freest GPU per job so masking work spreads across both
    # cards. sam3_multirep's --sam3-parallel needs BOTH GPUs visible, so it is
    # excluded and inherits the full environment (env=None).
    tool_env = None
    if not is_multirep:
        gpu_idx = pick_cuda_device()
        if gpu_idx is not None:
            tool_env = {"CUDA_VISIBLE_DEVICES": gpu_idx}

    image_path = ctx.source.path
    if ctx.source.kind == "raw":
        image_path = await _develop_raw(ctx, env=tool_env)
        alignment = "best_effort"

    out_path = ctx.workdir / "mask.png"

    # Depth-guided carve is the biggest lattice-quality win (see-through
    # subjects like BMX spokes: fill 0.878->0.486 in the bench). Generate the
    # depth map once, up front, for either carve-eligible branch below.
    depth_used = False
    depth_path = None
    carve = bool(p.get("carve"))
    if carve:
        depth_path = await _make_depth(ctx, image_path, env=tool_env)
        depth_used = depth_path is not None

    if mode == "prompt" and p.get("agentic"):
        target = p["query"]
        rr = ctx.source.rrdata or {}
        tags = rr.get("tags") or []
        if tags:
            target = f"{target} (photo context: {', '.join(tags)})"
        cmd, tool_env = _build_agentic_cmd(
            py, settings, image_path, out_path, target, backend,
            carve, depth_used, depth_path, p.get("agentic_mode", "precise"), tool_env)
    elif mode == "prompt" and backend == "sam3":
        cmd = [py, _tool(settings, "mask_c2f.py"), str(image_path),
               "--query", p["query"], "--backend", "sam3",
               "--birefnet-mode", "auto", "--out", str(out_path)]
        if p.get("sam3_multirep"):
            cmd += ["--sam3-multirep", "--sam3-parallel"]
        _apply_sam3_carve_flags(cmd, carve, depth_used, depth_path)
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
    elif mode == "box" and p.get("agentic"):
        # A box is a REGION HINT, not a bare SAM2 box prompt: render it as a
        # coarse ROI mask and route through the same full agentic pipeline as
        # prompt+agentic, restricting the search to the box's interior.
        roi_path = ctx.workdir / "roi.png"
        roi_path.write_bytes(box_to_roi_png(p["box"], ctx.source.width, ctx.source.height))
        target = p.get("query") or "the main subject."
        cmd, tool_env = _build_agentic_cmd(
            py, settings, image_path, out_path, target, backend,
            carve, depth_used, depth_path, p.get("agentic_mode", "precise"), tool_env,
            roi_path=roi_path)
    elif mode == "box" and backend == "sam3":
        # Same region-hint treatment for the non-agentic sam3 path: mask_c2f.py
        # --roi restricts SAM3's search to the box instead of a bare SAM2 box
        # prompt via mask_points.py.
        roi_path = ctx.workdir / "roi.png"
        roi_path.write_bytes(box_to_roi_png(p["box"], ctx.source.width, ctx.source.height))
        cmd = [py, _tool(settings, "mask_c2f.py"), str(image_path),
               "--query", p.get("query") or "the main subject.",
               "--roi", str(roi_path), "--backend", "sam3",
               "--birefnet-mode", "auto", "--out", str(out_path)]
        if p.get("sam3_multirep"):
            cmd += ["--sam3-multirep", "--sam3-parallel"]
        _apply_sam3_carve_flags(cmd, carve, depth_used, depth_path)
    elif mode == "box":
        # sam2 (default): unchanged, bare SAM2 box prompt via mask_points.py.
        # Same pixel-space caveats as "points" (raw sources: best_effort).
        cmd = [py, _bundled("mask_points.py"), str(image_path),
               "--box", ",".join(str(v) for v in p["box"]), "--backend", backend,
               "--tools-dir", settings.GATEWAY_TOOLS_DIR, "--out", str(out_path)]
        if p.get("points"):
            cmd += ["--points", json.dumps(p["points"])]
    elif mode == "paint":
        roi_path = ctx.workdir / "roi.png"
        roi_path.write_bytes(base64.b64decode(p["roi_mask_b64"]))
        cmd = [py, _tool(settings, "mask_c2f.py"), str(image_path),
               "--query", p.get("query") or "the main subject.",
               "--roi", str(roi_path), "--out", str(out_path)]
    else:  # preset
        preset = p["preset"]
        query = PRESET_QUERIES[preset]
        if backend == "sam3":
            # mask_hq.py (BiRefNet+ViTMatte) OOMs on smaller GPUs; route
            # sam3-backed presets through the lighter mask_c2f.py instead,
            # reusing the same carve/depth plumbing as the direct sam3 branch.
            cmd = [py, _tool(settings, "mask_c2f.py"), str(image_path),
                   "--query", query, "--backend", "sam3",
                   "--birefnet-mode", "auto", "--out", str(out_path)]
            _apply_sam3_carve_flags(cmd, carve, depth_used, depth_path)
        elif preset == "subject":
            cmd = [py, _tool(settings, "mask_hq.py"), str(image_path),
                   "--query", query, "--main-subject", "--out", str(out_path)]
        elif preset == "sky":
            cmd = [py, _tool(settings, "mask_hq.py"), str(image_path),
                   "--query", query, "--out", str(out_path)]
        else:  # foreground
            cmd = [py, _tool(settings, "mask_hq.py"), str(image_path),
                   "--no-sam", "--query", query, "--out", str(out_path)]

    code, stdout, stderr = await ctx.run_tool(cmd, env=tool_env)
    if code != 0:
        kind = await classify_tool_failure(stderr)
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
        "depth_used": depth_used,
        "timings": {"total_s": round(time.perf_counter() - t0, 2)},
    }


register(Capability(
    id="mask", title="AI masking (GroundedSAM/SAM2/BiRefNet/ViTMatte)",
    params_schema=PARAMS_SCHEMA, handler=handle,
    modes=["prompt", "points", "paint", "preset", "box"],
    presets=["subject", "sky", "foreground"]))
