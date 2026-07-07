import base64
import io
import json
import numpy as np
import pytest
from pathlib import Path
from PIL import Image
from engine import Settings
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
            # cmd = [py, raw_develop.py, RAW.ARW, OUTDIR, ...] per the tool's
            # documented CLI (RAW.ARW OUTDIR) -- outdir is the 4th element.
            # Filenames mirror raw_develop.py's REAL output (verified on the
            # GPU host): {base}_EV{ev:+g}.jpg, {base}_exif.json, and a
            # {base}_stack.jpg montage that must never be picked as the
            # working frame. No PNGs are ever produced.
            outdir = Path(cmd[3])
            outdir.mkdir(parents=True, exist_ok=True)
            for ev in ("-2", "+0", "+2", "+4"):
                Image.new("RGB", (8, 6)).save(outdir / f"src_EV{ev}.jpg")
            Image.new("RGB", (8, 6)).save(outdir / "src_stack.jpg")
            (outdir / "src_exif.json").write_text("{}")
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
    # The picker must select the EV closest to 0 (src_EV+0.jpg), never the
    # _stack montage or an off-zero EV frame.
    assert rec.calls[1][2].endswith("src_EV+0.jpg")
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


def test_pick_base_frame_prefers_ev0_and_ignores_montage(tmp_path):
    for name in ("a_EV-2.jpg", "a_EV+0.jpg", "a_EV+4.jpg", "a_stack.jpg", "a_exif.json"):
        (tmp_path / name).write_bytes(b"x")
    assert M.pick_base_frame(tmp_path).name == "a_EV+0.jpg"


def test_pick_base_frame_closest_to_zero_when_no_exact(tmp_path):
    for name in ("a_EV-2.jpg", "a_EV+2.jpg", "a_EV+4.jpg"):
        (tmp_path / name).write_bytes(b"x")
    assert M.pick_base_frame(tmp_path).name in ("a_EV-2.jpg", "a_EV+2.jpg")


def test_pick_base_frame_empty_raises(tmp_path):
    (tmp_path / "a_stack.jpg").write_bytes(b"x")
    import pytest as _pytest
    with _pytest.raises(RuntimeError):
        M.pick_base_frame(tmp_path)


@pytest.mark.asyncio
async def test_tool_failure_classified_comfyui_down(tmp_path):
    ctx = make_ctx(tmp_path, {"mode": "prompt", "query": "x."})
    async def failing(cmd, timeout=None):
        return 1, "", "ConnectionRefused connecting to 127.0.0.1:8188"
    ctx.run_tool = failing
    with pytest.raises(RuntimeError) as ei:
        await M.handle(ctx)
    assert getattr(ei.value, "kind", None) == "comfyui_down"


@pytest.mark.asyncio
async def test_tool_failure_classified_tool_error(tmp_path):
    ctx = make_ctx(tmp_path, {"mode": "prompt", "query": "x."})
    async def failing(cmd, timeout=None):
        return 2, "", "torch OOM"
    ctx.run_tool = failing
    with pytest.raises(RuntimeError) as ei:
        await M.handle(ctx)
    assert getattr(ei.value, "kind", None) == "tool_error"


@pytest.mark.asyncio
async def test_nonraw_mask_upscaled_to_source_dims(tmp_path):
    ctx = make_ctx(tmp_path, {"mode": "prompt", "query": "x."}, dims=(16, 12))
    rec = ToolRecorder()  # writes an 8x6 mask regardless
    ctx.run_tool = rec
    result = await M.handle(ctx)
    assert (result["width"], result["height"]) == (16, 12)
