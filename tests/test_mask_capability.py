import base64
import io
import json
import jsonschema
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
    async def __call__(self, cmd, timeout=None, env=None):
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
        return 0, "   detected: person(0.91)\n", ""

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


def test_schema_accepts_box_mode_with_box():
    jsonschema.validate({"mode": "box", "box": [1, 2, 3, 4]}, M.PARAMS_SCHEMA)


def test_schema_rejects_box_mode_without_box():
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate({"mode": "box"}, M.PARAMS_SCHEMA)


def test_schema_rejects_box_with_wrong_length():
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate({"mode": "box", "box": [1, 2, 3]}, M.PARAMS_SCHEMA)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate({"mode": "box", "box": [1, 2, 3, 4, 5]}, M.PARAMS_SCHEMA)


@pytest.mark.asyncio
async def test_box_mode_calls_points_tool_with_box_only(tmp_path):
    ctx = make_ctx(tmp_path, {"mode": "box", "box": [10, 20, 110, 220]})
    rec = ToolRecorder(); ctx.run_tool = rec
    await M.handle(ctx)
    cmd = rec.calls[0]
    assert cmd[1].endswith("mask_points.py")
    assert cmd[cmd.index("--box") + 1] == "10,20,110,220"
    assert "--points" not in cmd


@pytest.mark.asyncio
async def test_box_mode_with_points_passes_both(tmp_path):
    ctx = make_ctx(tmp_path, {"mode": "box", "box": [10, 20, 110, 220],
                              "points": [[50, 50, 1]]})
    rec = ToolRecorder(); ctx.run_tool = rec
    await M.handle(ctx)
    cmd = rec.calls[0]
    assert cmd[1].endswith("mask_points.py")
    assert cmd[cmd.index("--box") + 1] == "10,20,110,220"
    pts = json.loads(cmd[cmd.index("--points") + 1])
    assert pts == [[50, 50, 1]]

@pytest.mark.asyncio
async def test_preset_subject(tmp_path):
    ctx = make_ctx(tmp_path, {"mode": "preset", "preset": "subject"})
    rec = ToolRecorder(); ctx.run_tool = rec
    await M.handle(ctx)
    assert rec.calls[0][1].endswith("mask_hq.py")


@pytest.mark.asyncio
async def test_preset_default_backend_sam2_still_mask_hq(tmp_path):
    # backend defaults to sam2 -- presets keep routing to mask_hq (BiRefNet+
    # ViTMatte) unchanged, since that's the only regression risk this fix
    # must avoid.
    for preset, query in (("subject", "the main subject."), ("sky", "sky."),
                          ("foreground", "foreground.")):
        sub = tmp_path / preset
        sub.mkdir()
        ctx = make_ctx(sub, {"mode": "preset", "preset": preset})
        rec = ToolRecorder(); ctx.run_tool = rec
        await M.handle(ctx)
        cmd = rec.calls[0]
        assert cmd[1].endswith("mask_hq.py")
        assert cmd[cmd.index("--query") + 1] == query


@pytest.mark.asyncio
async def test_preset_sam3_backend_routes_to_mask_c2f(tmp_path):
    # presets ALWAYS routed to mask_hq (BiRefNet+ViTMatte), which OOMs on
    # smaller boxes. When backend == "sam3", route through mask_c2f.py
    # instead with canned queries, so presets actually work locally.
    cases = [("subject", "the main subject."), ("sky", "sky."),
             ("foreground", "foreground.")]
    for preset, expected_query in cases:
        sub = tmp_path / preset
        sub.mkdir()
        ctx = make_ctx(sub, {"mode": "preset", "preset": preset, "backend": "sam3"})
        rec = ToolRecorder(); ctx.run_tool = rec
        await M.handle(ctx)
        cmd = rec.calls[0]
        assert cmd[1].endswith("mask_c2f.py"), f"{preset}: {cmd}"
        assert cmd[cmd.index("--query") + 1] == expected_query
        assert cmd[cmd.index("--backend") + 1] == "sam3"


@pytest.mark.asyncio
async def test_preset_sam3_carve_reuses_shared_plumbing(tmp_path, monkeypatch):
    ctx = make_ctx(tmp_path, {"mode": "preset", "preset": "subject", "backend": "sam3",
                              "carve": True})
    rec = ToolRecorder(); ctx.run_tool = rec
    monkeypatch.setattr(M, "pick_cuda_device", lambda min_free_mb=3000: "0")
    result = await M.handle(ctx)
    assert rec.calls[0][1].endswith("make_depth.py")
    cmd = rec.calls[1]
    assert cmd[1].endswith("mask_c2f.py")
    assert "--sam3-carve" in cmd and "--depth-map" in cmd
    assert result["depth_used"] is True


@pytest.mark.asyncio
async def test_preset_sam2_backend_ignores_carve_like_before(tmp_path, monkeypatch):
    # sam2-backed presets (mask_hq.py) don't take --sam3-carve/--depth-map;
    # carve is a sam3-only concept here, matching the pre-existing direct
    # sam3 vs mask_hq split.
    ctx = make_ctx(tmp_path, {"mode": "preset", "preset": "sky", "carve": True})
    rec = ToolRecorder(); ctx.run_tool = rec
    monkeypatch.setattr(M, "pick_cuda_device", lambda min_free_mb=3000: "0")
    await M.handle(ctx)
    cmd = rec.calls[-1]
    assert cmd[1].endswith("mask_hq.py")
    assert "--sam3-carve" not in cmd and "--depth-map" not in cmd

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
    # Real stdout format (verified on inferno against mask_hq.py / grounded_sam.py):
    # "   detected: label(score)[, label(score)...]" -- score in parens, no space,
    # comma-separated for multiple detections, possibly prefixed with a count
    # ("[understanding] detected 3 objects: ...") and "none" when nothing matched.
    assert M.parse_labels("   detected: the main subject(0.61)\n") == ["the main subject"]
    assert M.parse_labels("   detected: person(0.91), dog(0.55)\n") == ["person", "dog"]
    assert M.parse_labels(
        "[understanding] detected 3 objects: person(0.91), dog(0.55), cat(0.30)\n"
    ) == ["person", "dog", "cat"]
    assert M.parse_labels("   detected: none\n") == []
    assert M.parse_labels("no matches here") == []

def test_parse_labels_sam3_concepts():
    out = "[sam3] 'the main subject': 1 instance(s) scores [0.52]\n[sam3] 2 instance(s) from concepts ['the main subject', 'bicycle']\n"
    assert M.parse_labels(out) == ["the main subject", "bicycle"]

def test_parse_labels_point_selection():
    assert M.parse_labels("detected: point-selection 0.87") == ["point-selection"]

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
async def test_tool_failure_classified_comfyui_down(tmp_path, monkeypatch):
    # Real ComfyUI-down stderr still contains "ComfyUI"/"8188" via the venv
    # interpreter path, so classification must not string-match on those --
    # it must actually probe ComfyClient.check_health().
    ctx = make_ctx(tmp_path, {"mode": "prompt", "query": "x."})
    async def failing(cmd, timeout=None, env=None):
        return 1, "", "ConnectionRefused connecting to 127.0.0.1:8188"
    ctx.run_tool = failing
    from engine import ComfyClient
    async def unhealthy():
        return False
    monkeypatch.setattr(ComfyClient, "check_health", unhealthy)
    with pytest.raises(RuntimeError) as ei:
        await M.handle(ctx)
    assert getattr(ei.value, "kind", None) == "comfyui_down"


@pytest.mark.asyncio
async def test_tool_failure_classified_tool_error_when_comfy_healthy(tmp_path, monkeypatch):
    # Generic failure + healthy ComfyUI -> tool_error, even though the
    # traceback contains "ComfyUI" (interpreter path) -- the false-positive
    # this fix targets.
    ctx = make_ctx(tmp_path, {"mode": "prompt", "query": "x."})
    async def failing(cmd, timeout=None, env=None):
        return 2, "", ("Traceback (most recent call last):\n"
                        '  File "/home/user/comfy/ComfyUI/.venv/lib/site-packages/torch/x.py"\n'
                        "RuntimeError: some tool error")
    ctx.run_tool = failing
    from engine import ComfyClient
    async def healthy():
        return True
    monkeypatch.setattr(ComfyClient, "check_health", healthy)
    with pytest.raises(RuntimeError) as ei:
        await M.handle(ctx)
    assert getattr(ei.value, "kind", None) == "tool_error"


@pytest.mark.asyncio
async def test_tool_failure_oom_classified_gpu_oom_even_with_comfyui_strings(tmp_path, monkeypatch):
    ctx = make_ctx(tmp_path, {"mode": "prompt", "query": "x."})
    async def failing(cmd, timeout=None, env=None):
        return 1, "", ('  File "/home/user/comfy/ComfyUI/.venv/lib/site-packages/torch/x.py"\n'
                        "torch.cuda.OutOfMemoryError: CUDA out of memory")
    ctx.run_tool = failing
    from engine import ComfyClient
    async def should_not_be_called():
        raise AssertionError("check_health should not be called when OOM is detected first")
    monkeypatch.setattr(ComfyClient, "check_health", should_not_be_called)
    with pytest.raises(RuntimeError) as ei:
        await M.handle(ctx)
    assert getattr(ei.value, "kind", None) == "gpu_oom"


@pytest.mark.asyncio
async def test_tool_failure_oom_lowercase_message_classified_gpu_oom(tmp_path, monkeypatch):
    ctx = make_ctx(tmp_path, {"mode": "prompt", "query": "x."})
    async def failing(cmd, timeout=None, env=None):
        return 1, "", "RuntimeError: CUDA error: out of memory"
    ctx.run_tool = failing
    from engine import ComfyClient
    async def should_not_be_called():
        raise AssertionError("check_health should not be called when OOM is detected first")
    monkeypatch.setattr(ComfyClient, "check_health", should_not_be_called)
    with pytest.raises(RuntimeError) as ei:
        await M.handle(ctx)
    assert getattr(ei.value, "kind", None) == "gpu_oom"


def test_classify_tool_failure_directly():
    import asyncio
    from unittest.mock import patch, AsyncMock
    from capabilities.mask import classify_tool_failure

    async def run():
        assert await classify_tool_failure("torch.cuda.OutOfMemoryError: out of memory") == "gpu_oom"
        assert await classify_tool_failure("some out of memory issue") == "gpu_oom"
        with patch("engine.ComfyClient.check_health", new=AsyncMock(return_value=False)):
            assert await classify_tool_failure("generic stderr, ComfyUI mentioned") == "comfyui_down"
        with patch("engine.ComfyClient.check_health", new=AsyncMock(return_value=True)):
            assert await classify_tool_failure("generic stderr") == "tool_error"
    asyncio.run(run())


@pytest.mark.asyncio
async def test_nonraw_mask_upscaled_to_source_dims(tmp_path):
    ctx = make_ctx(tmp_path, {"mode": "prompt", "query": "x."}, dims=(16, 12))
    rec = ToolRecorder()  # writes an 8x6 mask regardless
    ctx.run_tool = rec
    result = await M.handle(ctx)
    assert (result["width"], result["height"]) == (16, 12)


@pytest.mark.asyncio
async def test_prompt_sam3_backend_routes_to_c2f(tmp_path):
    ctx = make_ctx(tmp_path, {"mode": "prompt", "query": "the dog.", "backend": "sam3"})
    rec = ToolRecorder(); ctx.run_tool = rec
    await M.handle(ctx)
    cmd = rec.calls[0]
    assert cmd[1].endswith("mask_c2f.py")
    assert cmd[cmd.index("--backend") + 1] == "sam3"
    assert "--sam3-multirep" not in cmd

@pytest.mark.asyncio
async def test_prompt_sam3_multirep_flags(tmp_path):
    ctx = make_ctx(tmp_path, {"mode": "prompt", "query": "the dog.", "backend": "sam3",
                              "sam3_multirep": True})
    rec = ToolRecorder(); ctx.run_tool = rec
    await M.handle(ctx)
    cmd = rec.calls[0]
    assert "--sam3-multirep" in cmd and "--sam3-parallel" in cmd

@pytest.mark.asyncio
async def test_prompt_default_backend_still_mask_hq(tmp_path):
    ctx = make_ctx(tmp_path, {"mode": "prompt", "query": "the dog."})
    rec = ToolRecorder(); ctx.run_tool = rec
    await M.handle(ctx)
    assert rec.calls[0][1].endswith("mask_hq.py")


def test_parse_gpu_free():
    out = "0, 14876\n1, 6512\n"
    assert M.parse_gpu_free(out) == [(0, 14876), (1, 6512)]
    assert M.parse_gpu_free("garbage\n0, 100\n") == [(0, 100)]
    assert M.parse_gpu_free("") == []


@pytest.mark.asyncio
async def test_tool_env_pins_freest_gpu(tmp_path, monkeypatch):
    ctx = make_ctx(tmp_path, {"mode": "prompt", "query": "x.", "backend": "sam3"})
    rec = ToolRecorder()
    envs = []
    async def rec_env(cmd, timeout=None, env=None):
        envs.append(env)
        return await rec(cmd, timeout)
    ctx.run_tool = rec_env
    monkeypatch.setattr(M, "pick_cuda_device", lambda min_free_mb=3000: "1")
    await M.handle(ctx)
    assert envs[-1] == {"CUDA_VISIBLE_DEVICES": "1"}


@pytest.mark.asyncio
async def test_multirep_keeps_both_gpus_visible(tmp_path, monkeypatch):
    ctx = make_ctx(tmp_path, {"mode": "prompt", "query": "x.", "backend": "sam3",
                              "sam3_multirep": True})
    rec = ToolRecorder()
    envs = []
    async def rec_env(cmd, timeout=None, env=None):
        envs.append(env)
        return await rec(cmd, timeout)
    ctx.run_tool = rec_env
    monkeypatch.setattr(M, "pick_cuda_device", lambda min_free_mb=3000: "1")
    await M.handle(ctx)
    assert envs[-1] is None


@pytest.mark.asyncio
async def test_agentic_carve_and_mode_flags(tmp_path, monkeypatch):
    ctx = make_ctx(tmp_path, {"mode": "prompt", "query": "the bike", "agentic": True,
                              "carve": True, "agentic_mode": "removal"})
    rec = ToolRecorder(); envs = []
    async def rec_env(cmd, timeout=None, env=None):
        envs.append(env); return await rec(cmd, timeout)
    ctx.run_tool = rec_env
    monkeypatch.setattr(M, "pick_cuda_device", lambda min_free_mb=3000: "0")
    await M.handle(ctx)
    # carve=True runs the bundled depth tool first, then mask_agentic.py.
    assert rec.calls[0][1].endswith("make_depth.py")
    cmd = rec.calls[1]
    assert cmd[1].endswith("mask_agentic.py")
    assert "--carve" in cmd
    assert cmd[cmd.index("--mode") + 1] == "removal"
    env = envs[-1]
    assert env["VLM_MODEL"] == "minicpm-v4.5:q4_K_M"
    assert env["CUDA_VISIBLE_DEVICES"] == "0"


@pytest.mark.asyncio
async def test_non_agentic_gets_no_llm_env(tmp_path, monkeypatch):
    ctx = make_ctx(tmp_path, {"mode": "prompt", "query": "x.", "backend": "sam3"})
    rec = ToolRecorder(); envs = []
    async def rec_env(cmd, timeout=None, env=None):
        envs.append(env); return await rec(cmd, timeout)
    ctx.run_tool = rec_env
    monkeypatch.setattr(M, "pick_cuda_device", lambda min_free_mb=3000: "1")
    await M.handle(ctx)
    assert envs[-1] == {"CUDA_VISIBLE_DEVICES": "1"}


@pytest.mark.asyncio
async def test_non_carve_job_does_not_run_make_depth(tmp_path, monkeypatch):
    ctx = make_ctx(tmp_path, {"mode": "prompt", "query": "bike", "agentic": True})
    rec = ToolRecorder(); ctx.run_tool = rec
    monkeypatch.setattr(M, "pick_cuda_device", lambda min_free_mb=3000: "0")
    await M.handle(ctx)
    assert not any(c[1].endswith("make_depth.py") for c in rec.calls)


@pytest.mark.asyncio
async def test_carve_generates_depth_and_passes_map_agentic(tmp_path, monkeypatch):
    ctx = make_ctx(tmp_path, {"mode": "prompt", "query": "bike", "agentic": True, "carve": True})
    rec = ToolRecorder(); ctx.run_tool = rec
    monkeypatch.setattr(M, "pick_cuda_device", lambda min_free_mb=3000: "0")
    result = await M.handle(ctx)
    assert rec.calls[0][1].endswith("make_depth.py")
    cmd = rec.calls[1]
    assert cmd[1].endswith("mask_agentic.py")
    assert "--depth-map" in cmd and "--carve" in cmd
    assert result["depth_used"] is True


@pytest.mark.asyncio
async def test_carve_direct_sam3_gets_carve_and_depth(tmp_path, monkeypatch):
    ctx = make_ctx(tmp_path, {"mode": "prompt", "query": "bike", "backend": "sam3", "carve": True})
    rec = ToolRecorder(); ctx.run_tool = rec
    monkeypatch.setattr(M, "pick_cuda_device", lambda min_free_mb=3000: "0")
    result = await M.handle(ctx)
    assert rec.calls[0][1].endswith("make_depth.py")
    cmd = rec.calls[1]
    assert cmd[1].endswith("mask_c2f.py")
    assert "--sam3-carve" in cmd and "--depth-map" in cmd
    assert result["depth_used"] is True


@pytest.mark.asyncio
async def test_depth_failure_degrades_gracefully(tmp_path, monkeypatch):
    ctx = make_ctx(tmp_path, {"mode": "prompt", "query": "bike", "agentic": True, "carve": True})
    rec = ToolRecorder()
    async def failing_depth(cmd, timeout=None, env=None):
        if cmd[1].endswith("make_depth.py"):
            return 1, "", "no cuda"
        return await rec(cmd, timeout)
    ctx.run_tool = failing_depth
    monkeypatch.setattr(M, "pick_cuda_device", lambda min_free_mb=3000: "0")
    result = await M.handle(ctx)
    cmd = rec.calls[0]
    assert cmd[1].endswith("mask_agentic.py")
    assert "--depth-map" not in cmd and "--carve" in cmd
    assert result["depth_used"] is False


@pytest.mark.asyncio
async def test_direct_sam3_carve_depth_failure_degrades_gracefully(tmp_path, monkeypatch):
    ctx = make_ctx(tmp_path, {"mode": "prompt", "query": "bike", "backend": "sam3", "carve": True})
    rec = ToolRecorder()
    async def failing_depth(cmd, timeout=None, env=None):
        if cmd[1].endswith("make_depth.py"):
            return 1, "", "no cuda"
        return await rec(cmd, timeout)
    ctx.run_tool = failing_depth
    monkeypatch.setattr(M, "pick_cuda_device", lambda min_free_mb=3000: "0")
    result = await M.handle(ctx)
    cmd = rec.calls[0]
    assert cmd[1].endswith("mask_c2f.py")
    assert "--sam3-carve" in cmd and "--depth-map" not in cmd
    assert result["depth_used"] is False
