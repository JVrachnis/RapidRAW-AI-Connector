# tools/ — vendored pipeline scripts

This directory is a snapshot of the photo edit/removal pipeline that
`capabilities/mask.py` and `engine.py` shell out to. It is copied from the
maintainer's personal tuned pipeline (developed and iterated at `~/comfy` on
the GPU host), **not** written from scratch for this repo. Treat it as
third-party-ish vendored code: it works, it is battle-tested against the
gateway's test suite and live benchmarks, but its style and structure predate
the gateway and don't necessarily follow the gateway's own conventions.

## Provenance

Snapshot taken 2026-07 from `~/comfy` on the GPU box that also runs ComfyUI
and Ollama. It includes fixes made during the same month:

- **`pelib/sam3.py`** — the "sky_L carve fix": `carve_seethrough`/
  `carve_continuity` previously could over-carve a bright sky behind a
  see-through subject (spokes/mesh) because the L-channel threshold used the
  same cutoff for sky and non-sky pixels. Fixed to gate sky pixels separately.
- **`pelib/sam3.py`** — the "chan-B rep" fix to `multirep_instances`: one of
  the multi-representation image variants used for 2-GPU union detection was
  missing a blue-channel-boosted representation, which under-detected
  blue-dominant subjects (e.g. some bikes/vehicles) in one of the two
  parallel passes.

Both fixes are already applied to the vendored `sam3.py` in this directory.
The originating host keeps its own pre-fix backups (`sam3.py.bak-pre-*`) for
reference; those are not vendored here since they are not needed to run the
pipeline.

## What's vendored and why

Only the modules the gateway's `capabilities/` package actually shells into
or imports transitively, plus their local-module dependencies:

| File | Used by (gateway) | Notes |
|---|---|---|
| `mask_hq.py` | `mode=prompt` (sam2 backend), `preset` (subject/sky/foreground) | GroundedSAM + BiRefNet + ViTMatte high-quality masking |
| `mask_c2f.py` | `mode=prompt`/`box`/`paint` (sam3 backend), `preset` (sam3 backend) | SAM3 concept segmentation, carve, depth-gating, feather |
| `mask_agentic.py` | `mode=prompt`/`box` with `agentic: true` | LLM (intent) + VLM (judge) refinement loop over `mask_c2f.py` |
| `grounded_sam.py` | imported by `mask_hq.py`/`mask_c2f.py`; also by the bundled `capabilities/mask_tools/mask_points.py` | GroundingDINO + SAM2 predictor plumbing |
| `raw_develop.py` | RAW sources, before any masking | `rawpy`-based multi-EV RAW decode (runs in the separate `rawtools` venv) |
| `exif_tools.py` | preset/EXIF-aware flows | EXIF read/write helpers |
| `vitmatte_refine.py` | standalone matte refinement (used by `mask_hq.py`'s matting step) | ViTMatte trimap alpha refine |
| `pelib/` (whole package) | imported by the above | shared building blocks — see below |

`pelib/` modules actually imported by the required scripts today: `comfy`
(ComfyUI API client), `imaging` (mask utilities, adaptive feather, edge
snap), `matte` (ViTMatte), `depth` (Depth Pro / DA-v2), `sam3` (SAM3
concept segmentation, multirep, carve), `zoomcarve` (depth-tiled lattice
carve), `vlm` (agentic judge calls). `enhance`, `finish`, and `sdxl` are
vendored too (whole-package, per the source layout) even though the current
gateway call paths don't reach them directly — future tool invocations or
manual CLI use may need them, and keeping the package intact avoids partial
imports breaking.

Explicitly **not** vendored (used by sibling scripts at `~/comfy` that the
gateway does not call): `crowd_erase.py`, `inpaint_erase_refine.py`,
`content_analyze.py`, `qwen_edit_region.py`, `flux_*.py`, `scene_finish.py`,
`solar_relight.py`, the `bench/`, `testimg.jpg`, and one-off `*_bench*.py`/
`step_bench*.py`/`build_*_workflow.py` scripts, plus anything under
`__pycache__/`, `.venv`/`venv`, logs, or the `ComfyUI` symlink.

## `GATEWAY_TOOLS_DIR`

`engine.py`'s `Settings.GATEWAY_TOOLS_DIR` (default `~/comfy`) tells
`capabilities/mask.py` where to find these scripts at runtime. It can point
at either:

1. **This vendored copy** — set `GATEWAY_TOOLS_DIR=/path/to/rr-ai-gateway/tools`.
   Self-contained, reproducible, and what `deploy/setup.sh`'s `tools` section
   installs by default (it `rsync`s this directory's contents into
   `GATEWAY_TOOLS_DIR`).
2. **A synced `~/comfy`** working copy on a GPU host where the maintainer
   continues to iterate on the pipeline directly (the historical setup on
   this box). In that case `deploy/setup.sh tools` still runs — it only adds
   files, it never deletes extras — so ad hoc scripts living alongside these
   in `~/comfy` are preserved.

Either way, the scripts expect `~/comfy` (or wherever `GATEWAY_TOOLS_DIR`
points) to be importable as-is: they do `sys.path.insert(0, tools_dir)` and
then `import grounded_sam` / `from pelib import ...`, so `pelib/` must sit
directly inside `GATEWAY_TOOLS_DIR`, not nested further.

## Runtime dependencies

These scripts run inside the **ComfyUI venv** (`GATEWAY_COMFY_VENV_PY`,
default `~/comfy/ComfyUI/.venv/bin/python`) except `raw_develop.py`, which
runs in a separate lightweight **rawtools venv** (`GATEWAY_RAWTOOLS_PY`,
default `~/rawtools/bin/python`) since it only needs `rawpy`/`opencv-python`/
`numpy`/`tifffile`/`piexif`, not the full ComfyUI + torch + transformers
stack. See `docs/SETUP-GUIDE.md` and `deploy/setup.sh` (`venv-extras`
section) for exact package lists.
