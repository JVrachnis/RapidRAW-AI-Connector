# rr-ai-gateway — Setup Guide

This is the narrative companion to `deploy/setup.sh`. Read this once to
understand what the stack is made of and why; use the script to actually
install it.

## Architecture

```
                         ┌────────────────────┐
                         │      RapidRAW       │  (Tauri desktop app)
                         │  Self-Hosted AI      │
                         │  backend setting     │
                         └──────────┬───────────┘
                                    │ HTTP (source upload, jobs, /health)
                                    ▼
                         ┌────────────────────────────────────────┐
                         │            rr-ai-gateway                │
                         │  FastAPI app (main.py / gateway/app.py) │
                         │                                          │
                         │  routes.py    /sources /jobs /queue      │
                         │  legacy.py    /upload_source /inpaint    │
                         │  registry.py  capability params_schema   │
                         │  jobs.py      sqlite-backed job queue,   │
                         │               subprocess execution       │
                         │  store.py     content-addressed source   │
                         │               cache (LRU over disk GB)   │
                         └───┬───────────────────┬──────────────────┘
                             │                   │
        capabilities/mask.py │                   │ capabilities/inpaint.py
        (shells to tools)    │                   │ (talks to ComfyUI directly)
                             ▼                   ▼
        ┌────────────────────────────┐   ┌───────────────────────────┐
        │  subprocess tools           │   │        ComfyUI              │
        │  (GATEWAY_TOOLS_DIR, i.e.   │◄──┤  (COMFY_HOST:COMFY_PORT,   │
        │  tools/ vendored here or a  │   │   default 127.0.0.1:8188)  │
        │  synced ~/comfy):           │   │  workflow.json graph:      │
        │   mask_hq.py (sam2+BiRefNet │   │   RealVisXL Lightning ckpt │
        │     +ViTMatte)              │   │   + ControlNet union-sdxl  │
        │   mask_c2f.py (sam3 concept │   │   + InpaintCropImproved /  │
        │     segmentation, carve)    │   │     InpaintStitchImproved  │
        │   mask_agentic.py (LLM/VLM  │   └───────────────────────────┘
        │     refine loop over c2f)   │
        │   raw_develop.py (RAW→EV    │   ┌───────────────────────────┐
        │     frames, rawtools venv)  │   │         ollama              │
        │   grounded_sam.py, pelib/*  │──►│  (127.0.0.1:11434)          │
        └──────────┬──────────────────┘   │  gemma3:4b       (intent)   │
                   │ or, for eligible      │  minicpm-v4.5:q4_K_M (judge)│
                   │ sam3 jobs:            └───────────────────────────┘
                   ▼
        ┌────────────────────────────┐
        │  persistent mask worker     │   Resident aiohttp server,
        │  (mask_tools/mask_worker.py)│   127.0.0.1:5101, spawned on demand
        │  holds SAM3 models + a      │   in the ComfyUI venv. Falls back to
        │  per-source LRU + detection │   the subprocess path on ANY failure.
        │  cache in memory            │
        └────────────────────────────┘

        SAM2 ckpt: ~/tracking/models/sam2.1_hiera_large.pt
        ViTMatte:  ~/models/vitmatte-small
        HF cache:  jetjodh/sam3, IDEA-Research/grounding-dino-base,
                   apple/DepthPro-hf (transformers auto-download / hf CLI)
```

**What each piece does:**

- **RapidRAW** — the desktop editor. Points its "Self-Hosted" AI backend
  setting at the gateway's address.
- **rr-ai-gateway** (this repo) — a FastAPI middleware. Owns the job queue,
  the content-addressed source cache, capability discovery/param validation,
  and legacy byte-compatible endpoints for stock RapidRAW.
- **`tools/`** (vendored, or a synced `~/comfy`) — the actual masking/RAW
  pipeline: GroundingDINO+SAM2/SAM3 detection, BiRefNet/ViTMatte matting,
  Depth Pro depth-guided carve, RAW multi-EV development. Invoked as
  subprocesses in the ComfyUI venv (or the separate `rawtools` venv for RAW).
  See `tools/README.md` for exact provenance and file list.
- **ComfyUI** — runs the generative inpaint graph (`workflow.json`) and hosts
  the BiRefNet custom node used by the masking tools.
- **ollama** — serves the two small local LLMs used by the *agentic* masking
  mode: an intent/concept-expansion model and a VLM judge that scores mask
  quality and decides whether to retry.
- **Persistent mask worker** — an optional always-on process (spawned
  on-demand by the gateway) that keeps SAM3 + per-image state resident in
  memory so a "nudge" (small box/ROI adjustment on the same image) replays in
  seconds instead of re-paying a cold model load. Strictly additive: any
  failure falls back to the plain subprocess path.

## Quickstart

```bash
cd rr-ai-gateway
deploy/setup.sh                 # installs everything, idempotent, safe to re-run
deploy/setup.sh --dry-run       # see what it would do without changing anything
deploy/setup.sh --only verify   # just run the health-check table
```

Defaults assume a from-scratch box. Override with env vars if your paths
differ:

```bash
COMFY_HOME=/opt/ComfyUI GATEWAY_TOOLS_DIR=/opt/comfy-tools deploy/setup.sh
```

## Manual install, section by section

These mirror `deploy/setup.sh`'s sections; run them if you'd rather do it by
hand or need to understand what a section actually does.

### 1. Preflight (`check`)

- NVIDIA GPU + driver: `nvidia-smi`. The stack is GPU-bound (SAM2/SAM3,
  GroundingDINO, ViTMatte, Depth Pro, SDXL inpaint all run on CUDA).
- `python3 >= 3.10` (any recent CPython works; the reference box runs 3.14).
- `git`, `curl`.
- **~25GB free disk**: SDXL checkpoint (~7GB) + ControlNet union (~2.5GB) +
  VAE (~330MB) + SAM2 ckpt (~900MB) + SAM3/GroundingDINO/DepthPro HF caches
  (several GB combined) + ViTMatte (~200MB) + assorted venvs.

### 2. ComfyUI (`comfyui`)

```bash
git clone https://github.com/comfyanonymous/ComfyUI "$COMFY_HOME"
cd "$COMFY_HOME"
python3 -m venv venv && ln -s venv .venv     # the .venv symlink is this stack's convention
.venv/bin/pip install -r requirements.txt
# torch: only if not already present, and only from the CUDA index if a GPU exists
.venv/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

If `$COMFY_HOME` is already a symlink to an existing install (as on the
reference box, `~/comfy/ComfyUI -> ~/Apps/ComfyUI`), the script leaves it
alone and operates on the symlink's target.

### 3. Custom nodes (`custom-nodes`)

Only two node packs are required by the vendored tools + `workflow.json`:

| Node pack | Repo | Provides |
|---|---|---|
| `ComfyUI_BiRefNet_ll` | `https://github.com/lldacing/ComfyUI_BiRefNet_ll` | `AutoDownloadBiRefNetModel` / `GetMaskByBiRefNet` — used by `mask_hq.py` and `mask_c2f.py` |
| `ComfyUI-Inpaint-CropAndStitch` | `https://github.com/lquesada/ComfyUI-Inpaint-CropAndStitch.git` | `InpaintCropImproved` / `InpaintStitchImproved` — used by `workflow.json` (the generative inpaint graph) |

```bash
cd "$COMFY_HOME/custom_nodes"
git clone https://github.com/lldacing/ComfyUI_BiRefNet_ll
git clone https://github.com/lquesada/ComfyUI-Inpaint-CropAndStitch.git
"$COMFY_HOME/.venv/bin/pip" install -r ComfyUI_BiRefNet_ll/requirements.txt
# ComfyUI-Inpaint-CropAndStitch has no extra requirements.
```

Everything else in `workflow.json` (`CheckpointLoaderSimple`,
`CLIPTextEncode`, `ControlNetLoader`, `SetUnionControlNetType`,
`ControlNetApplyAdvanced`, `VAEEncode`/`VAEDecode`, `SetLatentNoiseMask`,
`KSampler`, `EmptyImage`, `ThresholdMask`, `ImageCompositeMasked`,
`LoadImage`, `InvertMask`, `VAELoader`, `PrimitiveInt`, `PreviewImage`) ships
in ComfyUI core — no extra node pack needed.

**Not required, and not installed by `deploy/setup.sh`:** `comfyui-inpaint-nodes`
and `ComfyUI_LayerStyle_Advance` exist on the reference box but only back
`crowd_erase.py` / `inpaint_erase_refine.py`, which are not part of the
vendored tool set (see `tools/README.md`). BiRefNet's "General-HR" model
also auto-downloads via the node on first use; pre-warm it by running a mask
job once, or leave it to lazy-load.

### 4. ComfyUI models (`models-comfy`)

| File | Destination (relative to `$COMFY_HOME/models/`) | Source |
|---|---|---|
| `RealVisXL_V5.0_Lightning_fp16.safetensors` | `checkpoints/` | `SG161222/RealVisXL_V5.0_Lightning` (HF) |
| `diffusion_pytorch_model_promax.safetensors` | `controlnet/SDXL/controlnet-union-sdxl-1.0/` | `xinsir/controlnet-union-sdxl-1.0` (HF) |
| `sdxl_vae.safetensors` | `vae/SDXL/` | `stabilityai/sdxl-vae` (HF; the repo ships a file literally named `sdxl_vae.safetensors`) |

```bash
hf download SG161222/RealVisXL_V5.0_Lightning RealVisXL_V5.0_Lightning_fp16.safetensors \
    --local-dir "$COMFY_HOME/models/checkpoints"
hf download xinsir/controlnet-union-sdxl-1.0 diffusion_pytorch_model_promax.safetensors \
    --local-dir "$COMFY_HOME/models/controlnet/SDXL/controlnet-union-sdxl-1.0"
hf download stabilityai/sdxl-vae sdxl_vae.safetensors \
    --local-dir "$COMFY_HOME/models/vae/SDXL"
```

All three repo IDs and exact filenames were verified against the live
HuggingFace API against the files actually present on the reference box (byte-identical names).

### 5. Python-side models (`models-python`)

| Model | Where | Source |
|---|---|---|
| SAM2.1 Hiera Large checkpoint | `~/tracking/models/sam2.1_hiera_large.pt` (898,083,611 bytes) | `https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt` |
| SAM3 | default HF cache (`~/.cache/huggingface/hub/models--jetjodh--sam3`) | `jetjodh/sam3` — an open mirror of the gated `facebook/sam3` (`pelib/sam3.py`'s `SAM3_ID`, byte-identical) |
| GroundingDINO | default HF cache | `IDEA-Research/grounding-dino-base` (`grounded_sam.py`'s `GD_ID`) |
| Depth Pro | default HF cache | `apple/DepthPro-hf` (`pelib/depth.py`'s `DEPTHPRO_ID`) |
| ViTMatte (small, Composition-1k) | `~/models/vitmatte-small` (also the `mask_hq.py --vitmatte` default) | `hustvl/vitmatte-small-composition-1k` |

```bash
mkdir -p ~/tracking/models
curl -L -C - -o ~/tracking/models/sam2.1_hiera_large.pt \
    https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt

hf download jetjodh/sam3
hf download IDEA-Research/grounding-dino-base
hf download apple/DepthPro-hf

hf download hustvl/vitmatte-small-composition-1k --local-dir ~/models/vitmatte-small
```

### 6. Vendored tools (`tools`)

```bash
rsync -a tools/ "$GATEWAY_TOOLS_DIR/"          # never --delete: additive only
ln -s "$COMFY_HOME" "$GATEWAY_TOOLS_DIR/ComfyUI"   # if not already present
```

See `tools/README.md` for what's vendored, why, and the two valid
`GATEWAY_TOOLS_DIR` conventions (a self-contained copy of this repo's
`tools/`, or a maintainer's live `~/comfy` working tree).

### 7. Extra venv packages (`venv-extras`)

Beyond ComfyUI's own `requirements.txt`:

```bash
"$COMFY_HOME/.venv/bin/pip" install sam2 transformers timm
python3 -m venv ~/rawtools
~/rawtools/bin/pip install rawpy opencv-python numpy tifffile piexif
```

`sam2`/`transformers`/`timm` back the SAM2 predictor, GroundingDINO/SAM3/
DepthPro/ViTMatte model classes, and BiRefNet's backbone, respectively. The
separate `rawtools` venv exists so `raw_develop.py` (RAW decode via `rawpy`)
doesn't need the multi-GB torch/transformers stack.

### 8. ollama (`ollama`)

```bash
curl -fsSL https://ollama.com/install.sh | sh    # system-wide install; gated behind --yes in the script
ollama pull gemma3:4b               # GATEWAY_INTENT_LLM
ollama pull minicpm-v4.5:q4_K_M     # GATEWAY_VLM_MODEL
```

`minicpm-v4.5:q4_K_M` is called out in the gateway's own docs as "the only
local VLM judge benchmarked to reliably discriminate good/bad masks" — don't
substitute a different VLM without re-benchmarking the agentic accept/retry
loop.

### 9. Gateway (`gateway`)

```bash
cd rr-ai-gateway
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
sed "s#%h/rr-ai-gateway#$(pwd)#g" deploy/rr-ai-gateway.service \
    > ~/.config/systemd/user/rr-ai-gateway.service
systemctl --user daemon-reload
systemctl --user enable --now rr-ai-gateway
curl -sf http://127.0.0.1:5000/health
```

`deploy/rr-ai-gateway.service` uses systemd's `%h` (home directory)
specifier for a from-scratch `~/rr-ai-gateway` checkout; `deploy/setup.sh`
substitutes the real repo path so it works from any clone location (this
repo lives at a nested path, not directly under `$HOME`).

### 10. Verify (`verify`)

```bash
curl -sf http://127.0.0.1:8188/system_stats     # ComfyUI up
curl -sf http://127.0.0.1:11434/api/version     # ollama up
curl -sf http://127.0.0.1:5000/health           # gateway up (also reports comfy connectivity)
"$COMFY_HOME/.venv/bin/python" -c "import sys; sys.path.insert(0, '$GATEWAY_TOOLS_DIR'); import pelib.sam3"
```

`deploy/setup.sh --only verify` runs all of the above plus a model-file
existence check and prints a status table.

## Environment variables

All are read by `engine.py`'s `Settings` (pydantic-settings; env vars
override the defaults below) unless noted otherwise.

| Variable | Default | Purpose |
|---|---|---|
| `HOST` | `0.0.0.0` | gateway bind address |
| `PORT` | `5000` | gateway HTTP port |
| `COMFY_HOST` | `127.0.0.1` | ComfyUI host |
| `COMFY_PORT` | `5545` in code / **`8188`** via the systemd unit and this stack's actual deployment | ComfyUI HTTP+WS port. Use `8188` — that's what the reference ComfyUI instance runs on and what `deploy/setup.sh` and the shipped systemd unit both assume. |
| `CACHE_DIR` | `./cache` | job workdirs, source cache, sqlite DB (unless overridden) |
| `WORKFLOW_FILE` | `workflow.json` | the ComfyUI inpaint graph template |
| `MAX_CACHE_FILES` / `MAX_CACHE_SIZE_MB` | `20` / `2048` | legacy in-memory `SourceCache` limits (the `/upload_source` legacy path) |
| `GATEWAY_TOKEN` | unset | if set, all `/capabilities`, `/sources`, `/jobs/*`, `/queue`, and legacy routes require `Authorization: Bearer <token>` |
| `GATEWAY_TOOLS_DIR` | `~/comfy` | where the vendored/ synced pipeline tools + `pelib/` live |
| `GATEWAY_CACHE_MAX_GB` | `20` | LRU cap for the content-addressed source store |
| `GATEWAY_RESULT_TTL_HOURS` | `24` | how long a `done`/`error`/`cancelled` job row (and its cached result) survives |
| `GATEWAY_JOB_TIMEOUT_S` | `600` | hard wall-clock timeout per job |
| `GATEWAY_DB_PATH` | `<CACHE_DIR>/gateway.sqlite3` | override the sqlite job/source DB path |
| `GATEWAY_COMFY_VENV_PY` | `~/comfy/ComfyUI/.venv/bin/python` | interpreter used for all ComfyUI-venv subprocess tools (mask_hq/mask_c2f/mask_agentic/vitmatte/the worker) |
| `GATEWAY_RAWTOOLS_PY` | `~/rawtools/bin/python` | interpreter used for `raw_develop.py` |
| `GATEWAY_LLM_URL` / `GATEWAY_INTENT_LLM` | ollama OpenAI-compat endpoint / `gemma3:4b` | agentic concept-expansion LLM |
| `GATEWAY_VLM_URL` / `GATEWAY_VLM_MODEL` | same endpoint / `minicpm-v4.5:q4_K_M` | agentic judge VLM (accept/retry the mask) |
| `GATEWAY_WORKER` | `auto` | `auto` = probe/spawn the persistent mask worker for eligible sam3 jobs, fall back to subprocess on any failure; `off` = always subprocess |
| `GATEWAY_WORKER_URL` | `http://127.0.0.1:5101` | resident mask worker endpoint |

`deploy/setup.sh`-only variables (not read by the gateway itself, just
control the installer): `COMFY_HOME`, `SAM2_CKPT_DIR`, `VITMATTE_DIR`,
`RAWTOOLS_VENV`, `PYTORCH_INDEX_URL`.

## Mask modes and params reference

`GET /capabilities` returns the live JSON Schema; this is the narrative
version. All modes are on the `mask` capability (`POST /jobs/mask`).

| `mode` | Required params | Backend routing |
|---|---|---|
| `prompt` | `query` | `backend: sam2` (default) → `mask_hq.py` (GroundingDINO+SAM2+BiRefNet+ViTMatte). `backend: sam3` → `mask_c2f.py` SAM3 concept segmentation. `agentic: true` → `mask_agentic.py`'s LLM/VLM refine loop over `mask_c2f.py`, regardless of `backend`. |
| `points` | `points: [[x,y,label],...]` | Always SAM2 (bundled `mask_points.py`); `backend: sam3` is accepted but falls back to SAM2 with a logged warning (SAM3 isn't wired for point prompts). |
| `box` | `box: [x0,y0,x1,y1]` | `backend: sam2` (default) → bare SAM2 box prompt via `mask_points.py`. `backend: sam3` → the box becomes a **region hint** (rendered as a coarse ROI mask) restricting `mask_c2f.py --roi`'s search, not a raw box prompt. `agentic: true` → same ROI-restricted search but through `mask_agentic.py`. |
| `paint` | `roi_mask_b64` | A free-form painted ROI mask restricts `mask_c2f.py --roi`'s search (SAM3-backed). |
| `preset` | `preset: subject\|sky\|foreground` | `backend: sam2` (default) → `mask_hq.py` with the preset's canned query (`mask_hq.py` OOMs on smaller GPUs for some presets, hence the `sam3` alternative). `backend: sam3` → routed through the lighter `mask_c2f.py` instead. |

Other params:

- `backend`: `sam2` (default) or `sam3`. `sam2` = GroundingDINO+SAM2+BiRefNet+
  ViTMatte (`mask_hq.py`), higher quality but heavier and prone to OOM on
  8GB cards for some presets. `sam3` = SAM3 concept segmentation
  (`mask_c2f.py`), semantic ("the main subject", multi-concept queries),
  LLM-in-the-loop when combined with `agentic`.
- `sam3_multirep`: (sam3 backend only, `prompt`/`box` modes) union SAM3
  detections over multiple image representations (contrast/gamma/saturation
  variants) for more robust detection; with 2 GPUs, `--sam3-parallel` splits
  the passes across both cards (~1.75x faster than serial). Excluded from
  the gateway's dynamic per-job GPU pinning since it needs both GPUs visible
  at once.
- `agentic`: runs `mask_agentic.py`'s intent-LLM → `mask_c2f.py` (sam2 or
  sam3) → judge-VLM loop, retrying with adjusted thresholds until the judge
  accepts the mask or `--max-iter` is hit. `agentic_mode`: `precise`
  (default, tight edges) or `removal` (looser, generation-friendly for
  object removal).
- `carve`: depth-tiled recovery of see-through/lattice structure (bicycle
  spokes, wire mesh, railings) that a coarse mask fills in solid. Available
  on the direct sam3 prompt path (`--sam3-carve`) and the agentic path
  (`--carve`), not sam2. The gateway first runs a bundled Depth Pro tool
  (`capabilities/mask_tools/make_depth.py`) to produce a near=bright depth
  map and passes it to the carve step — measured as the single biggest
  lattice-quality win in benchmarking (e.g. one BMX-spokes case: mask fill
  0.878→0.486, i.e. much less solid/more accurate). If depth generation
  fails, the job continues without it (texture-heuristic fallback) rather
  than failing; the result reports `depth_used: bool`.
- `matte`: accepted but not currently forwarded to the tools (v1 limit;
  tools use their own defaults).
- `ev_stack`: accepted but the multi-EV `--det-images` wiring into
  `mask_c2f.py` is not connected yet (v1 limit).

### Nudge caching + persistent worker

- **Per-source depth cache**: the first `carve`-enabled job on an uploaded
  image generates the depth map (~10-15s incl. model load) and caches it
  beside the source as `<source path>.depth.png`; every later carve job on
  the *same* source reuses it (`depth_used: true`, no depth tool invoked).
  Evicted alongside the source when the LRU trims it.
- **Exact-params result reuse**: `POST /jobs/{capability}` checks for an
  existing `done` job with the same capability/source/semantically-equal
  params and returns its result immediately (`cached: true`), unless the
  caller passes `no_cache: true`.
- **Persistent mask worker**: for eligible jobs (`backend: sam3`,
  non-agentic, `mode` in `prompt`/`box`/`paint`), the gateway probes (and, if
  needed, spawns) a resident aiohttp server that holds SAM3 models + a
  per-source LRU (cap 2) of decoded images, enhanced detection images,
  multirep representations, loaded depth maps, and a detection cache keyed
  `(concept, rep_name, sam3_threshold, mask_threshold)`. A nudged job whose
  detections are cached skips SAM3 detection entirely, redoing only ROI
  gate + depth gate + carve + feather. Falls back to the subprocess path on
  *any* failure (spawn, health probe, request, bad response) — enabling it
  carries zero behaviour-change risk. `GATEWAY_WORKER=off` disables probing
  entirely.

### Dynamic GPU selection

Script-side tools (SAM2/SAM3/GroundingDINO/ViTMatte/Depth Pro) default to
`cuda:0`, which would pile every masking job onto whichever card ComfyUI
happens to be pinned to. `capabilities/mask.py`'s `pick_cuda_device()` shells
to `nvidia-smi --query-gpu=index,memory.free` before each job (except
`sam3_multirep`, which needs both GPUs) and sets `CUDA_VISIBLE_DEVICES` to
whichever GPU currently has the most free VRAM, so masking work spreads
across multiple cards instead of contending with ComfyUI's inpaint jobs. On
the reference box, ComfyUI itself is pinned to GPU 1 (a 3060 Ti, 8GB) via
its own systemd unit's `CUDA_VISIBLE_DEVICES=1`, leaving GPU 0 (a 4070 Ti
Super, 16GB) free for on-demand masking work most of the time.

## Performance expectations

Measured live against `~/comfy/bench/bmx_full.jpg` (4000×6000), query "the
bmx bike and rider." + box + carve:

| Path | Time |
|---|---|
| subprocess (`GATEWAY_WORKER=off`) | ~18.5s |
| persistent worker, cold (no cached detections) | ~20.0s |
| persistent worker, nudged (`cached_detections: true`) | ~6.1s |

The nudged replay is dominated by the Depth Pro zoom + carve step (~6s),
which depends on the changed mask bbox and so can't be cached — SAM3
detection itself is skipped entirely. IoU between the subprocess and
worker-cold paths on that bench was 1.0000 (bit-identical masks), well above
the 0.95 parity gate.

Rough expectations for other paths (single RTX 4070 Ti Super/3060 Ti class
GPU; scale with image size and GPU headroom):

- **sam3 prompt, cold** (subprocess, no worker): dominated by SAM3 model
  load (~10-20s) + detection; carve/depth adds another few seconds.
- **sam3 prompt, worker nudge**: single-digit seconds once detections are
  cached (see table above).
- **sam2 prompt** (`mask_hq.py`, GroundingDINO+SAM2+BiRefNet+ViTMatte): a
  cold model load across four models; noticeably heavier than sam3 and more
  prone to VRAM pressure on 8GB cards (hence sam3-backed presets exist as a
  lighter alternative).
- **`sam3_multirep` with 2 GPUs**: ~1.75x faster than the serial
  single-GPU multirep pass, since the representations are split across
  both cards via `--sam3-parallel`.
- **agentic mode**: adds one ollama intent-LLM call up front and one
  VLM-judge call per retry iteration on top of the underlying sam2/sam3
  cost; each retry re-runs the whole `mask_c2f.py` pass (not worker-eligible).

## Troubleshooting

- **`gpu_oom`** (job error kind): a mask tool hit CUDA OOM
  (`classify_tool_failure` in `capabilities/mask.py` matches
  `OutOfMemoryError`/"out of memory" in the tool's stderr before anything
  else, since it's unambiguous). Mitigations: prefer `backend: sam3` over
  `sam2` for presets/prompts on 8GB cards; make sure `pick_cuda_device()` is
  actually finding a freer GPU (`nvidia-smi` must be on `PATH` and report
  free VRAM); avoid `sam3_multirep` (needs both GPUs, no per-job pinning) on
  a box under heavy concurrent load.
- **`comfyui_down`** (job error kind): after ruling out OOM,
  `classify_tool_failure` does an actual `ComfyClient.check_health()` probe
  (not string-matching "ComfyUI"/port numbers in the traceback — those
  appear in *every* tool traceback via the venv's interpreter path and
  previously caused unrelated failures to be misreported). Check
  `curl 127.0.0.1:$COMFY_PORT/system_stats` and `journalctl --user -u
  comfyui` (or however ComfyUI is run on your box).
- **Depth fallback warning** (`"[depth] Depth Pro failed, falling back to
  DA-V2"` / `"[depth] zoom-depth failed"` in tool stdout/stderr): `pelib.depth
  .depth_pro` (Apple DepthPro-hf via transformers) is the primary path; its
  fallback, `depth_map` (Depth Anything V2 via a ComfyUI node,
  `DepthAnythingV2Preprocessor`), requires a node that is **not** part of
  this stack's required custom-node set and is not installed by
  `deploy/setup.sh`. This is expected — the fallback failing is caught and
  logged, and `carve`/depth-gating features degrade gracefully to their
  texture-only heuristics rather than failing the job. If you need the DA-v2
  fallback to actually work, add whichever ComfyUI node pack provides
  `DepthAnythingV2Preprocessor` on top of this stack.
- **Worker fallback**: any failure in the persistent mask worker path
  (spawn, health probe timeout, request error, malformed response) is
  logged and the job silently retries on the subprocess path — this is by
  design, not a bug. Check `CACHE_DIR/worker.log` for why the worker itself
  is unhealthy (common causes: ComfyUI venv missing `aiohttp`, port 5101
  already bound by a stale process, GPU OOM inside the worker itself).
- **ollama cold-swap latency**: ollama unloads/reloads models between calls
  when VRAM is tight and multiple models are in rotation on the box (this
  reference box has 30+ pulled models). The first agentic call after a
  cold swap pays a model-load penalty on top of the ~gemma3:4b /
  minicpm-v4.5 inference cost; subsequent calls within ollama's keep-alive
  window are fast. If agentic latency is inconsistent, check `ollama ps`
  for eviction churn and consider a longer `OLLAMA_KEEP_ALIVE`.
- **`unknown_capability`/`timeout`/`cancelled`/`handler_error`** (other job
  error `kind`s from `gateway/jobs.py`): `timeout` = exceeded
  `GATEWAY_JOB_TIMEOUT_S`; `cancelled` = `DELETE /jobs/{id}` or a shutdown
  raced the handler; `handler_error` = an uncaught exception in the
  capability handler itself (check `error.detail` for the traceback tail).
