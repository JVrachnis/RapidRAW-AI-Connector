<p align="center">
  <img src="https://raw.githubusercontent.com/CyberTimon/RapidRAW/assets/.github/assets/editor.png" alt="RapidRAW Editor">
</p>

<div align="center">

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-%23009688.svg?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg?style=for-the-badge)](https://opensource.org/licenses/Apache-2.0)

</div>

# RapidRAW AI Connector

A lightweight middleware that connects [RapidRAW](https://github.com/CyberTimon/RapidRAW) to a [ComfyUI](https://github.com/comfyanonymous/ComfyUI) backend for fast, self-hosted generative AI edits.

> **Warning:** This project is a work in progress and considered unstable for the average user. Official support will begin with the release of **RapidRAW v1.4.9**.

---

## What It Does

This server acts as an intelligent cache between RapidRAW and ComfyUI to make generative edits *fast*.

Instead of sending a huge source image for every prompt change, the full image is sent **only once**. For all subsequent edits, only the tiny mask and text prompt are transferred. The connector sends the full job to ComfyUI and returns only the cropped, edited patch. This minimizes network traffic and makes the editing experience feel instant.

## Getting Started

#### 1. Prerequisites
*   A running instance of [ComfyUI](https://github.com/comfyanonymous/ComfyUI).
*   Python 3.10+

#### 2. Installation
```bash
git clone https://github.com/CyberTimon/RapidRAW-AI-Connector.git
cd RapidRAW-AI-Connector
pip install -r requirements.txt
```

#### 3. Configuration
All settings are managed via environment variables. The defaults should work for a standard local ComfyUI setup. You can change them by setting variables like `COMFY_HOST` and `COMFY_PORT` before running the script.

#### 4. Run It
```bash
python main.py
```

#### 5. Connect RapidRAW
In RapidRAW's settings, point the `Self-Hosted` AI Backend to the connector's address (e.g., `http://127.0.0.1:5000`).

## Customization
Tweak your generative process by editing the `workflow.json` file. You can use custom models, nodes, and samplers by updating the workflow and corresponding node IDs in `engine.py`.

## License
This AI Connector is licensed under the **Apache License 2.0**. See the [LICENSE](LICENSE) file for more details.

## rr-ai-gateway (this fork)

This fork extends the connector into a general, self-hostable AI-operations
gateway for RapidRAW: a job-queued FastAPI middleware in front of ComfyUI +
ollama + a vendored masking/RAW pipeline, with a content-addressed source
cache, nudge-aware result caching, and a persistent worker for fast
micro-adjustments.

### Quickstart

```bash
git clone <this repo> rr-ai-gateway && cd rr-ai-gateway
deploy/setup.sh              # idempotent: ComfyUI + custom nodes + models +
                              # tools + venvs + ollama + the gateway itself
deploy/setup.sh --dry-run    # preview without changing anything
deploy/setup.sh --only verify
```

Full architecture diagram, per-component explanations, manual (non-script)
install instructions, the complete env-var table, the mask modes/params
reference, performance numbers, and troubleshooting all live in
**[`docs/SETUP-GUIDE.md`](docs/SETUP-GUIDE.md)**. The vendored pipeline
scripts themselves (and their provenance) are documented in
**[`tools/README.md`](tools/README.md)**.

### Capability API

- `GET /capabilities` — discover capabilities + JSON Schema param definitions
- `POST /sources` — upload image (TIFF/RAW/JPEG) + optional `exif`/`rrdata` JSON once (content-addressed; re-uploading identical bytes is a no-op)
- `POST /jobs/{capability}` — enqueue; `GET /jobs/{id}` — poll; `DELETE /jobs/{id}` — cancel; `GET /queue` — list in-flight/queued jobs
- Capabilities v1: **`mask`** (GroundedSAM/SAM2/SAM3/BiRefNet/ViTMatte via the tools in `GATEWAY_TOOLS_DIR`) and **`inpaint`** (generative inpaint via ComfyUI's `workflow.json`)
- Legacy `/upload_source` + `/inpaint` kept byte-compatible for stock RapidRAW (see `gateway/legacy.py`)
- Job error `kind`s surfaced in `GET /jobs/{id}`'s `error.kind`: `gpu_oom`, `comfyui_down`, `tool_error`, `timeout`, `cancelled`, `handler_error`, `unknown_capability` — see the troubleshooting section of the setup guide for what causes each.

### Mask modes

`prompt` (backend `sam2` = `mask_hq.py` GroundedSAM+BiRefNet+ViTMatte, `sam3` =
`mask_c2f.py` SAM3 concept segmentation, `sam3_multirep` for multi-representation
2-GPU union detection), `points` (SAM2 point/label prompts), `paint` (a painted
ROI restricts the SAM3 search), `box` (sam2: bare box prompt; sam3/agentic: the
box becomes a region hint restricting the search, not a raw prompt), `preset`
(`subject`/`sky`/`foreground` canned queries, sam2 or sam3-backed), plus
`agentic` LLM/VLM refinement layered on top of `prompt`/`box` (with
`agentic_mode` `precise`|`removal`).

**SAM3 + agentic + carve/depth.** `mask_c2f.py`'s SAM3 path does semantic,
multi-concept segmentation ("the main subject", "bicycle, rider") with
optional `sam3_multirep` (union over multiple image representations, split
across both GPUs when available). `agentic: true` wraps it in
`mask_agentic.py`'s loop: an **intent LLM** (`GATEWAY_INTENT_LLM`, default
`gemma3:4b`) expands the target into search concepts, `mask_c2f.py` runs, and
a **judge VLM** (`GATEWAY_VLM_MODEL`, default `minicpm-v4.5:q4_K_M` — the only
local VLM judge benchmarked to reliably discriminate good/bad masks) scores
the result and decides whether to retry with adjusted thresholds. `carve`
(available on both the direct sam3 path via `--sam3-carve` and the agentic
path) recovers see-through/lattice structure (bike spokes, wire mesh) that a
coarse mask fills in solid; the gateway first runs a bundled Depth Pro tool
(`capabilities/mask_tools/make_depth.py`, wrapping `pelib.depth.depth_pro`
from `GATEWAY_TOOLS_DIR`) to produce a near=bright depth map and feeds it to
the carve step — the single biggest lattice-quality win measured in
benchmarking (e.g. BMX spokes fill 0.878→0.486). Depth generation is
best-effort: on failure the job continues with carve's own texture
heuristic rather than failing, and the result reports whether it was used
via `depth_used` (bool).

**Dynamic GPU selection.** Script-side tools default to `cuda:0`, which would
pile every masking job onto whichever card ComfyUI is pinned to.
`capabilities/mask.py`'s `pick_cuda_device()` picks the GPU with the most
free VRAM (via `nvidia-smi`) per job and pins it with `CUDA_VISIBLE_DEVICES`,
so masking work spreads across multiple cards instead of contending with
ComfyUI. Excluded for `sam3_multirep`, which needs both GPUs visible at once.

### Nudge caching

Micro-adjustment jobs — nudging a box or points slightly and re-running the
same capability on the same uploaded image — no longer pay for redundant
work:

- **Per-source depth cache.** Depth Pro (`_make_depth` in `capabilities/mask.py`) depends only on the source image bytes, never on job params, so its output is cached beside the source file as `<source path>.depth.png`. The first carve job on an image generates it (~10-15s including model load); every later carve job on the *same* uploaded image — including nudged box/points retries — reuses the cached map and reports `depth_used: true` without invoking the depth tool at all. Evicted alongside the source when the LRU (`GATEWAY_CACHE_MAX_GB`) trims it.
- **Exact-params result reuse.** `POST /jobs/{capability}` first checks for an existing `done` job with the same capability, `source_id`, and semantically-equal `params` (compared as parsed JSON, so key order never matters). A match returns `{"job_id": ..., "status": "done", "cached": true}` immediately with the *same* job id. Cancelled or errored jobs are never treated as cache hits. Staleness is bounded by `GATEWAY_RESULT_TTL_HOURS` job-row pruning; `source_id` is content-addressed so a hit can never point at stale image bytes. Pass `"no_cache": true` to force a fresh run.

### Persistent mask worker

Every subprocess mask job pays a cold model load (SAM3 ~10-20s) plus detection
on every run, even when only the geometric gating (a nudged box/ROI) changed.
A **resident worker** (`capabilities/mask_tools/mask_worker.py`) holds the
SAM3 models and per-source state in memory, so a micro-adjustment on the
*same* image+query replays in seconds instead of re-running the whole
pipeline.

**Architecture.** The worker runs in the ComfyUI venv (`GATEWAY_COMFY_VENV_PY`,
where `aiohttp` is already available) as an aiohttp server on
`GATEWAY_WORKER_URL` (default `127.0.0.1:5101`):

- `POST /mask` `{image_path, source_id, query, roi_png_b64?, box?, carve, depth_path?, multirep, out_path, mode}`
  → writes the final grayscale mask PNG to `out_path`, returns
  `{ok, out_path, coverage, cached_detections, timings}`.
- `GET /health` → `{ok, loaded_sources, models_loaded}`.
- **Per-source LRU (cap 2)** holding: the decoded BGR image, the enhanced detection
  image, the `pelib.sam3._representations` list (multirep only), the loaded depth map,
  and a **detection cache** keyed `(concept, rep_name, sam3_threshold, mask_threshold)`
  → `[(mask, score)]`. A nudged job whose detections are already cached **skips SAM3
  detection entirely** and redoes only ROI-gate + depth-gate + carve + feather.
- VRAM hygiene: `torch.cuda.empty_cache()` after each job; on CUDA OOM the LRU source
  state is dropped and the job retried once.
- The worker imports torch/cv2/pelib **lazily inside functions**, so its pure logic
  (LRU eviction, detection-cache keying) is unit-testable without a GPU.

**Pipeline parity.** The `/mask` handler mirrors `mask_c2f.py`'s non-agentic SAM3
branch faithfully — same `pelib` calls in the same order with the same defaults.
`birefnet`/`matte`/`protect`/`edge-snap`/`dilate` are all `backend == "sam2"`-gated
in `mask_c2f`, so the SAM3-only worker correctly skips them too. The worker does
not replicate LLM/VLM enhancement decisions, but those only exist on the
*agentic* path, which stays subprocess.

**Eligibility + fallback semantics.** Eligible jobs = `backend=sam3`, **non-agentic**,
`mode ∈ {prompt, box, paint}`. Points (SAM2) and agentic stay on the subprocess path.
When `GATEWAY_WORKER=auto` and a job is eligible, the gateway probes the worker's
`/health` (1s); if it is down it **spawns** it (detached subprocess, stdout →
`CACHE_DIR/worker.log`) and polls health up to 20s. **Any** failure (spawn, health,
request, bad response) is logged and falls back to the subprocess path — enabling
the worker carries zero behaviour-change risk. `GATEWAY_WORKER=off` always uses the
subprocess path and never probes.

**Parity + speed** (live gate, `~/comfy/bench/bmx_full.jpg`, query "the bmx bike and
rider." + box + carve, 4000×6000): subprocess **18.46s**, worker cold **20.01s**,
worker nudged (`cached_detections: true`) **~6.1s**. IoU(subprocess, worker-cold) =
1.0000 (bit-identical masks), well above the 0.95 parity gate. See
[`docs/SETUP-GUIDE.md`](docs/SETUP-GUIDE.md#performance-expectations) for the full
table and other paths' expected latency.

### Configuration

All settings are env vars read by `engine.py`'s `Settings`. Full table (every
variable, default, and purpose) is in
[`docs/SETUP-GUIDE.md`](docs/SETUP-GUIDE.md#environment-variables); the
headline ones: `GATEWAY_TOKEN` (bearer auth), `GATEWAY_TOOLS_DIR` (default
`~/comfy`), `GATEWAY_CACHE_MAX_GB` (20), `GATEWAY_RESULT_TTL_HOURS` (24),
`GATEWAY_JOB_TIMEOUT_S` (600), `GATEWAY_COMFY_VENV_PY`, `GATEWAY_RAWTOOLS_PY`,
`GATEWAY_LLM_URL`/`GATEWAY_INTENT_LLM`, `GATEWAY_VLM_URL`/`GATEWAY_VLM_MODEL`,
`GATEWAY_WORKER` (`auto`|`off`), `GATEWAY_WORKER_URL`.

### Integration points

- **RapidRAW desktop app**: point its "Self-Hosted" AI backend setting at the
  gateway's address (`http://<host>:5000` by default). See the sibling
  `RapidRAW/` checkout in this workspace for the client side.
- **RapidRAW plugin system**: `RapidRAW/src-tauri/src/plugins.rs` and
  `RapidRAW/plugin-examples/` (including a `gateway-mask-tools` example) show
  how RapidRAW plugins can call into a gateway like this one directly.

### Known v1 limits

- `ev_stack` param accepted but multi-EV `--det-images` wiring into `mask_c2f.py` is not connected yet.
- `matte` param accepted but not forwarded (tools' own defaults apply).
- Detected-object label parsing is best-effort per tool stdout format (see `parse_labels` in `capabilities/mask.py`).
- Legacy `/inpaint` waits behind the whole job queue (worst case `GATEWAY_JOB_TIMEOUT_S`); the original blocked only on its own ComfyUI call. Single-user deployments with no batch jobs are unaffected.
- Legacy source aliases (`/upload_source`) are in-memory and process-wide (single gateway instance only).
- Subprocess mask jobs pay a model cold-load (~5-30s). The persistent mask worker removes this for eligible sam3 jobs; ineligible jobs (agentic, points/SAM2) still pay it per subprocess. The worker's nudge replay still recomputes the Depth Pro zoom + carve (~6s) since those depend on the changed mask bbox.
- SAM3 points backend falls back to SAM2 with a warning (not wired for point prompts).
- The Depth Pro→DA-v2 fallback (`pelib.depth.depth_map`) needs a ComfyUI node (`DepthAnythingV2Preprocessor`) that isn't part of this stack's required custom-node set; if Depth Pro itself fails, that fallback will also fail (logged, non-fatal) unless you add the node yourself. See the setup guide's troubleshooting section.

### Manual / non-scripted deployment

For a from-scratch remote host without `deploy/setup.sh` (e.g. a bare rsync
deploy):
```bash
rsync -a --exclude .venv --exclude cache --exclude .git ./ inferno:~/rr-ai-gateway/
ssh inferno 'cd ~/rr-ai-gateway && python3 -m venv .venv && .venv/bin/pip install -r requirements.txt'
scp deploy/rr-ai-gateway.service inferno:~/.config/systemd/user/
ssh inferno 'systemctl --user daemon-reload && systemctl --user enable --now rr-ai-gateway'
```
This only sets up the gateway process itself — ComfyUI, custom nodes, models,
ollama, and the vendored tools still need `deploy/setup.sh` (or the manual
steps in `docs/SETUP-GUIDE.md`) run once per host.