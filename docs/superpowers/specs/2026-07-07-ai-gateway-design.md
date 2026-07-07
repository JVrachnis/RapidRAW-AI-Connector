# rr-ai-gateway — General AI-Operations Gateway (Design)

**Date:** 2026-07-07
**Repo:** fork of CyberTimon/RapidRAW-AI-Connector (JVrachnis/RapidRAW-AI-Connector)
**Status:** approved design, pre-implementation

## Purpose

Evolve the RapidRAW AI Connector from a single-purpose inpainting middleware into a
general AI-operations gateway for a whole photography stack. Clients (the RapidRAW
fork, a future photography-management app, plain scripts, ComfyUI nodes) submit
images once and run any registered AI capability against them through one common
job API. The first shipped capability is high-quality AI masking, wrapping the
proven scripts already on inferno (`~/comfy/mask_hq.py`, `mask_agentic.py`,
`mask_c2f.py`, `raw_develop.py`). Existing inpainting keeps working unchanged.

## Goals

- One common HTTP API for all capabilities: discover, upload source, enqueue job,
  poll result. A client written once works with every capability.
- Capability plugin system: adding denoise / auto-edit / auto-tag / blur-regions
  later means registering a module, not touching the core.
- Masking capability v1 with five input modes: text prompt, click points,
  painted ROI, one-shot presets, plus an agentic toggle (LLM/VLM loop).
- Source payloads carry optional EXIF and optional RapidRAW `.rrdata` sidecar
  content as signals for the pipelines.
- Backward compatibility: stock RapidRAW pointed at this gateway still gets
  working `/health`, `/upload_source`, `/inpaint`.
- Queue with priorities suited to a single-GPU host (inferno).

## Non-Goals (v1)

- Implementing denoise, auto-edit, auto-tagging, blur-regions capabilities
  (the API is designed for them; they ship later as separate efforts).
- The photography-management app itself.
- Push notifications (WebSocket/SSE) — v1 is polling-only; an event stream is a
  planned later addition and must not be precluded by the design.
- Multi-GPU scheduling, multi-host workers, user accounts.

## Architecture

```
Clients                        Gateway (inferno)                 Executors
─────────                      ─────────────────────────         ─────────────
RapidRAW fork      ──┐         FastAPI                           ComfyUI API
Photo-mgmt app     ──┼──HTTP──►  capability registry     ──────► transformers scripts
curl / ComfyUI     ──┘           job queue (GPU-serial)          (mask_hq, mask_agentic,
nodes / scripts                  shared source cache              raw_develop, LLM/VLM…)
```

The gateway runs on inferno next to the GPU, ComfyUI, and the masking scripts.
It is a FastAPI app (extending the existing `main.py`/`engine.py`) composed of:

- **Capability registry** — each capability is a Python module registering
  `{id, title, params JSON-schema, handler}`. Registry is scanned at startup
  from a `capabilities/` package. v1 registers `mask` and `inpaint`.
- **Job queue** — a serial worker (one job on the GPU at a time) with two
  priority classes: `interactive` (default; mask requests from an editor) and
  `batch` (bulk operations from the management app). Interactive jobs are
  dequeued first. Queue state journaled to SQLite so a restart does not lose
  pending jobs; the in-flight job at crash time is re-marked `queued`.
- **Source cache** — content-addressed store (BLAKE3 hash of bytes → `source_id`)
  shared across all capabilities. Upload a photo once; mask, tag, and denoise it
  without re-sending. Entries hold the image file plus optional `exif.json` and
  `rrdata.json` sidecars. LRU eviction over a configurable size cap
  (`GATEWAY_CACHE_MAX_GB`, default 20). The existing inpaint JPEG cache is
  merged into this store.

### Handlers, not reimplementations

The `mask` handler shells into the existing scripts with their own venvs
(`~/comfy/ComfyUI/.venv/bin/python`, `~/rawtools/bin/python`) via `subprocess`,
exchanging files in a per-job temp dir. Rationale: the scripts are
battle-tested, own heavyweight model state, and evolve independently; the
gateway stays a thin orchestrator. Script paths and interpreters are
configuration (`GATEWAY_TOOLS_DIR`, default `~/comfy`), not hardcoded.

## Common Job API

All routes JSON unless noted. Optional bearer-token auth (mirrors the existing
connector's token behavior; `GATEWAY_TOKEN` env var, unauthenticated when unset).

### `GET /capabilities`

Returns the registry: `[{id, title, params_schema, modes?, presets?}, …]`.
Clients use `params_schema` (JSON Schema) to validate/build job params.

### `POST /sources` (multipart)

Fields:
- `file` — the image. Accepted kinds: `tiff` (linear 16-bit TIFF exported by the
  client, the alignment-exact path), `raw` (original camera file, e.g. .ARW),
  or standard 8-bit formats (JPEG/PNG) for casual clients.
- `exif` (optional) — JSON object of extracted EXIF fields.
- `rrdata` (optional) — the RapidRAW sidecar JSON verbatim.

Returns `{source_id, kind, width?, height?}` . Deduplicated by content hash:
re-uploading the same bytes returns the same `source_id` instantly.

### `POST /jobs/{capability}`

Body: `{source_id, params, priority?}` (`priority`: `interactive` | `batch`,
default `interactive`). Validates `params` against the capability schema.
Returns `202 {job_id, status: "queued", queue_position}`.

### `GET /jobs/{id}`

Returns `{job_id, capability, status, progress?, result?, error?}` with
`status ∈ queued | running | done | error | cancelled`. `result` is
capability-defined (see mask below). Completed results are kept for
`GATEWAY_RESULT_TTL_HOURS` (default 24) then pruned.

### `DELETE /jobs/{id}`

Cancels a queued job; best-effort kill (process group SIGTERM) for a running one.

### `GET /queue`

Lists all non-pruned jobs with statuses — powers a queue view in any client.

### `GET /health`

Extends the stock response with: ComfyUI reachability, registered capabilities,
queue depth, GPU name/VRAM if available.

### Back-compat adapters

`POST /upload_source` and `POST /inpaint` remain at their current paths and
semantics (synchronous response), implemented as thin adapters over the source
cache and an `inpaint` capability job enqueued at `interactive` priority and
awaited server-side. Stock RapidRAW does not know it is talking to the gateway.

## Mask capability (v1)

`POST /jobs/mask` params:

```jsonc
{
  "mode": "prompt" | "points" | "paint" | "preset",
  "query": "the person on the left",        // mode=prompt
  "points": [[x, y, 1], …],                 // mode=points; label 1=fg, 0=bg (source-pixel coords)
  "roi_mask_b64": "…png…",                  // mode=paint; rough painted region, any size (rescaled)
  "preset": "subject" | "sky" | "foreground",  // mode=preset
  "agentic": false,                          // LLM+VLM self-correcting loop (mask_agentic.py)
  "backend": "sam2" | "sam3",               // default sam2
  "matte": true,                             // ViTMatte edge refinement + defringe
  "ev_stack": "auto" | "off" | [-2,0,2,4]   // multi-EV develop for shadow recall (raw/tiff sources)
}
```

Mode → tooling:
- `prompt` → `mask_hq.py` (GroundedSAM → BiRefNet gate → ViTMatte); with
  `agentic: true` → `mask_agentic.py` wrapping `mask_c2f.py`.
- `points` → SAM2/SAM3 point prompts (via `grounded_sam.py` segment path).
- `paint` → ROI-gated segmentation: the ROI constrains detection/segmentation
  (passed as `--roi` where supported; otherwise used as the gating region).
- `preset` → fixed pipelines: `subject` = BiRefNet General-HR + matte;
  `sky` / `foreground` = corresponding dedicated flows.

Result:

```jsonc
{
  "mask_png_b64": "…",          // grayscale alpha at source resolution
  "width": 8640, "height": 5760,
  "labels": ["person", "bicycle"],  // what the detector found (UI feedback)
  "alignment": "exact" | "best_effort",
  "timings": {"total_s": 14.2}
}
```

### Source kinds and alignment

- `tiff` source: the mask is computed on exactly the client's pixels →
  `alignment: "exact"`. Multi-EV recall still works: EV multipliers apply to
  linear TIFF data the same as to a rawpy develop.
- `raw` source: the handler runs `raw_develop.py` (rawpy) first. Decoder margins
  can differ from the client's decode by a few pixels. Reconciliation: if
  `rrdata` (or client-supplied dimensions) is present, the mask is center-crop/
  padded to the client's stated full-image dimensions; response says
  `best_effort`. Clients should prefer `tiff` when pixel-exactness matters.

### rrdata / EXIF as signals

When present on the source, handlers may use:
- rrdata exposure/shadow adjustments → choose EV-stack push strength.
- rrdata crop/orientation → raw-mode alignment reconciliation.
- rrdata existing mask bitmaps → agentic-loop protect/seed regions.
- rrdata tags → extra grounding concepts appended to the query.
- EXIF (ISO, focal length) → available to handlers (e.g., ISO-scaled denoise
  later); for masking v1 it is stored and passed through, no hard dependency.

All signals are optional; every job must work with a bare image.

## Error handling

- Params failing schema validation → `422` with the schema error, job never queued.
- Handler subprocess non-zero exit / timeout (`GATEWAY_JOB_TIMEOUT_S`, default
  600) → job `error` with captured stderr tail (truncated) in `error.detail`.
- ComfyUI unreachable at job start for a ComfyUI-dependent step → job `error`
  with a distinguishable `error.kind: "comfyui_down"` so clients can hint the fix.
- Oversized inputs: handlers cap working resolution (as `mask_hq.py --maxside`
  does) and upscale the alpha back to source resolution — existing behavior,
  kept.
- Unknown `source_id` → `404`. Cache-evicted source → `410 Gone` so clients
  know to re-upload.

## Deployment

systemd user unit on inferno (`rr-ai-gateway.service`), listening on
`0.0.0.0:5000` (same port as the current connector — it replaces it). Env vars:
`COMFY_HOST/COMFY_PORT` (existing), `GATEWAY_TOKEN`, `GATEWAY_TOOLS_DIR`,
`GATEWAY_CACHE_MAX_GB`, `GATEWAY_RESULT_TTL_HOURS`, `GATEWAY_JOB_TIMEOUT_S`,
`GATEWAY_DB_PATH` (SQLite journal). The user's current local connector instance
(`~/Apps/RapidRAW-AI-Connector`) keeps running until the gateway is deployed
and verified; its `workflow.json` model-path tweaks carry over.

## Testing

- **Unit**: registry loading, params schema validation, queue ordering
  (priority + FIFO within class), SQLite journal recovery, source-cache
  dedup/eviction, raw-mode dimension reconciliation math.
- **Contract**: httpx test client against a fake capability (instant echo
  handler) covering the full job lifecycle including cancel and result TTL.
- **Integration (on inferno)**: pytest marked `@integration` running the real
  mask capability per mode against fixture images with known subjects,
  asserting mask non-emptiness, dimensions, and label detection. Back-compat:
  stock `/inpaint` flow against the real ComfyUI.
- **Manual E2E**: covered by the RapidRAW-fork spec's checklist.

## Future (explicitly anticipated, not built)

- Capabilities: `denoise`, `auto_edit` (style-conditioned), `auto_tag`,
  `blur_regions` (tattoos/faces), `upload_publish` (collaboration site).
- SSE/WebSocket job events replacing client polling.
- Job dependency chains (develop → mask → edit) for the management app.
