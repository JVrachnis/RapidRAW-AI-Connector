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

This fork extends the connector into a general AI-operations gateway:

- `GET /capabilities` — discover capabilities + param schemas
- `POST /sources` — upload image (TIFF/RAW/JPEG) + optional `exif`/`rrdata` JSON once (content-addressed)
- `POST /jobs/{capability}` — enqueue; `GET /jobs/{id}` — poll; `DELETE /jobs/{id}` — cancel; `GET /queue` — list
- Capabilities v1: `mask` (GroundedSAM/SAM2/SAM3/BiRefNet/ViTMatte via the tools in `GATEWAY_TOOLS_DIR`), `inpaint`
- Mask modes: `prompt` (backend `sam2` = mask_hq GroundedSAM+BiRefNet, `sam3` = mask_c2f SAM3 concept segmentation, `sam3_multirep` for multi-representation 2-GPU union), `points`, `paint`, `preset`, plus `agentic` LLM/VLM refinement (with `carve` for see-through/lattice recovery and `agentic_mode` `precise`|`removal`)
- Legacy `/upload_source` + `/inpaint` kept byte-compatible for stock RapidRAW

Config env vars: `GATEWAY_TOKEN`, `GATEWAY_TOOLS_DIR` (default `~/comfy`),
`GATEWAY_CACHE_MAX_GB` (20), `GATEWAY_RESULT_TTL_HOURS` (24),
`GATEWAY_JOB_TIMEOUT_S` (600), `GATEWAY_DB_PATH`,
`GATEWAY_COMFY_VENV_PY`, `GATEWAY_RAWTOOLS_PY`,
`GATEWAY_LLM_URL`/`GATEWAY_INTENT_LLM` (agentic concept-expansion LLM),
`GATEWAY_VLM_URL`/`GATEWAY_VLM_MODEL` (agentic judge VLM).

Known v1 limits:
- `ev_stack` param accepted but multi-EV `--det-images` wiring into mask_c2f is not connected yet.
- `matte` param accepted but not forwarded (tools' own defaults apply).
- Labels parsing is best-effort per tool stdout format.
- Legacy `/inpaint` waits behind the whole job queue (worst case `GATEWAY_JOB_TIMEOUT_S`); the original blocked only on its own ComfyUI call. Single-user deployments with no batch jobs are unaffected.
- Legacy source aliases (`/upload_source`) are in-memory and process-wide.
- Each mask job pays model cold-load in its subprocess (~5-30s); a persistent tool-server is a future optimization.
- SAM3 points backend falls back to SAM2 with a warning.
- Agentic depth-map input not yet wired through the API (texture fallback active); Depth Pro integration is a follow-up.

Deploy on inferno (or any GPU host):
```bash
rsync -a --exclude .venv --exclude cache --exclude .git ./ inferno:~/rr-ai-gateway/
ssh inferno 'cd ~/rr-ai-gateway && python3 -m venv .venv && .venv/bin/pip install -r requirements.txt'
scp deploy/rr-ai-gateway.service inferno:~/.config/systemd/user/
ssh inferno 'systemctl --user daemon-reload && systemctl --user enable --now rr-ai-gateway'
```