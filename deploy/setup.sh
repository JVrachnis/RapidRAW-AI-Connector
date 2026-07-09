#!/usr/bin/env bash
# deploy/setup.sh - idempotent installer for the rr-ai-gateway stack:
# ComfyUI + custom nodes + models, the SAM2/SAM3/DepthPro/ViTMatte python
# stack, ollama (intent + judge models), and the gateway itself.
#
# Every step is check-then-act: if the thing it installs is already present,
# it prints a "skip" line and moves on. Safe to re-run any time.
#
# Usage:
#   deploy/setup.sh [--dry-run] [--only SECTION] [--skip SECTION[,SECTION...]] [--yes] [-h|--help]
#
# Sections (run in this order by default):
#   check        - sanity-check the host (GPU, driver, python, git, curl, disk space)
#   comfyui      - clone ComfyUI, create its venv, install requirements + torch
#   custom-nodes - clone the custom node packs the vendored tools need
#   models-comfy - download the inpaint checkpoint + controlnet into ComfyUI/models
#   models-python- download SAM2/SAM3/GroundingDINO/DepthPro/ViTMatte weights
#   tools        - rsync tools/ into GATEWAY_TOOLS_DIR, wire up symlinks
#   venv-extras  - extra pip packages (sam2, transformers, timm) + rawtools venv
#   ollama       - install ollama (gated behind --yes) + pull the intent/judge models
#   gateway      - gateway venv, requirements, systemd user unit, health check
#   verify       - end-to-end probes; prints a status table
#
# Config (env vars, all optional -- defaults match a from-scratch box):
#   COMFY_HOME          ComfyUI checkout (default: ~/comfy/ComfyUI)
#   GATEWAY_TOOLS_DIR   where the vendored tools/ live at runtime (default: ~/comfy)
#   COMFY_PORT          ComfyUI HTTP port (default: 8188)
#   GATEWAY_PORT        gateway HTTP port (default: 5000, matches engine.py Settings.PORT)
#   SAM2_CKPT_DIR       SAM2 checkpoint directory (default: ~/tracking/models)
#   VITMATTE_DIR        ViTMatte weights directory (default: ~/models/vitmatte-small)
#   RAWTOOLS_VENV       rawtools venv path (default: ~/rawtools)
#   PYTORCH_INDEX_URL   pip index for torch when torch is missing
#                       (default: https://download.pytorch.org/whl/cu121)
set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults / config
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

COMFY_HOME="${COMFY_HOME:-$HOME/comfy/ComfyUI}"
GATEWAY_TOOLS_DIR="${GATEWAY_TOOLS_DIR:-$HOME/comfy}"
COMFY_PORT="${COMFY_PORT:-8188}"
GATEWAY_PORT="${GATEWAY_PORT:-5000}"
SAM2_CKPT_DIR="${SAM2_CKPT_DIR:-$HOME/tracking/models}"
VITMATTE_DIR="${VITMATTE_DIR:-$HOME/models/vitmatte-small}"
RAWTOOLS_VENV="${RAWTOOLS_VENV:-$HOME/rawtools}"
PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"

DRY_RUN=0
ASSUME_YES=0
ONLY=""
SKIP_CSV=""
CURRENT_SECTION="setup"

SECTIONS=(check comfyui custom-nodes models-comfy models-python tools venv-extras ollama gateway verify)

# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
log() { printf '[setup:%s] %s\n' "$CURRENT_SECTION" "$*"; }

run_cmd() {
    if (( DRY_RUN )); then
        log "DRY-RUN would run: $*"
        return 0
    fi
    log "+ $*"
    "$@"
}

contains() {
    local needle="$1"; shift
    local x
    for x in "$@"; do
        [[ "$x" == "$needle" ]] && return 0
    done
    return 1
}

usage() {
    sed -n '2,33p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

resolve_comfy_home() {
    if [[ -L "$COMFY_HOME" ]]; then
        COMFY_REAL="$(readlink -f "$COMFY_HOME")"
        log "COMFY_HOME ($COMFY_HOME) is a symlink -> $COMFY_REAL; leaving it alone"
    else
        COMFY_REAL="$COMFY_HOME"
    fi
}

comfy_venv_python() {
    resolve_comfy_home
    printf '%s/.venv/bin/python' "$COMFY_REAL"
}

hf_bin() {
    if command -v hf >/dev/null 2>&1; then
        echo "hf"
    elif command -v huggingface-cli >/dev/null 2>&1; then
        echo "huggingface-cli"
    else
        echo ""
    fi
}

hf_cache_dir_for() {
    local repo="$1"
    echo "$HOME/.cache/huggingface/hub/models--${repo//\//--}"
}

hf_download_file() {
    # hf_download_file REPO FILE DEST_DIR - download a single file from a HF
    # repo into DEST_DIR (flat, matching the file's basename). Skips cleanly
    # if the file already exists.
    local repo="$1" file="$2" dest_dir="$3"
    local dest="$dest_dir/$file"
    if [[ -f "$dest" ]]; then
        log "already present: $dest (skip)"
        return 0
    fi
    run_cmd mkdir -p "$dest_dir"
    local hf; hf="$(hf_bin)"
    if [[ -n "$hf" ]]; then
        run_cmd "$hf" download "$repo" "$file" --local-dir "$dest_dir"
    else
        log "no hf/huggingface-cli found; falling back to curl (resumable)"
        run_cmd curl -L -C - --create-dirs -o "$dest" "https://huggingface.co/$repo/resolve/main/$file"
    fi
}

hf_download_repo() {
    # hf_download_repo REPO - pre-warm the default HF cache with a whole repo
    # snapshot. Skips cleanly if the repo's cache dir already exists.
    local repo="$1"
    local cache_dir; cache_dir="$(hf_cache_dir_for "$repo")"
    if [[ -d "$cache_dir" ]]; then
        log "already cached: $repo ($cache_dir) (skip)"
        return 0
    fi
    local hf; hf="$(hf_bin)"
    if [[ -n "$hf" ]]; then
        run_cmd "$hf" download "$repo"
    else
        log "WARNING: no hf/huggingface-cli found; $repo will lazy-download on first pipeline use instead"
    fi
}

# ---------------------------------------------------------------------------
# check
# ---------------------------------------------------------------------------
section_check() {
    if command -v nvidia-smi >/dev/null 2>&1; then
        log "nvidia-smi: $(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | tr '\n' '; ')"
    else
        log "WARNING: nvidia-smi not found; GPU-backed steps (torch CUDA, mask tools) will not work"
    fi

    local pyver
    if command -v python3 >/dev/null 2>&1; then
        pyver="$(python3 -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
        if python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)'; then
            log "python3: $pyver (OK, >= 3.10)"
        else
            log "WARNING: python3 $pyver found, but >= 3.10 is required"
        fi
    else
        log "WARNING: python3 not found"
    fi

    if command -v git >/dev/null 2>&1; then
        log "git: $(git --version)"
    else
        log "WARNING: git not found"
    fi

    if command -v curl >/dev/null 2>&1; then
        log "curl: $(curl --version | head -1)"
    else
        log "WARNING: curl not found"
    fi

    local avail_kb avail_gb need_gb=25
    avail_kb="$(df -Pk "$HOME" | tail -1 | awk '{print $4}')"
    avail_gb=$(( avail_kb / 1024 / 1024 ))
    if (( avail_gb >= need_gb )); then
        log "disk space: ${avail_gb}GB free under \$HOME (OK, need ~${need_gb}GB for models)"
    else
        log "WARNING: only ${avail_gb}GB free under \$HOME; need ~${need_gb}GB for the model set"
    fi
}

# ---------------------------------------------------------------------------
# comfyui
# ---------------------------------------------------------------------------
section_comfyui() {
    resolve_comfy_home

    if [[ -L "$COMFY_HOME" ]]; then
        : # symlink already handled/logged by resolve_comfy_home
    elif [[ -d "$COMFY_REAL/.git" ]]; then
        log "ComfyUI already checked out at $COMFY_REAL (skip clone)"
    elif [[ -e "$COMFY_REAL" ]]; then
        log "$COMFY_REAL exists but is not a git checkout; leaving it alone (skip clone)"
    else
        run_cmd git clone https://github.com/comfyanonymous/ComfyUI "$COMFY_REAL"
    fi

    local venv_dir="$COMFY_REAL/venv"
    if [[ -d "$COMFY_REAL/.venv" && ! -L "$COMFY_REAL/.venv" ]]; then
        venv_dir="$COMFY_REAL/.venv"
    fi
    if [[ -x "$venv_dir/bin/python" ]]; then
        log "ComfyUI venv already present at $venv_dir (skip create)"
    else
        run_cmd python3 -m venv "$venv_dir"
    fi

    if [[ -e "$COMFY_REAL/.venv" ]]; then
        log ".venv already present at $COMFY_REAL/.venv (skip symlink)"
    else
        run_cmd ln -s "$(basename "$venv_dir")" "$COMFY_REAL/.venv"
    fi

    local py="$COMFY_REAL/.venv/bin/python"
    if [[ -f "$COMFY_REAL/requirements.txt" ]]; then
        if [[ -x "$py" ]]; then
            run_cmd "$py" -m pip install -r "$COMFY_REAL/requirements.txt"
        else
            log "DRY-RUN would install $COMFY_REAL/requirements.txt (venv python not present yet)"
        fi
    else
        log "no requirements.txt at $COMFY_REAL yet (fresh clone not materialized in dry-run; skip)"
    fi

    if [[ -x "$py" ]] && "$py" -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" >/dev/null 2>&1; then
        local tv; tv="$("$py" -c 'import torch; print(torch.__version__)')"
        log "torch already installed with CUDA available (skip): $tv"
    elif [[ -x "$py" ]] && "$py" -c "import torch" >/dev/null 2>&1; then
        local tv; tv="$("$py" -c 'import torch; print(torch.__version__)')"
        log "torch already installed ($tv) but CUDA not available; leaving as-is (skip reinstall)"
    else
        if command -v nvidia-smi >/dev/null 2>&1; then
            log "torch missing; installing CUDA build from $PYTORCH_INDEX_URL"
            run_cmd "$py" -m pip install torch torchvision --index-url "$PYTORCH_INDEX_URL"
        else
            log "torch missing and no GPU detected; installing CPU build"
            run_cmd "$py" -m pip install torch torchvision
        fi
    fi
}

# ---------------------------------------------------------------------------
# custom-nodes
# ---------------------------------------------------------------------------
section_custom_nodes() {
    resolve_comfy_home
    local nodes_dir="$COMFY_REAL/custom_nodes"
    run_cmd mkdir -p "$nodes_dir"

    local name url
    for pair in \
        "ComfyUI_BiRefNet_ll=https://github.com/lldacing/ComfyUI_BiRefNet_ll" \
        "ComfyUI-Inpaint-CropAndStitch=https://github.com/lquesada/ComfyUI-Inpaint-CropAndStitch.git"
    do
        name="${pair%%=*}"; url="${pair#*=}"
        local dest="$nodes_dir/$name"
        if [[ -d "$dest/.git" ]]; then
            log "$name already present at $dest (skip clone)"
        else
            run_cmd git clone "$url" "$dest"
        fi

        local py="$COMFY_REAL/.venv/bin/python"
        if [[ -f "$dest/requirements.txt" ]]; then
            if [[ -x "$py" ]]; then
                run_cmd "$py" -m pip install -r "$dest/requirements.txt"
            else
                log "DRY-RUN would install $dest/requirements.txt (venv python not present yet)"
            fi
        fi
    done

    log "note: comfyui-inpaint-nodes and ComfyUI_LayerStyle_Advance are NOT installed by" \
        "this section -- the vendored tools/workflow.json don't need them (see" \
        "docs/SETUP-GUIDE.md). Add them manually if you use the unvendored crowd_erase.py" \
        "/ inpaint_erase_refine.py scripts from the maintainer's full ~/comfy tree."
}

# ---------------------------------------------------------------------------
# models-comfy
# ---------------------------------------------------------------------------
section_models_comfy() {
    resolve_comfy_home
    hf_download_file "SG161222/RealVisXL_V5.0_Lightning" \
        "RealVisXL_V5.0_Lightning_fp16.safetensors" \
        "$COMFY_REAL/models/checkpoints"
    hf_download_file "xinsir/controlnet-union-sdxl-1.0" \
        "diffusion_pytorch_model_promax.safetensors" \
        "$COMFY_REAL/models/controlnet/SDXL/controlnet-union-sdxl-1.0"
    hf_download_file "stabilityai/sdxl-vae" \
        "sdxl_vae.safetensors" \
        "$COMFY_REAL/models/vae/SDXL"
}

# ---------------------------------------------------------------------------
# models-python
# ---------------------------------------------------------------------------
section_models_python() {
    local sam2_dest="$SAM2_CKPT_DIR/sam2.1_hiera_large.pt"
    if [[ -f "$sam2_dest" ]]; then
        log "already present: $sam2_dest (skip)"
    else
        run_cmd mkdir -p "$SAM2_CKPT_DIR"
        run_cmd curl -L -C - -o "$sam2_dest" \
            "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt"
    fi

    hf_download_repo "jetjodh/sam3"
    hf_download_repo "IDEA-Research/grounding-dino-base"
    hf_download_repo "apple/DepthPro-hf"

    if [[ -f "$VITMATTE_DIR/model.safetensors" ]]; then
        log "already present: $VITMATTE_DIR/model.safetensors (skip)"
    else
        run_cmd mkdir -p "$VITMATTE_DIR"
        local hf; hf="$(hf_bin)"
        if [[ -n "$hf" ]]; then
            run_cmd "$hf" download hustvl/vitmatte-small-composition-1k --local-dir "$VITMATTE_DIR"
        else
            log "WARNING: no hf/huggingface-cli found; skipping vitmatte download"
        fi
    fi
}

# ---------------------------------------------------------------------------
# tools
# ---------------------------------------------------------------------------
section_tools() {
    resolve_comfy_home
    run_cmd mkdir -p "$GATEWAY_TOOLS_DIR"
    # rsync WITHOUT --delete: adds/updates vendored files, never removes
    # anything else already living in GATEWAY_TOOLS_DIR (e.g. an existing
    # ~/comfy working tree with extra ad hoc scripts).
    run_cmd rsync -a "$REPO_DIR/tools/" "$GATEWAY_TOOLS_DIR/"

    local comfy_link="$GATEWAY_TOOLS_DIR/ComfyUI"
    if [[ -e "$comfy_link" ]]; then
        log "$comfy_link already present (skip symlink)"
    else
        run_cmd ln -s "$COMFY_REAL" "$comfy_link"
    fi

    if [[ -e "$comfy_link/.venv" ]]; then
        log "$comfy_link/.venv already present (skip)"
    else
        log "$comfy_link/.venv missing; run --only comfyui to create the ComfyUI venv first"
    fi
}

# ---------------------------------------------------------------------------
# venv-extras
# ---------------------------------------------------------------------------
section_venv_extras() {
    resolve_comfy_home
    local py="$COMFY_REAL/.venv/bin/python"
    if [[ ! -x "$py" ]]; then
        log "ComfyUI venv not found at $py; run --only comfyui first (skip)"
    else
        local pkg
        for pkg in sam2 transformers timm; do
            if "$py" -c "import $pkg" >/dev/null 2>&1; then
                log "$pkg already importable in the ComfyUI venv (skip)"
            else
                run_cmd "$py" -m pip install "$pkg"
            fi
        done
    fi

    if [[ -x "$RAWTOOLS_VENV/bin/python" ]]; then
        log "rawtools venv already present at $RAWTOOLS_VENV (skip create)"
    else
        run_cmd python3 -m venv "$RAWTOOLS_VENV"
    fi

    if [[ -x "$RAWTOOLS_VENV/bin/python" ]] && \
       "$RAWTOOLS_VENV/bin/python" -c "import rawpy, cv2, numpy, tifffile, piexif" >/dev/null 2>&1; then
        log "rawtools deps already satisfied (skip)"
    else
        run_cmd "$RAWTOOLS_VENV/bin/pip" install rawpy opencv-python numpy tifffile piexif
    fi
}

# ---------------------------------------------------------------------------
# ollama
# ---------------------------------------------------------------------------
section_ollama() {
    if command -v ollama >/dev/null 2>&1; then
        log "ollama already installed: $(ollama --version 2>&1 | head -1)"
    else
        if (( ASSUME_YES )); then
            log "installing ollama via the official install script"
            run_cmd bash -c 'curl -fsSL https://ollama.com/install.sh | sh'
        else
            log "ollama not found; re-run with --yes to auto-install it system-wide," \
                "or install it yourself: https://ollama.com/download"
        fi
    fi

    local model
    for model in "gemma3:4b" "minicpm-v4.5:q4_K_M"; do
        if command -v ollama >/dev/null 2>&1 && ollama list 2>/dev/null | awk '{print $1}' | grep -qxF "$model"; then
            log "$model already pulled (skip)"
        elif command -v ollama >/dev/null 2>&1; then
            run_cmd ollama pull "$model"
        else
            log "DRY-RUN would pull $model (ollama not installed yet)"
        fi
    done
}

# ---------------------------------------------------------------------------
# gateway
# ---------------------------------------------------------------------------
section_gateway() {
    local py="$REPO_DIR/.venv/bin/python"
    if [[ -x "$py" ]]; then
        log "gateway venv already present at $REPO_DIR/.venv (skip create)"
    else
        run_cmd python3 -m venv "$REPO_DIR/.venv"
    fi
    run_cmd "$REPO_DIR/.venv/bin/pip" install -r "$REPO_DIR/requirements.txt"

    local unit_dir="$HOME/.config/systemd/user"
    local unit_dest="$unit_dir/rr-ai-gateway.service"
    local rendered
    rendered="$(sed "s#%h/rr-ai-gateway#$REPO_DIR#g" "$REPO_DIR/deploy/rr-ai-gateway.service")"

    if [[ -f "$unit_dest" ]] && diff -q <(printf '%s\n' "$rendered") "$unit_dest" >/dev/null 2>&1; then
        log "systemd user unit already up to date at $unit_dest (skip)"
    elif (( DRY_RUN )); then
        log "DRY-RUN would write $unit_dest (paths substituted for $REPO_DIR) and enable+start it"
    else
        mkdir -p "$unit_dir"
        printf '%s\n' "$rendered" > "$unit_dest"
        log "wrote $unit_dest"
        if command -v systemctl >/dev/null 2>&1; then
            systemctl --user daemon-reload
            systemctl --user enable --now rr-ai-gateway
            log "enabled + started rr-ai-gateway.service"
        else
            log "WARNING: systemctl not found; unit written but not enabled/started"
        fi
    fi

    if (( DRY_RUN )); then
        log "DRY-RUN would health-check http://127.0.0.1:$GATEWAY_PORT/health"
    else
        sleep 1
        if curl -sf "http://127.0.0.1:$GATEWAY_PORT/health" >/dev/null 2>&1; then
            log "gateway health check OK (http://127.0.0.1:$GATEWAY_PORT/health)"
        else
            log "gateway health check did not respond yet; it may still be starting" \
                "(check: journalctl --user -u rr-ai-gateway -n 50)"
        fi
    fi
}

# ---------------------------------------------------------------------------
# verify
# ---------------------------------------------------------------------------
section_verify() {
    resolve_comfy_home
    declare -A results

    if curl -sf "http://127.0.0.1:${COMFY_PORT}/system_stats" >/dev/null 2>&1; then
        results["comfyui /system_stats"]="OK"
    else
        results["comfyui /system_stats"]="FAIL"
    fi

    if curl -sf "http://127.0.0.1:11434/api/version" >/dev/null 2>&1; then
        results["ollama /api/version"]="OK"
    else
        results["ollama /api/version"]="FAIL"
    fi

    if curl -sf "http://127.0.0.1:${GATEWAY_PORT}/health" >/dev/null 2>&1; then
        results["gateway /health"]="OK"
    else
        results["gateway /health"]="FAIL"
    fi

    local py="$COMFY_REAL/.venv/bin/python"
    if [[ -x "$py" ]] && "$py" -c "
import sys
sys.path.insert(0, '$GATEWAY_TOOLS_DIR')
import pelib.sam3
" >/dev/null 2>&1; then
        results["pelib.sam3 importable (ComfyUI venv)"]="OK"
    else
        results["pelib.sam3 importable (ComfyUI venv)"]="FAIL"
    fi

    local entry label path
    for entry in \
        "checkpoint (RealVisXL Lightning fp16)=$COMFY_REAL/models/checkpoints/RealVisXL_V5.0_Lightning_fp16.safetensors" \
        "controlnet (SDXL union promax)=$COMFY_REAL/models/controlnet/SDXL/controlnet-union-sdxl-1.0/diffusion_pytorch_model_promax.safetensors" \
        "vae (sdxl_vae)=$COMFY_REAL/models/vae/SDXL/sdxl_vae.safetensors" \
        "SAM2 checkpoint=$SAM2_CKPT_DIR/sam2.1_hiera_large.pt" \
        "ViTMatte weights=$VITMATTE_DIR/model.safetensors"
    do
        label="${entry%%=*}"; path="${entry#*=}"
        if [[ -f "$path" ]]; then
            results["model: $label"]="OK"
        else
            results["model: $label"]="MISSING"
        fi
    done

    echo
    echo "== rr-ai-gateway verify status =="
    printf '%-42s %s\n' "CHECK" "STATUS"
    printf '%-42s %s\n' "-----" "------"
    local key
    for key in "${!results[@]}"; do
        printf '%-42s %s\n' "$key" "${results[$key]}"
    done | sort
    echo
}

# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
run_section() {
    local name="$1"
    CURRENT_SECTION="$name"
    case "$name" in
        check)         section_check ;;
        comfyui)       section_comfyui ;;
        custom-nodes)  section_custom_nodes ;;
        models-comfy)  section_models_comfy ;;
        models-python) section_models_python ;;
        tools)         section_tools ;;
        venv-extras)   section_venv_extras ;;
        ollama)        section_ollama ;;
        gateway)       section_gateway ;;
        verify)        section_verify ;;
        *)
            echo "unknown section: $name" >&2
            echo "valid sections: ${SECTIONS[*]}" >&2
            exit 1
            ;;
    esac
}

main() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --dry-run) DRY_RUN=1; shift ;;
            --yes) ASSUME_YES=1; shift ;;
            --only) ONLY="$2"; shift 2 ;;
            --only=*) ONLY="${1#*=}"; shift ;;
            --skip) SKIP_CSV="$2"; shift 2 ;;
            --skip=*) SKIP_CSV="${1#*=}"; shift ;;
            -h|--help) usage; exit 0 ;;
            *)
                echo "unknown argument: $1" >&2
                usage
                exit 1
                ;;
        esac
    done

    if [[ -n "$ONLY" ]]; then
        if ! contains "$ONLY" "${SECTIONS[@]}"; then
            echo "unknown section: $ONLY" >&2
            echo "valid sections: ${SECTIONS[*]}" >&2
            exit 1
        fi
        run_section "$ONLY"
        return 0
    fi

    local -a skip_list=()
    if [[ -n "$SKIP_CSV" ]]; then
        IFS=',' read -r -a skip_list <<< "$SKIP_CSV"
    fi

    local section
    for section in "${SECTIONS[@]}"; do
        if [[ ${#skip_list[@]} -gt 0 ]] && contains "$section" "${skip_list[@]}"; then
            CURRENT_SECTION="$section"
            log "skipping section (--skip)"
            continue
        fi
        run_section "$section"
    done
}

main "$@"
