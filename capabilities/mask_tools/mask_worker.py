#!/usr/bin/env python3
"""mask_worker.py - persistent resident mask worker (rr-ai-gateway Phase 2).

Runs in the ComfyUI venv (`~/comfy/ComfyUI/.venv/bin/python`) and holds the
SAM3 models + per-source detection state resident across jobs so that a nudged
box/points/ROI on the SAME image+query re-runs only the cheap geometric gating
(~2-4s) instead of the whole cold pipeline (model load ~10-20s + detection).

HTTP server on 127.0.0.1:5101 (aiohttp; ComfyUI's own server already uses it in
this venv). Endpoints:
  POST /mask   -> run/replay a mask job, write grayscale PNG to out_path
  GET  /health -> {ok, loaded_sources, models_loaded}

PIPELINE PARITY: the /mask handler mirrors mask_c2f.py's non-agentic SAM3 branch
(single-concept path) faithfully -- same pelib calls, same order, same defaults:
enhance_for_detection (auto-tuned) -> S3.instances per concept -> ROI gate ->
(multirep: combine_instances) -> depth-aware hole-fill -> optional carve
(continuity w/ depth else seethrough) -> adaptive feather (precise) else binary.
Divergences from mask_c2f are documented at DIVERGENCES below.

IMPORTANCE OF THE torch GUARD: everything heavy (torch, cv2, pelib) is imported
lazily INSIDE functions, so the pure-logic classes below (SourceState LRU,
detection cache keying) import and unit-test without a GPU/torch present.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import traceback
from collections import OrderedDict
from typing import Any, Optional

logger = logging.getLogger("mask_worker")

# ---------------------------------------------------------------------------
# DIVERGENCES from mask_c2f.py (accepted; the live parity gate decides if they
# matter):
#  * No LLM/VLM enhancement decisions -- the worker only serves the
#    non-agentic path, so there is nothing to diverge on there.
#  * enhance auto-tuning, concept split, ROI gate, hole-fill, carve and feather
#    are replicated exactly. `matte`/`birefnet`/`protect`/`edge-snap` do NOT run
#    on the SAM3 path in mask_c2f (they are gated to `backend == "sam2"`), so the
#    worker (SAM3-only) correctly skips them too.
#  * mask_c2f re-reads a precomputed --depth-map from disk for the depth
#    hole-fill/gate; the worker loads the same depth PNG (depth_path) once and
#    caches it per source, matching the read semantics.
# ---------------------------------------------------------------------------

# Defaults mirrored from mask_c2f.py's argparse so cache keys + behaviour match.
SAM3_THRESHOLD = 0.5
SAM3_MASK_THRESHOLD = 0.5
DEPTH_TOL = 0.12
ENH_GAMMA = 0.55
ENH_CLAHE = 3.0
ENH_SHARP = 0.6
FEATHER_MAX = 20.0


def detection_key(concept: str, rep_name: str, sam3_threshold: float,
                  mask_threshold: float) -> tuple:
    """Cache key for a per-source SAM3 detection: identical (concept,
    representation, thresholds) => identical instance masks, independent of the
    geometric gating (box/ROI) applied afterwards. Thresholds are rounded so
    float jitter never fragments the cache."""
    return (concept, rep_name, round(float(sam3_threshold), 4),
            round(float(mask_threshold), 4))


class SourceState:
    """Per-source resident state: the decoded BGR image, the enhanced detection
    image, its representations list (pelib.sam3._representations output), an
    optional depth map, and a detection cache keyed by detection_key(). Held in
    the worker's LRU so a nudged job on the same source reuses all of it."""

    def __init__(self, source_id: str):
        self.source_id = source_id
        self.bgr = None            # np.ndarray HxWx3 (original, BGR)
        self.det = None            # np.ndarray HxWx3 (enhanced-for-detection)
        self.representations = None  # [(name, bgr), ...] from _representations
        self.depth = None          # np.ndarray HxW uint8 (near=bright) or None
        self.depth_path = None     # str: which depth PNG `depth` was loaded from
        # detection cache: detection_key -> [(mask bool HxW, score float)]
        self.detections: dict[tuple, list] = {}

    def get_detections(self, concept, rep_name, sam3_threshold, mask_threshold):
        return self.detections.get(
            detection_key(concept, rep_name, sam3_threshold, mask_threshold))

    def put_detections(self, concept, rep_name, sam3_threshold, mask_threshold, value):
        self.detections[detection_key(
            concept, rep_name, sam3_threshold, mask_threshold)] = value


class SourceLRU:
    """Bounded LRU of SourceState (cap 2 sources by default). Pure-logic; no
    torch. Eviction drops the least-recently-used source's whole state (image +
    representations + depth + detection cache) so VRAM/host memory stays
    bounded across many distinct images."""

    def __init__(self, cap: int = 2):
        self.cap = cap
        self._map: "OrderedDict[str, SourceState]" = OrderedDict()

    def __len__(self):
        return len(self._map)

    def __contains__(self, source_id):
        return source_id in self._map

    def ids(self):
        return list(self._map.keys())

    def get(self, source_id: str) -> Optional[SourceState]:
        st = self._map.get(source_id)
        if st is not None:
            self._map.move_to_end(source_id)
        return st

    def get_or_create(self, source_id: str) -> SourceState:
        st = self._map.get(source_id)
        if st is None:
            st = SourceState(source_id)
            self._map[source_id] = st
        self._map.move_to_end(source_id)
        self._evict()
        return st

    def drop(self, source_id: str) -> None:
        self._map.pop(source_id, None)

    def _evict(self) -> list:
        """Evict from the front until within cap. Returns dropped source_ids."""
        dropped = []
        while len(self._map) > self.cap:
            sid, _ = self._map.popitem(last=False)
            dropped.append(sid)
        return dropped


# ---------------------------------------------------------------------------
# GPU pipeline (heavy imports guarded inside). Only reached at job time.
# ---------------------------------------------------------------------------

def _ensure_tools_dir(tools_dir: str) -> None:
    if tools_dir not in sys.path:
        sys.path.insert(0, tools_dir)


def _enhance_det(bgr, gamma=ENH_GAMMA, clahe=ENH_CLAHE, sharp=ENH_SHARP):
    """Replicate mask_c2f.py's enhance_for_detection + its auto-tune (median
    brightness -> gamma, contrast -> clahe). Returns the enhanced BGR used as
    the base detection image (representations are derived from the ORIGINAL bgr,
    matching mask_c2f, which passes `full`/`det` per pelib.sam3._representations
    -- note _representations itself always starts from the raw bgr it is given)."""
    import cv2
    import numpy as np
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    med = float(np.median(g))
    contrast = float(g.std())
    auto_gamma = float(np.clip(0.42 + med * 0.35, 0.42, gamma))
    auto_clahe = float(np.clip(clahe + (0.20 - contrast) * 12, 2.0, 6.0))

    def enhance_for_detection(bgr, clahe, gamma, sharp=0.6, sat=1.25):
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = cv2.createCLAHE(clipLimit=clahe, tileGridSize=(8, 8)).apply(l)
        out = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR).astype(np.float32) / 255.0
        out = np.power(np.clip(out, 0, 1), gamma)
        if sat != 1.0:
            hsv = cv2.cvtColor((out * 255).astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[..., 1] = np.clip(hsv[..., 1] * sat, 0, 255)
            out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32) / 255.0
        if sharp > 0:
            blur = cv2.GaussianBlur(out, (0, 0), 2.0)
            out = np.clip(out * (1 + sharp) - blur * sharp, 0, 1)
        return (out * 255).astype(np.uint8)

    return enhance_for_detection(bgr, auto_clahe, auto_gamma, sharp)


def _load_source_state(state: SourceState, image_path: str, tools_dir: str,
                       depth_path: str = "", multirep: bool = False) -> None:
    """Populate a SourceState's image/det/representations/depth if missing.
    Idempotent: only does the work not already cached. This is where the cold
    cost lives -- once populated, nudge jobs reuse it."""
    import cv2
    _ensure_tools_dir(tools_dir)
    if state.bgr is None:
        bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"cannot read image {image_path}")
        state.bgr = bgr
        state.det = _enhance_det(bgr)
    if multirep and state.representations is None:
        from pelib import sam3 as S3
        state.representations = S3._representations(state.bgr)
    # Depth is loaded (not computed) from the path the gateway passes -- the
    # gateway owns depth generation + its per-source cache. Load once per path.
    if depth_path and (state.depth is None or state.depth_path != depth_path):
        d = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)
        if d is not None:
            state.depth = d
            state.depth_path = depth_path


def _run_mask(state: SourceState, query: str, roi_png_path: str, box,
              carve: bool, multirep: bool, tools_dir: str, mode: str = "precise"):
    """The parity core. Mirrors mask_c2f.py's non-agentic SAM3 branch.
    Returns (out_mask uint8 HxW, coverage float, cached_detections bool)."""
    import cv2
    import numpy as np
    _ensure_tools_dir(tools_dir)
    from pelib import sam3 as S3

    full = state.bgr
    det = state.det
    H, W = full.shape[:2]

    # ROI: from an explicit ROI png, or synthesized from a box (region hint).
    roi = None
    if roi_png_path:
        r = cv2.imread(roi_png_path, cv2.IMREAD_UNCHANGED)
        if r is not None:
            if r.ndim == 3:
                r = r[..., 3] if r.shape[2] == 4 else cv2.cvtColor(r, cv2.COLOR_BGR2GRAY)
            roi = cv2.resize(r, (W, H), interpolation=cv2.INTER_NEAREST) > 127
    elif box is not None:
        roi = np.zeros((H, W), bool)
        x0, y0, x1, y1 = [int(round(v)) for v in box]
        x0, x1 = sorted((max(0, min(x0, W)), max(0, min(x1, W))))
        y0, y1 = sorted((max(0, min(y0, H)), max(0, min(y1, H))))
        if x1 > x0 and y1 > y0:
            roi[y0:y1, x0:x1] = True

    concepts = [c.strip() for c in query.split(".") if c.strip()] or [query.strip()]

    cached_all = True
    inst = []  # list of (bool mask, score)
    inst_labels = None

    if multirep:
        # Multi-rep: detect every (representation x concept), cache each combo.
        reps = state.representations
        pooled = []
        for name, v in reps:
            for c in concepts:
                cached = state.get_detections(c, name, SAM3_THRESHOLD, SAM3_MASK_THRESHOLD)
                if cached is None:
                    cached_all = False
                    try:
                        ms, ss = S3.instances(v, text=c, threshold=SAM3_THRESHOLD,
                                              mask_threshold=SAM3_MASK_THRESHOLD)
                        cached = list(zip([m.astype(bool) for m in ms],
                                          [float(s) for s in ss]))
                    except Exception as e:
                        logger.warning("multirep '%s'/'%s' failed: %s", name, c, str(e)[:80])
                        cached = []
                    state.put_detections(c, name, SAM3_THRESHOLD, SAM3_MASK_THRESHOLD, cached)
                pooled += cached
        inst = list(pooled)
    else:
        # Single-concept path (rep_name "det" = the enhanced detection image).
        for c in concepts:
            cached = state.get_detections(c, "det", SAM3_THRESHOLD, SAM3_MASK_THRESHOLD)
            if cached is None:
                cached_all = False
                try:
                    ms, ss = S3.instances(det, text=c, threshold=SAM3_THRESHOLD,
                                          mask_threshold=SAM3_MASK_THRESHOLD)
                    cached = list(zip([m.astype(bool) for m in ms],
                                      [float(s) for s in ss]))
                except Exception as e:
                    logger.warning("sam3 '%s' failed: %s", c, str(e)[:80])
                    cached = []
                state.put_detections(c, "det", SAM3_THRESHOLD, SAM3_MASK_THRESHOLD, cached)
            inst += cached

    # ROI gate: keep instances overlapping the ROI (mask_c2f semantics).
    if roi is not None and inst:
        kept = [(m, s) for (m, s) in inst if (m & roi).sum() > 0]
        if kept:
            inst = kept

    mask = np.zeros((H, W), np.uint8)
    Dc_pre = None
    if multirep and inst:
        raw = np.zeros((H, W), bool)
        for m, s in inst:
            raw |= m
        Dc_pre = _zoom_depth(full, raw, state, tools_dir)
        binm, inst_labels, kept = S3.combine_instances(inst, full, depth=Dc_pre)
        mask = binm.astype(np.uint8)
    else:
        for m, s in inst:
            mask = np.maximum(mask, m.astype(np.uint8))

    # DEPTH gate/complete (matches mask_c2f: only if depth_map provided).
    if state.depth is not None and mask.sum() > 500:
        D = state.depth
        if D.shape != (H, W):
            D = cv2.resize(D, (W, H), interpolation=cv2.INTER_LINEAR)
        D = D.astype(np.float32) / 255.0
        pd = float(np.median(D[mask > 0]))
        band = (np.abs(D - pd) < DEPTH_TOL).astype(np.uint8)
        if mode == "precise":
            mask = ((mask > 0) & (band > 0)).astype(np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        else:
            reach = 90
            region = cv2.dilate(mask, cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (2 * reach + 1, 2 * reach + 1)))
            if roi is not None:
                region = ((region > 0) | roi).astype(np.uint8)
            add = ((band > 0) & (region > 0) & (mask == 0)).astype(np.uint8)
            add = cv2.morphologyEx(add, cv2.MORPH_OPEN,
                                   cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
            mask = ((mask > 0) | (add > 0)).astype(np.uint8)

    # SAM3 path: no dilate, no birefnet, no matte, no protect (all sam2-gated).
    # Zoomed depth for carve / hole-fill / feather.
    want_depth = (carve or True) and mask.sum() > 500  # feather adaptive always wants Dc in precise
    Dc = Dc_pre if (Dc_pre is not None and Dc_pre[mask > 0].any()) else None
    if want_depth and Dc is None:
        Dc = _zoom_depth(full, mask > 0, state, tools_dir)

    # DEPTH-AWARE HOLE-FILL (sam3, not carving, not multirep).
    if (not carve) and (not multirep) and Dc is not None and mask.sum() > 500:
        d = Dc.astype(np.float32) / 255.0
        fg = float(np.percentile(d[mask > 0], 60))
        inv = (mask == 0).astype(np.uint8)
        nlab, lab = cv2.connectedComponents(inv, 8)
        border = set(np.unique(np.concatenate([lab[0, :], lab[-1, :], lab[:, 0], lab[:, -1]])))
        for c in range(1, nlab):
            if c in border:
                continue
            reg = (lab == c)
            if float(np.median(d[reg])) >= fg - 0.14:
                mask[reg] = 1

    if carve and mask.sum() > 500:
        before = 100 * mask.mean()
        if Dc is not None:
            mask = S3.carve_continuity((mask > 0), full, Dc).astype(np.uint8)
            method = "continuity"
        else:
            mask = S3.carve_seethrough((mask > 0), full).astype(np.uint8)
            method = "heuristic"
        logger.info("carve see-through (%s) %.1f%% -> %.1f%%", method, before, 100 * mask.mean())

    # OUTPUT border: precise gets adaptive depth/defocus soft alpha; else binary.
    if mode == "precise" and Dc is not None and mask.sum() > 500:
        from pelib.imaging import adaptive_feather
        alpha = adaptive_feather(full, mask * 255, depth=Dc, max_f=FEATHER_MAX)
        out_mask = np.clip(alpha * 255, 0, 255).astype(np.uint8)
    else:
        out_mask = (mask * 255).astype(np.uint8)

    coverage = float(mask.mean())
    return out_mask, coverage, cached_all


def _zoom_depth(full, mask_bool, state: SourceState, tools_dir: str):
    """Zoomed depth on the mask bbox, mirroring mask_c2f.zoom_depth. Uses the
    same pelib.depth path; falls back to the loaded full-frame depth map."""
    import cv2
    import numpy as np
    _ensure_tools_dir(tools_dir)
    H, W = full.shape[:2]
    mys, mxs = np.where(mask_bool)
    if not len(mxs):
        return None
    pad = 60
    zy0, zy1 = max(0, mys.min() - pad), min(H, mys.max() + pad)
    zx0, zx1 = max(0, mxs.min() - pad), min(W, mxs.max() + pad)
    crop = full[zy0:zy1, zx0:zx1]
    from pelib import depth as PD
    dz = None
    try:
        dz = PD.depth_pro(crop)
    except Exception as e:
        logger.info("depth_pro failed, DA-V2 fallback: %s", str(e)[:60])
    if dz is None:
        try:
            dz = PD.depth_map(crop, res=1024)
            if dz.ndim == 3:
                dz = cv2.cvtColor(dz, cv2.COLOR_BGR2GRAY)
        except Exception as e:
            logger.info("zoom-depth failed: %s", str(e)[:60])
            if state.depth is not None:
                dz = state.depth[zy0:zy1, zx0:zx1]
    if dz is None:
        return None
    Dc = np.zeros((H, W), np.uint8)
    Dc[zy0:zy1, zx0:zx1] = cv2.resize(dz, (zx1 - zx0, zy1 - zy0))
    return Dc


# ---------------------------------------------------------------------------
# HTTP server (aiohttp). Kept thin; all logic above.
# ---------------------------------------------------------------------------

class Worker:
    def __init__(self, tools_dir: str, cap: int = 2):
        self.tools_dir = tools_dir
        self.sources = SourceLRU(cap=cap)
        self.models_loaded = False
        self._lock = None  # asyncio.Lock, created in run()

    def _process(self, req: dict) -> dict:
        """Synchronous job processing (runs off the event loop via executor).
        On CUDA OOM, drop the source state and retry once."""
        import cv2  # noqa: F401 (ensures cv2 present early for a clean error)
        image_path = req["image_path"]
        source_id = req["source_id"]
        query = req["query"]
        roi_png_path = req.get("_roi_png_path", "")
        box = req.get("box")
        carve = bool(req.get("carve"))
        multirep = bool(req.get("multirep"))
        depth_path = req.get("depth_path", "") or ""
        out_path = req["out_path"]
        mode = req.get("mode", "precise")

        def attempt():
            state = self.sources.get_or_create(source_id)
            _load_source_state(state, image_path, self.tools_dir,
                               depth_path=depth_path, multirep=multirep)
            self.models_loaded = True
            return _run_mask(state, query, roi_png_path, box, carve, multirep,
                             self.tools_dir, mode=mode)

        t0 = time.perf_counter()
        try:
            out_mask, coverage, cached = attempt()
        except Exception as e:
            if _is_oom(e):
                logger.warning("CUDA OOM; dropping source %s and retrying once", source_id)
                self.sources.drop(source_id)
                _empty_cache()
                out_mask, coverage, cached = attempt()
            else:
                raise
        finally:
            _empty_cache()

        import cv2
        cv2.imwrite(out_path, out_mask)
        return {
            "ok": True,
            "out_path": out_path,
            "coverage": round(coverage, 6),
            "cached_detections": bool(cached),
            "timings": {"total_s": round(time.perf_counter() - t0, 2)},
        }

    async def handle_mask(self, request):
        from aiohttp import web
        try:
            req = await request.json()
        except Exception as e:
            return web.json_response({"ok": False, "error": f"bad json: {e}"}, status=400)
        # Materialize an inline roi_png_b64 to a temp file for the sync core.
        roi_b64 = req.get("roi_png_b64")
        tmp_roi = None
        if roi_b64:
            import base64
            import tempfile
            tmp_roi = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            tmp_roi.write(base64.b64decode(roi_b64))
            tmp_roi.close()
            req["_roi_png_path"] = tmp_roi.name
        loop = __import__("asyncio").get_event_loop()
        async with self._lock:  # serialize GPU jobs; state is not thread-safe
            try:
                result = await loop.run_in_executor(None, self._process, req)
                return web.json_response(result)
            except Exception as e:
                logger.error("mask job failed: %s\n%s", e, traceback.format_exc(limit=6))
                return web.json_response(
                    {"ok": False, "error": str(e)[:600],
                     "oom": _is_oom(e)}, status=500)
            finally:
                if tmp_roi is not None:
                    try:
                        os.unlink(tmp_roi.name)
                    except OSError:
                        pass

    async def handle_health(self, request):
        from aiohttp import web
        return web.json_response({
            "ok": True,
            "loaded_sources": self.sources.ids(),
            "models_loaded": self.models_loaded,
        })

    def run(self, host: str, port: int):
        import asyncio
        from aiohttp import web
        self._lock = asyncio.Lock()
        app = web.Application(client_max_size=64 * 1024 * 1024)
        app.router.add_post("/mask", self.handle_mask)
        app.router.add_get("/health", self.handle_health)
        logger.info("mask_worker listening on %s:%d (tools_dir=%s)", host, port, self.tools_dir)
        web.run_app(app, host=host, port=port, print=None)


def _is_oom(e: Exception) -> bool:
    s = (str(e) or "").lower()
    return "out of memory" in s or "outofmemoryerror" in s or type(e).__name__ == "OutOfMemoryError"


def _empty_cache() -> None:
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5101)
    ap.add_argument("--tools-dir", default=os.path.expanduser("~/comfy"))
    ap.add_argument("--cap", type=int, default=2)
    a = ap.parse_args()
    Worker(tools_dir=a.tools_dir, cap=a.cap).run(a.host, a.port)


if __name__ == "__main__":
    main()
