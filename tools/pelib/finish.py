"""pelib.finish - make a fill/generation look SHOT: grain + depth-bokeh + harmonize + vignette.

Two modes, auto-selected:
  MATCH (mask + original given)  - measure real grain sigma + sharpness in a ring around the fill and
                                   match them; optional harmonize (pull exposure/colour to ambient).
  PARAMETRIC (no original)       - apply grain/vignette/bokeh by parameters (for whole-image generate).
All operate on BGR uint8 numpy arrays. Shared by exif_finish.py (CLI) and the ComfyUI RealismFinish node.
"""
import numpy as np
import cv2


def _lum_bgr(x):
    return x @ np.array([0.114, 0.587, 0.299], np.float32)


def finish(bgr, mask=None, original=None, depth=None, iso=400.0, fnumber=4.0,
           grain=1.0, bokeh=1.0, harmonize=False, harmonize_strength=0.6, vignette=0.0, feather=12):
    """Return a finished BGR uint8 image."""
    fill = bgr.astype(np.float32)
    H, W = fill.shape[:2]
    iso_scale = float(np.clip(iso / 400.0, 0.6, 3.0))

    if mask is not None:
        mb = (cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST) > 127).astype(np.uint8)
    else:
        mb = np.ones((H, W), np.uint8)
    core = mb > 0

    # reference ring (real pixels around the fill) or the whole frame in parametric mode
    if original is not None and mask is not None:
        orig = original.astype(np.float32)
        if orig.shape[:2] != (H, W):
            orig = cv2.resize(orig, (W, H))
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (61, 61))
        ring = (cv2.dilate(mb, k) - mb).astype(bool)
        if ring.sum() < 500:
            ring = ~core
    else:
        orig, ring = None, None

    out = fill.copy()

    # ---- harmonize (placed objects): pull exposure + colour toward ambient ----
    if harmonize and orig is not None and core.sum() > 200 and ring.sum() > 200:
        s = float(np.clip(harmonize_strength, 0, 1))
        r_l, f_l = float(np.median(_lum_bgr(orig[ring]))), float(np.median(_lum_bgr(out[core])))
        scale = float(np.clip((s * r_l + (1 - s) * f_l) / max(f_l, 1.0), 0.2, 1.3))
        r_mean, f_mean = orig[ring].reshape(-1, 3).mean(0), out[core].reshape(-1, 3).mean(0)
        wb = 1 + s * 0.5 * (r_mean / np.maximum(f_mean, 1.0) - 1)
        out = np.where(core[..., None], out * scale * wb[None, None, :], out)

    # ---- bokeh: soften fill to match surroundings (measured) or by aperture (parametric) ----
    if orig is not None:
        def hf(img, m):
            e = np.abs(cv2.Laplacian(cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32), cv2.CV_32F))
            return float(np.median(e[m]))
        ring_sharp, fill_sharp = hf(orig, ring), hf(out, core) + 1e-3
        soft = np.clip((fill_sharp / max(ring_sharp, 1.0) - 1.0), 0, 4.0) * 0.6 * bokeh
        base_sigma = soft * float(np.clip(2.8 / max(fnumber, 1.0), 0.6, 1.6))
    else:
        base_sigma = bokeh * float(np.clip(2.8 / max(fnumber, 1.0), 0.0, 1.6)) if bokeh > 0 else 0.0

    if base_sigma > 0.3:
        if depth is not None:
            D = cv2.resize(depth, (W, H)).astype(np.float32) / 255.0 if depth.shape[:2] != (H, W) else depth.astype(np.float32) / 255.0
            fd = float(np.median(D[ring])) if ring is not None else float(np.median(D))
            dw = np.clip(np.abs(D - fd) / 0.4, 0.25, 1.0)
        else:
            dw = np.ones((H, W), np.float32)
        for sg, lo, hi in [(base_sigma * 0.5, 0.0, 0.4), (base_sigma, 0.4, 0.7), (base_sigma * 1.8, 0.7, 1.01)]:
            if sg < 0.3:
                continue
            b = cv2.GaussianBlur(out, (0, 0), sg)
            w = cv2.GaussianBlur(((dw >= lo) & (dw < hi)).astype(np.float32), (0, 0), 8)[..., None]
            out = out * (1 - w) + b * w

    # ---- grain: match measured ring noise, or default sigma, scaled by ISO ----
    if grain > 0:
        if orig is not None:
            ohp = orig - cv2.GaussianBlur(orig, (0, 0), 1.4)
            sig = np.array([ohp[..., c][ring].std() for c in range(3)])
        else:
            sig = np.array([2.2, 2.2, 2.4], np.float32)
        rng = np.random.default_rng(0)
        g = (rng.standard_normal((H, W, 1)).astype(np.float32) * 0.8 +
             rng.standard_normal((H, W, 3)).astype(np.float32) * 0.35) * sig[None, None, :] * iso_scale * grain
        out = out + g

    # ---- vignette (parametric, whole frame) ----
    if vignette > 0:
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        r = np.sqrt(((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2)
        vig = (1 - np.clip((r - 0.6) / 0.6, 0, 1) * vignette)[..., None]
        out = out * vig

    # composite finished region back (feathered) if masked
    if mask is not None and not core.all():
        fm = cv2.GaussianBlur(mb.astype(np.float32) * 255, (0, 0), feather)[..., None] / 255.0
        out = fill * (1 - fm) + out * fm
    return np.clip(out, 0, 255).astype(np.uint8)
