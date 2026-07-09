"""pelib.imaging - mask IO, alignment, mask-context cropping, feather compositing."""
import io
import numpy as np
import cv2


def align8(v):
    return (int(v) // 8) * 8


def load_mask(src, H, W):
    """Load a mask from a path OR raw bytes -> single-channel uint8 at (H,W). Handles 3ch/4ch(alpha)/gray."""
    if isinstance(src, (bytes, bytearray)):
        from PIL import Image
        m = np.array(Image.open(io.BytesIO(src)))
    else:
        m = cv2.imread(src, cv2.IMREAD_UNCHANGED)
    if m is None:
        return None
    if m.ndim == 3:
        m = m[..., 3] if m.shape[2] == 4 else cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)
    if m.shape != (H, W):
        m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
    return m


def grow(mask, px):
    if px <= 0:
        return mask
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px + 1, 2 * px + 1))
    return cv2.dilate((mask > 127).astype(np.uint8) * 255, k)


def context_box(mask, ctx=0.4):
    """Bounding box of the mask expanded by `ctx` fraction of its size, /8-aligned. None if empty."""
    H, W = mask.shape[:2]
    ys, xs = np.where(mask > 127)
    if len(xs) == 0:
        return None
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    bw, bh = x1 - x0, y1 - y0
    px, py = int(bw * ctx), int(bh * ctx)
    cx0, cy0 = max(0, x0 - px), max(0, y0 - py)
    cx1, cy1 = min(W, x1 + px), min(H, y1 + py)
    cx1 = cx0 + align8(cx1 - cx0)
    cy1 = cy0 + align8(cy1 - cy0)
    return (cx0, cy0, cx1, cy1)


def fit_long(img, side, interp=cv2.INTER_AREA):
    """Resize so the long side <= `side`, /8-aligned. Returns (resized, scale)."""
    h, w = img.shape[:2]
    s = min(1.0, side / max(h, w))
    return cv2.resize(img, (align8(w * s), align8(h * s)), interpolation=interp), s


def adaptive_feather(image_bgr, mask, depth=None, min_f=3.0, max_f=40.0,
                     levels=(2, 6, 14, 26, 44)):
    """Per-pixel feather alpha from local blur + depth edges.
    Feather WIDTH grows where the image is out-of-focus (bokeh) and stays TIGHT where it's sharp
    (in-focus edges) or across a depth discontinuity (an occluder in front of the masked area).
    Returns a float alpha [0,1] the same size as mask."""
    H, W = mask.shape[:2]
    mb = (mask > 127).astype(np.float32)
    g = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    lap = np.abs(cv2.Laplacian(g, cv2.CV_32F, ksize=3))
    S = cv2.GaussianBlur(lap, (0, 0), 12)
    S = np.clip(S / (np.percentile(S, 90) + 1e-3), 0, 1)          # 0=blurred .. 1=sharp
    blur = 1.0 - S
    # DARK != out-of-focus: a dark low-contrast edge reads as "blurred" but is actually sharp. Only widen
    # the feather where there's real bokeh SIGNAL (some brightness/local texture); in shadow, stay tight.
    lum = cv2.GaussianBlur(g, (0, 0), 12) / 255.0
    signal = np.clip((lum - 0.10) / 0.35, 0.0, 1.0)              # ~0 in deep shadow -> suppresses widening
    blur = blur * signal
    fw = min_f + blur * (max_f - min_f)                          # feather width per pixel
    if depth is not None:
        d = depth.astype(np.float32)
        if d.shape[:2] != (H, W):
            d = cv2.resize(d, (W, H))
        dg = np.abs(cv2.Laplacian(cv2.GaussianBlur(d, (0, 0), 3), cv2.CV_32F))
        dg = np.clip(dg / (np.percentile(dg, 95) + 1e-3), 0, 1)   # 1 at occluder/depth edges
        fw = fw * (1 - dg) + min_f * dg                          # force tight across depth edges
    # blend blurred-mask levels: per-pixel linear interp between the two bracketing feather widths
    lv = np.array(levels, np.float32)
    blurs = [cv2.GaussianBlur(mb, (0, 0), max(0.6, L / 2.0)) for L in lv]
    idx = np.clip(np.searchsorted(lv, fw) - 1, 0, len(lv) - 2)
    lo = lv[idx]; hi = lv[idx + 1]
    t = np.clip((fw - lo) / np.maximum(hi - lo, 1e-3), 0, 1)
    stack = np.stack(blurs, -1)                                   # H,W,levels
    a_lo = np.take_along_axis(stack, idx[..., None], -1)[..., 0]
    a_hi = np.take_along_axis(stack, (idx + 1)[..., None], -1)[..., 0]
    return np.clip(a_lo * (1 - t) + a_hi * t, 0, 1)


def guided_edge_snap(image_bgr, mask, radius=8, eps=1e-4, pre_blur=1.2, band=0):
    """Snap a coarse mask's alpha to real image edges via a guided filter ("guided feathering",
    He et al.). The image luminance guides the filter, so the alpha boundary follows actual
    intensity edges instead of the blocky/eroded segmentation border -> recovers thin structures
    (spokes, poles, hair) that overlap busy backgrounds and removes matte-line halos.
    `mask` may be binary (0/255) or a soft alpha (0..255). Returns float alpha [0,1] at mask size.
    radius is in pixels of the guidance edge neighbourhood; eps (on normalized [0,1]) sets how
    strongly it snaps (smaller = sharper/edge-hugging).
    BAND-LIMITED (trimap-style): the filter over-smooths a solid interior into translucency, so we only
    let it rewrite a ring `band` px around the original boundary — confident interior stays alpha 1,
    confident exterior stays 0, and the true soft edge is resolved inside the ring. band<=0 -> auto (3*radius)."""
    H, W = mask.shape[:2]
    guide = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    if guide.shape[:2] != (H, W):
        guide = cv2.resize(guide, (W, H))
    src = mask.astype(np.float32) / 255.0
    if pre_blur > 0:                                             # soft seed -> the filter has a gradient to pull to edges
        src = cv2.GaussianBlur(src, (0, 0), pre_blur)
    if hasattr(cv2, "ximgproc"):
        a = cv2.ximgproc.guidedFilter(guide=guide, src=src, radius=int(radius), eps=float(eps))
    else:                                                        # fallback: edge-aware bilateral
        a = cv2.bilateralFilter(src, d=0, sigmaColor=0.1, sigmaSpace=float(radius))
    a = np.clip(a, 0.0, 1.0)
    b = int(band) if band and band > 0 else max(6, 3 * int(radius))
    binm = (mask > 127).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * b + 1, 2 * b + 1))
    core = cv2.erode(binm, k)                                    # deep interior -> force solid
    outer = cv2.dilate(binm, k)                                  # beyond the ring -> force empty
    a[core > 0] = 1.0
    a[outer == 0] = 0.0
    return a


def feather_paste(base, patch, mask, box, feather=12, alpha=None):
    """Composite `patch` into `base` inside `box`, blended by a feathered `mask` (full-frame, at box)."""
    cx0, cy0, cx1, cy1 = box
    crop = base[cy0:cy1, cx0:cx1].astype(np.float32)
    if patch.shape[:2] != crop.shape[:2]:
        patch = cv2.resize(patch, (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_CUBIC)
    if alpha is not None:                                        # precomputed adaptive feather (full-frame)
        fm = alpha[cy0:cy1, cx0:cx1][..., None].astype(np.float32)
    else:
        fm = cv2.GaussianBlur((mask[cy0:cy1, cx0:cx1] > 127).astype(np.float32) * 255, (0, 0), feather)[..., None] / 255.0
    comp = (crop * (1 - fm) + patch.astype(np.float32) * fm).astype(np.uint8)
    out = base.copy()
    out[cy0:cy1, cx0:cx1] = comp
    return out
