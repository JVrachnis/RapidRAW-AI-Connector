"""pelib.matte - ViTMatte trimap alpha refine of a coarse mask (bbox-crop so full-res fits VRAM)."""
import os
import numpy as np
import cv2

_MODEL = None
_PROC = None
_DEV = None


def _pick_device():
    """Pick the CUDA device with the most free VRAM (ViTMatte's ViTDet attention is memory-heavy; GPU0
    is usually crowded by the resident LLM, so this lands us on the free P100)."""
    import torch
    if not torch.cuda.is_available():
        return "cpu"
    free = []
    for i in range(torch.cuda.device_count()):
        f, _ = torch.cuda.mem_get_info(i)
        free.append((f, i))
    return f"cuda:{max(free)[1]}"


def _load(model_dir, device=None):
    global _MODEL, _PROC, _DEV
    if _MODEL is None:
        import torch
        from transformers import VitMatteForImageMatting, VitMatteImageProcessor
        _DEV = device or _pick_device()
        _PROC = VitMatteImageProcessor.from_pretrained(model_dir)
        _MODEL = VitMatteForImageMatting.from_pretrained(model_dir, torch_dtype=torch.float32).to(_DEV).eval()
    return _MODEL, _PROC


def vitmatte(image_rgb, coarse01, model_dir=os.path.expanduser("~/models/vitmatte-base"),
             band=20, maxside=2048, firm=1.25, defringe=True):
    """Return a full-res alpha (float 0..1) for the coarse mask, refined with ViTMatte on the subject bbox."""
    import torch
    from PIL import Image
    H, W = image_rgb.shape[:2]
    mb = (coarse01 > 0.5).astype(np.uint8)
    ys, xs = np.where(mb > 0)
    if len(xs) == 0:
        return coarse01
    pad = 64
    x0, x1 = max(0, xs.min() - pad), min(W, xs.max() + pad)
    y0, y1 = max(0, ys.min() - pad), min(H, ys.max() + pad)
    crop = image_rgb[y0:y1, x0:x1]
    cm = (coarse01[y0:y1, x0:x1] * 255).astype(np.uint8)
    ch, cw = crop.shape[:2]
    s = min(1.0, maxside / max(ch, cw))
    if s < 1:
        crop_s = cv2.resize(crop, (int(cw * s), int(ch * s)), cv2.INTER_AREA)
        cm_s = cv2.resize(cm, (int(cw * s), int(ch * s)), cv2.INTER_LINEAR)
    else:
        crop_s, cm_s = crop, cm
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * band + 1, 2 * band + 1))
    cmb = (cm_s > 128).astype(np.uint8)
    fg = cv2.erode(cmb, k); bgo = cv2.dilate(cmb, k)
    tri = np.full(cm_s.shape, 0.5, np.float32); tri[fg > 0] = 1.0; tri[bgo == 0] = 0.0
    model, proc = _load(model_dir)
    inp = proc(images=Image.fromarray(crop_s), trimaps=Image.fromarray((tri * 255).astype(np.uint8)),
               return_tensors="pt")
    inp = {kk: v.to(_DEV) for kk, v in inp.items()}
    with torch.no_grad():
        alpha = model(**inp).alphas[0, 0].float().cpu().numpy()
    alpha = alpha[:crop_s.shape[0], :crop_s.shape[1]]
    if defringe:
        lum = (crop_s.astype(np.float32) @ np.array([0.2126, 0.7152, 0.0722], np.float32)) / 255.0
        core = alpha > 0.9
        if core.sum() > 50:
            fl = float(np.median(lum[core])); edge = (alpha > 0.03) & (alpha < 0.97)
            supp = 1.0 - np.clip(np.clip(lum - (fl + 0.12), 0, 1) * 4.0, 0, 0.85)
            alpha = np.where(edge, alpha * supp, alpha)
    alpha = np.clip((alpha - 0.5) * firm + 0.5, 0, 1)
    if s < 1:
        alpha = cv2.resize(alpha, (cw, ch), cv2.INTER_LINEAR)
    out = np.zeros((H, W), np.float32)
    out[y0:y1, x0:x1] = np.clip(alpha, 0, 1)
    return out


def _matte_tile(model, proc, tile_rgb, tri):
    """One ViTMatte forward on a small native-res tile (rgb + trimap 0/0.5/1). Returns alpha [0,1]."""
    import torch
    from PIL import Image
    inp = proc(images=Image.fromarray(tile_rgb), trimaps=Image.fromarray((tri * 255).astype(np.uint8)),
               return_tensors="pt")
    inp = {k: v.to(_DEV) for k, v in inp.items()}
    with torch.no_grad():
        a = model(**inp).alphas[0, 0].float().cpu().numpy()
    return a[:tile_rgb.shape[0], :tile_rgb.shape[1]]


def _hull_trimap(coarse01, fg_erode=21, hull_dilate=0):
    """Trimap where UNKNOWN = the convex hull of each mask blob (so a wheel's whole disk — where the
    thin spokes live — is unknown and gets carved), FG = eroded blob core, BG = outside the hull.
    This is the key interconnection: the hull turns 'ring' detections into 'fill the interior detail'."""
    mb = (coarse01 > 0.5).astype(np.uint8)
    n, lbl, st, _ = cv2.connectedComponentsWithStats(mb, 8)
    hull = np.zeros_like(mb)
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] < 200:
            continue
        cnts, _ = cv2.findContours((lbl == i).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            cv2.fillConvexPoly(hull, cv2.convexHull(np.vstack(cnts)), 1)
    if hull_dilate > 0:
        hull = cv2.dilate(hull, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*hull_dilate+1,)*2))
    fg = cv2.erode(mb, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*fg_erode+1,)*2))
    tri = np.zeros(mb.shape, np.float32)
    tri[hull > 0] = 0.5
    tri[fg > 0] = 1.0
    return tri, hull


def depth_edge_trimap(coarse01, bgr, depth, reach=70, depth_tol=0.12, band=12,
                      grad_lo=0.14, dark_thr=8, sky_lo=0.55):
    """Fuse mask + DEPTH + EDGES into a trimap. The interconnection that makes matting carve thin
    structures instead of filling: SKY/background is flat + far + edgeless -> confident BG; a spoke is
    a thin edge AT THE OBJECT'S DEPTH -> UNKNOWN (ViTMatte then decides it FG from image evidence).
      FG      = eroded mask core.
      UNKNOWN = (boundary band)  OR  (near the mask  AND  on a strong edge / dark thin line  AND
                 at the object's depth) -> boundary + depth-consistent thin structures.
      BG      = everything else (far-depth flat sky, background edges away from the object).
    `depth` = uint8/float near=bright (Depth Pro disparity). Returns trimap float {0,0.5,1}."""
    H, W = coarse01.shape[:2]
    mb = (coarse01 > 0.5).astype(np.uint8)
    d = depth.astype(np.float32); d = (d - d.min()) / (np.ptp(d) + 1e-6)   # near->1
    if d.shape[:2] != (H, W):
        d = cv2.resize(d, (W, H))
    k = lambda r: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*r+1, 2*r+1))
    # object depth from the NEAR cluster under the mask (a solid-filled wheel mask mixes near tire + far
    # sky, so a plain median is dragged toward sky -> use a high percentile = the near/object plane)
    obj_d = float(np.percentile(d[mb > 0], 65)) if mb.sum() else 1.0
    at_obj = (np.abs(d - obj_d) < depth_tol).astype(np.uint8)   # same depth plane as the object
    far = ((obj_d - d) > depth_tol).astype(np.uint8)            # farther than the object (sky / background behind)
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    grad = np.abs(cv2.Sobel(g.astype(np.float32), cv2.CV_32F, 1, 0)) + \
           np.abs(cv2.Sobel(g.astype(np.float32), cv2.CV_32F, 0, 1))
    grad = grad / (np.percentile(grad, 97) + 1e-3)
    blackhat = cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, k(3))     # thin DARK lines (spokes/wire on bright bg)
    edge = ((grad > grad_lo) | (blackhat > dark_thr)).astype(np.uint8)
    # CONFIDENT BG = flat (no edge) background, by EITHER signal:
    #  - depth 'far'  -> catches genuine depth discontinuities (object in front of a distant backdrop), OR
    #  - bright luma  -> catches a uniform BRIGHT backdrop (sky) that monocular depth wrongly FILLS across
    #    spoke gaps (Depth Pro renders a spoked wheel as a solid near disk, so depth alone can't carve it).
    # This OVERRIDES the coarse mask, carving back out the sky it wrongly filled inside a solid wheel disk.
    near0 = cv2.dilate(mb, k(reach))                           # local region for an ADAPTIVE bright/dark split
    if near0.sum() > 200:                                      # Otsu: the raw frame is dark (8-bit ceiling), so a
        t, _ = cv2.threshold(g[near0 > 0], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # FIXED thr fails; use relative
    else:
        t = 255 * sky_lo
    sky_bg = (g > max(t, 255 * 0.12)).astype(np.uint8)         # brighter-than-object backdrop (sky between spokes)
    confident_bg = ((far | sky_bg) & (1 - edge)).astype(np.uint8)
    confident_bg = cv2.morphologyEx(confident_bg, cv2.MORPH_OPEN, k(2))   # drop speckle
    fg = (cv2.erode(mb, k(band)) & (1 - confident_bg)).astype(np.uint8)   # object core minus leaked background
    if fg.sum() < 50:
        fg = (mb & (1 - confident_bg)).astype(np.uint8)
    near = cv2.dilate(mb, k(reach))
    bandmask = cv2.dilate(mb, k(band)) & (1 - cv2.erode(mb, k(band)))
    thin = (near & edge & at_obj)                               # depth-consistent thin structures (spokes) near obj
    unknown = ((bandmask | thin) & (1 - fg) & (1 - confident_bg)).astype(np.uint8)
    tri = np.zeros((H, W), np.float32)                          # default BG (incl. confident_bg / carved sky)
    tri[unknown > 0] = 0.5
    tri[fg > 0] = 1.0
    return tri


def zoom_matte(image_rgb, coarse01, trimap=None, model_dir=os.path.expanduser("~/models/vitmatte-base"),
               tile=640, overlap=128, fg_erode=21, hull_dilate=8, firm=1.0):
    """ViTMatte at NATIVE resolution over small overlapping tiles (the 'zoom' trick), so thin structures
    (wheel spokes, wire, hair) are resolved full-res without ViTDet's O(tokens^2) attention OOM.
    `trimap` (float {0,0.5,1}) drives it — pass depth_edge_trimap(...) to carve depth+edge-consistent
    thin structures; if None, falls back to the convex-HULL trimap (fills interiors — good for removal).
    Tiles with no unknown pixels are skipped. Overlaps Hann-blended. Returns (alpha[0,1], n_tiles)."""
    H, W = image_rgb.shape[:2]
    if trimap is None:
        trimap, _ = _hull_trimap(coarse01, fg_erode=fg_erode, hull_dilate=hull_dilate)
    unknown = (trimap == 0.5)
    out = (trimap >= 1.0).astype(np.float32)                    # fg core = 1, everything else 0 (bg)
    ys, xs = np.where(unknown)
    if len(xs) == 0:
        return out, 0
    model, proc = _load(model_dir)
    bx0, by0, bx1, by1 = xs.min(), ys.min(), xs.max()+1, ys.max()+1
    acc = np.zeros((H, W), np.float32); wsum = np.zeros((H, W), np.float32)
    win = np.outer(np.hanning(tile), np.hanning(tile)).astype(np.float32) + 1e-3
    step = tile - overlap
    ntiles = 0
    for ty in range(by0, by1, step):
        for tx in range(bx0, bx1, step):
            y0 = min(ty, H - tile) if H >= tile else 0
            x0 = min(tx, W - tile) if W >= tile else 0
            y1, x1 = min(H, y0 + tile), min(W, x0 + tile)
            tri_t = trimap[y0:y1, x0:x1]
            if not (tri_t == 0.5).any():
                continue
            th, tw = y1 - y0, x1 - x0
            a = _matte_tile(model, proc, image_rgb[y0:y1, x0:x1], tri_t)
            w = win[:th, :tw]
            acc[y0:y1, x0:x1] += a * w
            wsum[y0:y1, x0:x1] += w
            ntiles += 1
    m = wsum > 0
    ref = np.zeros((H, W), np.float32); ref[m] = acc[m] / wsum[m]
    out = np.where(unknown, ref, out)                           # unknown -> refined; fg stays 1; bg stays 0
    out[trimap >= 1.0] = 1.0
    out = np.clip((out - 0.5) * firm + 0.5, 0, 1)
    return out, ntiles
