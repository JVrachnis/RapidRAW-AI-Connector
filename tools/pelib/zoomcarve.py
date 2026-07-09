"""pelib.zoomcarve - recover thin lattice structure (spokes/mesh/railings) that SAM3 fills solid.

Reuses the pipeline's zoom trick + Depth Pro: a coarse mask often FILLS a spoked wheel / wire mesh into
a solid blob (and monocular depth fills it too, IF run on the whole subject -- it downsamples and the
gaps blur shut). The fix, per component:
  1. find_carve_regions: locate lattice regions by INTERNAL edge-density (spokes = dense edges) + enclosed
     holes, per connected component so scale is local. Tight box per hotspot.
  2. zoom_carve: tight-crop each region -> Depth Pro at native res (resolves the gaps) -> SOFT depth matte
     (sigmoid between the sky-far and object-near depth clusters under the mask) -> composite back.
The soft matte keeps every near structure (spokes, laces, tire) and fades only the far background.
"""
import os
import numpy as np
import cv2


def _lifted_gray(bgr):
    return cv2.convertScaleAbs(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), alpha=2.0, beta=10)


def find_carve_regions(mask, bgr, min_frac=0.02, max_regions=8):
    """Return tight ROIs [ (x0,y0,x1,y1,tag) ] worth zoom-carving. Detected PER mask component at local
    scale: lattice = high internal-edge-density (spokes/mesh); hole = enclosed background."""
    H, W = mask.shape[:2]
    mb = (mask > 127).astype(np.uint8)
    gray = _lifted_gray(bgr)
    edges = cv2.Canny(gray, 60, 160)
    n, lbl, st, _ = cv2.connectedComponentsWithStats(mb, 8)
    lattice, holes = [], []
    for i in range(1, n):
        area = st[i, cv2.CC_STAT_AREA]
        if area < min_frac * mb.sum():
            continue
        x, y, w, h = (st[i, cv2.CC_STAT_LEFT], st[i, cv2.CC_STAT_TOP],
                      st[i, cv2.CC_STAT_WIDTH], st[i, cv2.CC_STAT_HEIGHT])
        span = max(w, h)
        cc = (lbl == i).astype(np.uint8)
        # (a) lattice via internal edge-density at local scale
        inside = ((edges > 0) & (cc > 0)).astype(np.float32)
        kd = max(11, int(0.05 * span)) | 1
        dens = cv2.boxFilter(inside, -1, (kd, kd)) * (cc > 0)
        if (cc > 0).sum() > 100:
            dm = dens[cc > 0]
            thr = max(0.06, float(np.median(dm)) + 2.0 * float(dm.std()))   # hotspot = well above local median
            thr = min(thr, float(np.percentile(dm, 90)))                    # but never above the 90th pct (always fires)
            hi = cv2.morphologyEx(((dens > thr) & (cc > 0)).astype(np.uint8),
                                  cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
            nh, lh, sh, _ = cv2.connectedComponentsWithStats(hi, 8)
            for j in range(1, nh):
                a = sh[j, cv2.CC_STAT_AREA]
                if a < 0.006 * area:
                    continue
                bx, by, bw, bh = sh[j, 1], sh[j, 2], sh[j, 3], sh[j, 4]
                r = int(max(bw, bh) * 0.70)                       # snug square around the hotspot
                cx, cy = bx + bw // 2, by + bh // 2
                lattice.append([max(0, cx-r), max(0, cy-r), min(W, cx+r), min(H, cy+r), "lattice"])
        # (b) enclosed holes (bg the mask surrounds)
        sub = cc[y:y+h, x:x+w]
        ff = sub.copy(); m2 = np.zeros((h+2, w+2), np.uint8)
        cv2.floodFill(ff, m2, (0, 0), 1)
        hl = ((ff == 0) & (sub == 0)).astype(np.uint8)
        nh, lh, sh, _ = cv2.connectedComponentsWithStats(hl, 8)
        for j in range(1, nh):
            if sh[j, cv2.CC_STAT_AREA] < 0.006 * area:
                continue
            bx, by, bw, bh = sh[j, 1], sh[j, 2], sh[j, 3], sh[j, 4]
            r = int(max(bw, bh) * 0.75)
            cx, cy = x + bx + bw // 2, y + by + bh // 2
            holes.append([max(0, cx-r), max(0, cy-r), min(W, cx+r), min(H, cy+r), "hole"])

    def overlaps(a, b, frac=0.4):
        ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
        return ix * iy > frac * min((a[2]-a[0])*(a[3]-a[1]), (b[2]-b[0])*(b[3]-b[1]))

    kept = []
    for r in lattice:                                            # lattice first (specific), then de-dup
        if not any(overlaps(r, k) for k in kept):
            kept.append(r)
    for r in holes:                                              # holes only where no lattice covers them
        if not any(overlaps(r, k) for k in kept):
            kept.append(r)
    kept.sort(key=lambda r: -(r[2]-r[0])*(r[3]-r[1]))
    return kept[:max_regions]


def soft_depth_matte(crop_bgr, cc_mask01, depth, near_pct=65, far_pct=12, firm=1.15):
    """Soft alpha from the zoomed depth: sigmoid between the far (sky) and near (object) depth clusters
    UNDER the mask. Keeps near thin structure, fades far background. Returns alpha[0,1] at crop size."""
    d = depth.astype(np.float32)
    d = (d - d.min()) / (np.ptp(d) + 1e-6)                       # near -> 1
    if d.shape[:2] != cc_mask01.shape[:2]:
        d = cv2.resize(d, (cc_mask01.shape[1], cc_mask01.shape[0]))
    sel = cc_mask01 > 0.5
    if sel.sum() < 50:
        return cc_mask01.astype(np.float32)
    dm = d[sel]
    obj_d = float(np.percentile(dm, near_pct)); sky_d = float(np.percentile(dm, far_pct))
    if obj_d - sky_d < 0.05:                                     # no depth separation here -> leave as-is
        return cc_mask01.astype(np.float32)
    mid = 0.5 * (obj_d + sky_d); width = max(0.03, 0.35 * (obj_d - sky_d))
    a = 1.0 / (1.0 + np.exp(-(d - mid) / width))
    a = np.clip((a - 0.5) * firm + 0.5, 0, 1)
    return (cc_mask01.astype(np.float32) * a)


def depth_valley(values, lo=5, hi=96, step=5):
    """Find the object/background split DYNAMICALLY from a depth distribution: the largest gap between
    consecutive percentiles (20/40/60...) is the natural valley between the near object cluster and the
    far background cluster. Returns (valley, sep, total, far_frac). No fixed threshold -> adapts per scene
    and per tile (a wheel tile is strongly bimodal; a solid-body tile has no real gap)."""
    v = np.percentile(values, np.arange(lo, hi, step)).astype(np.float32)
    gaps = np.diff(v)
    i = int(np.argmax(gaps))
    valley = float(0.5 * (v[i] + v[i + 1])); sep = float(gaps[i]); total = float(v[-1] - v[0])
    far_frac = float((values < valley).mean())
    return valley, sep, total, far_frac


def depth_continuity_carve(crop_bgr, cc_mask01, depth, carve_level=None, sep=None,
                           min_sep_frac=0.16, max_far_frac=0.60, cliff_frac=0.40,
                           max_carve_area=0.25, aa_frac=0.15, supersample=2, anchor_band=4):
    """Keep the CONNECTED object surface, carve only pieces at the BACKGROUND depth. The object/sky split
    is DYNAMIC (depth_valley) unless a global (carve_level, sep) is supplied -- then this tile uses the
    global split for cross-tile consistency while its native-res depth resolves the fine structure.
    Cliff = a real depth JUMP (morph-gradient in depth units, scaled by sep) so smooth object ramps (arm/
    butt) stay connected and protected; only object->sky jumps (spokes, hole rims) are cliffs."""
    d = depth.astype(np.float32)
    if d.shape[:2] != cc_mask01.shape[:2]:
        d = cv2.resize(d, (cc_mask01.shape[1], cc_mask01.shape[0]))
    cc = cc_mask01 > 0.5
    if cc.sum() < 100:
        return cc_mask01.astype(np.float32)
    vals = d[cc]
    if carve_level is None:                                                # LOCAL dynamic split
        valley, s, total, far_frac = depth_valley(vals)
        if s < min_sep_frac * max(total, 1e-9) or far_frac > max_far_frac:  # not clearly bimodal -> solid
            return cc.astype(np.float32)
        carve_level, sep = valley, s
    if sep is None or sep <= 0:
        sep = max(1e-6, float(np.ptp(vals)) * 0.25)
    mg = cv2.morphologyEx(d, cv2.MORPH_GRADIENT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    cliff = cv2.dilate((mg > max(1e-6, cliff_frac * sep)).astype(np.uint8), np.ones((3, 3), np.uint8))
    free = (cc & (cliff == 0)).astype(np.uint8)
    n, lbl, st, _ = cv2.connectedComponentsWithStats(free, 8)
    ccarea = float(cc.sum())
    carve = np.zeros_like(cc)
    for i in range(1, n):
        a = st[i, cv2.CC_STAT_AREA]
        if a > max_carve_area * ccarea or a < 12:                          # never carve the body; drop speckle
            continue
        sub = lbl == i
        dm = d[sub]
        if float(np.median(dm)) < carve_level and float(dm.std()) < 0.6 * sep:  # at sky plane, low variance
            carve |= sub
    carve = (cv2.dilate(carve.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0) & cc & (d < carve_level)
    keepb = (cc & ~carve).astype(np.uint8)
    # DEPTH-DRIVEN BOUNDARY GRANULARITY (image-independent). The base mask is only an ANCHOR: its eroded
    # interior is forced solid, everything beyond a small OUTER band is forced empty, and INSIDE that band
    # -- BOTH the outer silhouette AND the inner gap edges -- the alpha is a soft ramp across carve_level
    # (the depth iso-contour). So the edge floats +/- a few px to the true sub-pixel depth contour instead
    # of being hard-cut to SAM3's jagged pixel boundary. Supersampling the depth (bicubic) makes that
    # contour smooth -> round curves. Uses ONLY the object's own depth surface (no image edges = no grab/lose).
    aa_w = max(1e-6, aa_frac * sep)
    Hc, Wc = d.shape
    S = int(supersample) if supersample and supersample > 1 else 1
    b = max(1, anchor_band * S)
    disk = lambda r: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    if S > 1:
        d = cv2.resize(d, (Wc * S, Hc * S), interpolation=cv2.INTER_CUBIC)
        ccu = cv2.resize(cc.astype(np.uint8) * 255, (Wc * S, Hc * S), interpolation=cv2.INTER_NEAREST) > 0
        carveu = cv2.resize(carve.astype(np.uint8) * 255, (Wc * S, Hc * S), interpolation=cv2.INTER_NEAREST) > 0
    else:
        ccu, carveu = cc, carve.astype(bool)
    solid = (cv2.erode(ccu.astype(np.uint8), disk(b)) > 0) & ~carveu           # confident object interior
    empty = (cv2.dilate(ccu.astype(np.uint8), disk(b)) == 0)                   # confident background (outer)
    empty |= (cv2.erode(carveu.astype(np.uint8), disk(max(1, b // 2))) > 0)    # + carved gap interiors
    ramp = np.clip((d - carve_level) / aa_w * 0.5 + 0.5, 0, 1).astype(np.float32)
    alpha = ramp
    alpha[solid] = 1.0
    alpha[empty] = 0.0
    if S > 1:
        alpha = cv2.resize(alpha, (Wc, Hc), interpolation=cv2.INTER_AREA)      # area-average = anti-alias
    return alpha


def _align(d_loc, d_glob_tile, cc):
    """Linear-align a tile's metric depth to the global metric depth on the tile's mask pixels (percentile
    matching, robust to the bimodal object/sky mix). Puts every tile in ONE consistent depth space so the
    global object/sky valley applies everywhere, while d_loc keeps the native-res fine structure."""
    sel = cc > 0.5
    if sel.sum() < 30:
        return d_loc
    ql = np.percentile(d_loc[sel], [20, 50, 80]).astype(np.float64)
    qg = np.percentile(d_glob_tile[sel], [20, 50, 80]).astype(np.float64)
    if np.ptp(ql) < 1e-6:
        return d_loc
    a, b = np.polyfit(ql, qg, 1)
    return (a * d_loc + b).astype(np.float32)


def _internal_edge_density(edges, cc):
    """Fraction of the tile's INTERIOR (eroded mask) that is an image edge. Solid silhouettes have edges
    only on their outline -> ~0 interior; lattices (spokes/mesh) have dense interior edges. Used to SKIP
    the (expensive) depth pass on tiles that can't carve anyway."""
    er = cv2.erode(cc.astype(np.uint8), np.ones((7, 7), np.uint8))
    return float(edges[er > 0].mean()) if er.sum() > 30 else 0.0


def zoom_carve_tiled(bgr, mask, depth_fn, tile=1024, overlap=0.30, min_mask_frac=0.05,
                     min_comp_frac=0.02, mode="continuity", global_depth_fn=None,
                     depth_batch_fn=None, max_batch=3, parallel=True, edge_min=0.02, return_ntiles=False):
    """TILE the whole masked area, carve each with zoomed depth-continuity, Hann-blend. SPEED: (1) a cheap
    internal-edge-density PREFILTER skips tiles with no lattice (solid limbs can't carve) so Depth Pro only
    runs where it matters; (2) the surviving tiles run in PARALLEL across both P100s. `depth_fn(bgr, device)`
    returns metric depth on that GPU. `global_depth_fn(bgr)` sets the global object/sky valley; each tile is
    aligned to it for a consistent split."""
    import torch
    H, W = mask.shape[:2]
    mb = (mask > 127).astype(np.uint8)
    edges = (cv2.Canny(cv2.convertScaleAbs(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), alpha=2.0, beta=10),
                       60, 160) > 0).astype(np.float32)
    g_cl = g_sep = None; Dg = None
    if global_depth_fn is not None:
        try:
            Dg = global_depth_fn(bgr).astype(np.float32)
            if Dg.shape[:2] != (H, W):
                Dg = cv2.resize(Dg, (W, H))
            valley, s, total, ff = depth_valley(Dg[mb > 0])
            if s >= 0.12 * max(total, 1e-9):
                g_cl, g_sep = valley, s
        except Exception:
            Dg = None
    win = np.outer(np.hanning(tile), np.hanning(tile)).astype(np.float32) + 1e-3
    step = max(1, int(tile * (1 - overlap)))
    # --- build the job list: tiles with enough mask AND internal structure ---
    jobs = []
    n, lbl, st, _ = cv2.connectedComponentsWithStats(mb, 8)
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] < min_comp_frac * mb.sum():
            continue
        bx, by = st[i, cv2.CC_STAT_LEFT], st[i, cv2.CC_STAT_TOP]
        bw, bh = st[i, cv2.CC_STAT_WIDTH], st[i, cv2.CC_STAT_HEIGHT]
        for ty in range(by, by + bh, step):
            for tx in range(bx, bx + bw, step):
                y0 = min(ty, H - tile) if H >= tile else 0
                x0 = min(tx, W - tile) if W >= tile else 0
                y1, x1 = min(H, y0 + tile), min(W, x0 + tile)
                cc = (mb[y0:y1, x0:x1] > 0).astype(np.float32)
                if cc.mean() < min_mask_frac:
                    continue
                if _internal_edge_density(edges[y0:y1, x0:x1], cc) < edge_min:  # solid tile -> stays base, skip depth
                    continue
                jobs.append((y0, x0, y1, x1))

    def carve_one(y0, x0, y1, x1, d, acc, wsum):
        cc = (mb[y0:y1, x0:x1] > 0).astype(np.float32)
        cl, sep = None, None
        if Dg is not None and g_cl is not None:
            d = _align(d, Dg[y0:y1, x0:x1], cc); cl, sep = g_cl, g_sep
        a = (depth_continuity_carve(bgr[y0:y1, x0:x1], cc, d, carve_level=cl, sep=sep)
             if mode == "continuity" else soft_depth_matte(bgr[y0:y1, x0:x1], cc, d))
        w = win[:y1-y0, :x1-x0]
        acc[y0:y1, x0:x1] += a * w; wsum[y0:y1, x0:x1] += w

    def process(job_list, device, acc, wsum):
        for c in range(0, len(job_list), max_batch):          # batch tiles -> one Depth Pro forward per chunk
            chunk = job_list[c:c + max_batch]
            crops = [bgr[y0:y1, x0:x1] for (y0, x0, y1, x1) in chunk]
            try:
                if depth_batch_fn is not None and len(crops) > 1:
                    ds = depth_batch_fn(crops, device)
                else:
                    ds = [depth_fn(cr, device) for cr in crops]
            except Exception:
                ds = [depth_fn(cr, device) for cr in crops]
            for job, d in zip(chunk, ds):
                carve_one(*job, d.astype(np.float32), acc, wsum)

    ndev = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if parallel and ndev >= 2 and len(jobs) > 1:
        import threading
        try:
            from . import depth as _PD
            _PD.prewarm_dpro(("cuda:0", "cuda:1"))
        except Exception:
            pass
        accs = [np.zeros((H, W), np.float32), np.zeros((H, W), np.float32)]
        wsums = [np.zeros((H, W), np.float32), np.zeros((H, W), np.float32)]
        halves = [jobs[k::2] for k in range(2)]
        ts = [threading.Thread(target=process, args=(halves[k], f"cuda:{k}", accs[k], wsums[k])) for k in range(2)]
        for t in ts: t.start()
        for t in ts: t.join()
        acc = accs[0] + accs[1]; wsum = wsums[0] + wsums[1]
    else:
        acc = np.zeros((H, W), np.float32); wsum = np.zeros((H, W), np.float32)
        process(jobs, "cuda:0", acc, wsum)

    covered = wsum > 0
    blended = np.where(covered, acc / np.maximum(wsum, 1e-6), mb.astype(np.float32))
    # allow the depth-floated boundary to extend a few px beyond the base mask (outer-silhouette AA); clamp
    # far outside so we never expand wholesale. Uncovered (skipped solid) tiles keep the base mask.
    guard = cv2.dilate(mb, np.ones((9, 9), np.uint8)) > 0
    out = np.where(covered, blended * guard, mb.astype(np.float32) * (mb > 0))
    return (out, len(jobs)) if return_ntiles else out


def zoom_carve(bgr, mask, depth_fn, regions=None, pad=0.15, return_regions=False):
    """Carve thin lattice structure into `mask` using per-region zoomed depth. `depth_fn(bgr_crop)->depth`
    (near=bright). Returns a full-frame soft alpha [0,1] (base mask elsewhere unchanged)."""
    H, W = mask.shape[:2]
    alpha = (mask > 127).astype(np.float32)
    if regions is None:
        regions = find_carve_regions(mask, bgr)
    used = []
    for (x0, y0, x1, y1, tag) in regions:
        pw, ph = x1 - x0, y1 - y0
        px, py = int(pw * pad), int(ph * pad)
        cx0, cy0 = max(0, x0 - px), max(0, y0 - py)
        cx1, cy1 = min(W, x1 + px), min(H, y1 + py)
        crop = bgr[cy0:cy1, cx0:cx1]
        cc = (mask[cy0:cy1, cx0:cx1] > 127).astype(np.float32)
        if cc.sum() < 100:
            continue
        try:
            d = depth_fn(crop)
        except Exception:
            continue
        a = soft_depth_matte(crop, cc, d)
        alpha[cy0:cy1, cx0:cx1] = np.where(cc > 0.5, a, alpha[cy0:cy1, cx0:cx1])
        used.append((cx0, cy0, cx1, cy1, tag))
    return (alpha, used) if return_regions else alpha
