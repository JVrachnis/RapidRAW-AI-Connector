"""pelib.sam3 - SAM3 concept/box segmentation (transformers Sam3Model) + see-through carve.

SAM3 does joint detection+segmentation from a TEXT concept ("bicycle") or a geometric BOX prompt,
returning per-instance masks. This pairs with:
  - the LLM intent parser (which emits target concepts)  -> text prompt
  - RapidRAW's rough user selection (a mask/box)         -> box prompt
  - the VLM judge loop (which names missing/wrong objects) -> re-prompt / re-threshold

Runs fp16 + eager attention so it works on the Pascal P100s (SM 6.0, no flash-attn). ~2s/inference.
Model is cached (loaded once). facebook/sam3 is gated; jetjodh/sam3 is an open, byte-identical mirror.
"""
from __future__ import annotations
import os
import numpy as np
import cv2
import torch

import threading

SAM3_ID = os.environ.get("SAM3_ID", "jetjodh/sam3")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
_S3 = {}  # lazy singleton cache
_LOCK = threading.Lock()  # guards _S3 population: concurrent lazy loads race (half-init model -> dtype mismatch)


def _load(device=None):
    """Return (processor, model) with the model on `device` (default cuda:0). A separate model copy is
    cached per device so we can run one SAM3 on each P100 in parallel. Cache population is locked so two
    worker threads loading different device copies at once can't corrupt each other's init (was causing
    'mat1 and mat2 must have the same dtype' on the second device)."""
    device = device or (DEV if DEV == "cpu" else "cuda:0")
    from transformers import Sam3Processor, Sam3Model
    key = f"m::{device}"
    with _LOCK:
        if "p" not in _S3:
            _S3["p"] = Sam3Processor.from_pretrained(SAM3_ID)
        if key not in _S3:
            _S3[key] = (Sam3Model.from_pretrained(SAM3_ID, torch_dtype=torch.float16,
                                                  attn_implementation="eager").to(device).eval())
    return _S3["p"], _S3[key]


def _to_rgb(image):
    """Accept a BGR ndarray (cv2, the pipeline default) or a PIL image; return an RGB ndarray."""
    if isinstance(image, np.ndarray):
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if image.ndim == 3 else image
    return np.array(image.convert("RGB"))


def instances(image, text=None, boxes=None, threshold=0.5, mask_threshold=0.5, device=None):
    """Per-instance SAM3 masks. Prompt by `text` concept OR `boxes` (list of xyxy). `device` picks which
    GPU's model copy to use. Returns (masks: list[HxW bool], scores: list[float])."""
    from PIL import Image
    device = device or (DEV if DEV == "cpu" else "cuda:0")
    proc, model = _load(device)
    rgb = _to_rgb(image)
    H, W = rgb.shape[:2]
    pil = Image.fromarray(rgb)
    kw = {}
    if text:
        kw["text"] = text
    if boxes is not None and len(boxes):
        kw["input_boxes"] = [[[float(v) for v in b] for b in boxes]]
    if not kw:
        raise ValueError("sam3.instances needs a text concept or boxes")
    inp = proc(images=pil, return_tensors="pt", **kw).to(device)
    for k in inp:
        if hasattr(inp[k], "dtype") and inp[k].dtype == torch.float32:
            inp[k] = inp[k].half()
    with torch.no_grad():
        out = model(**inp)
    r = proc.post_process_instance_segmentation(
        out, threshold=threshold, mask_threshold=mask_threshold, target_sizes=[(H, W)])[0]
    masks = [np.asarray(m.cpu() if hasattr(m, "cpu") else m).astype(bool) for m in r["masks"]]
    scores = [float(s) for s in r["scores"]]
    return masks, scores


def _representations(bgr):
    """A few complementary optimizations of the image. Different shadow-lifts/contrast reveal different
    objects & parts to SAM3, so segmenting each and unioning recovers what any single pass drops."""
    def enh(g, c):
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        L, A, B = cv2.split(lab)
        L = np.clip(255.0 * (L.astype(np.float32) / 255.0) ** g, 0, 255).astype(np.uint8)
        L = cv2.createCLAHE(clipLimit=c, tileGridSize=(8, 8)).apply(L)
        return cv2.cvtColor(cv2.merge([L, A, B]), cv2.COLOR_LAB2BGR)
    # chan-B: blue channel as gray. Benchmarked 2026-07-08 (~/comfy/bench/REPORT.md):
    # on warm-toned shots it alone out-recalled the whole lift pool (real foliage/twig
    # edges, VLM-verified true positives); negligible cost (~2.2s/rep warm).
    chan_b = cv2.cvtColor(bgr[:, :, 0], cv2.COLOR_GRAY2BGR)
    return [("raw", bgr), ("enhance", enh(0.45, 3.0)),
            ("strong-lift", enh(0.35, 2.0)), ("high-clahe", enh(0.60, 5.0)),
            ("chan-B", chan_b)]


def multirep_instances(image, text, threshold=0.5, mask_threshold=0.5, variants=None, parallel=False):
    """Run SAM3 on several complementary optimizations of the image (multi-representation) and pool ALL
    instance masks. Higher recall on dark/low-contrast subjects than a single pass. Returns
    [(mask HxW bool, score)] across all representations. (~2.2s/rep warm; sequential on one P100.
    Full-res batching OOMs on 16GB — split across the 2 cards for ~2x if speed matters.)"""
    bgr = image if isinstance(image, np.ndarray) else cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
    reps = variants or _representations(bgr)
    concepts = [c.strip() for c in text.split(".") if c.strip()] or [text.strip()]
    jobs = [(name, v, c) for name, v in reps for c in concepts]  # every (representation x concept)

    ndev = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if parallel and ndev >= 2:                               # split across both P100s, one SAM3 copy each
        devs = ["cuda:0", "cuda:1"]
        for d in devs:                                      # warm both copies in the MAIN thread first
            _load(d)                                        # (concurrent lazy load races -> dtype corruption)
        halves = [jobs[i::2] for i in range(2)]             # interleave so load balances
        results = [[], []]

        def worker(i):
            for name, v, c in halves[i]:
                try:
                    ms, ss = instances(v, text=c, threshold=threshold, mask_threshold=mask_threshold, device=devs[i])
                    results[i] += list(zip(ms, ss))
                except Exception as e:
                    print(f"[multirep|{devs[i]}] '{name}'/'{c}' failed:", str(e)[:50])
        ts = [threading.Thread(target=worker, args=(i,)) for i in range(2)]
        for t in ts: t.start()
        for t in ts: t.join()
        return results[0] + results[1]

    pooled = []                                             # sequential (single GPU)
    for name, v, c in jobs:
        try:
            ms, ss = instances(v, text=c, threshold=threshold, mask_threshold=mask_threshold)
            pooled += list(zip(ms, ss))
        except Exception as e:
            print(f"[multirep] '{name}'/'{c}' failed:", str(e)[:50])
    return pooled


def smart_combine(instances, bgr, depth=None, core_votes=2, core_score=0.85,
                  add_score=0.55, depth_margin=0.16, min_add=30):
    """Combine pooled multi-rep instances smarter than a blind union — vote + confidence + depth.

    An object detected in SEVERAL representations (or once with high confidence) is a trusted CORE.
    Single-representation additions are kept only if they (a) touch the core, or (b) are reasonably
    confident AND sit at the object's foreground depth. So a real part one lift recovered survives,
    but an isolated low-score blob at BACKGROUND depth (an artifact of one lift) is dropped.
    `instances` = [(mask HxW bool, score)]. Returns a bool mask."""
    if not instances:
        H, W = bgr.shape[:2]
        return np.zeros((H, W), bool)
    H, W = bgr.shape[:2]
    votes = np.zeros((H, W), np.int16)
    best = np.zeros((H, W), np.float32)
    for m, s in instances:
        votes[m] += 1
        best[m] = np.maximum(best[m], np.float32(s))
    core = (votes >= core_votes) | (best >= core_score)
    add = (votes >= 1) & (~core)
    keep = core.astype(np.uint8)
    if add.any():
        d = fg = None
        if depth is not None:
            d = (depth if depth.ndim == 2 else cv2.cvtColor(depth, cv2.COLOR_BGR2GRAY)).astype(np.float32)
            if d.shape != (H, W): d = cv2.resize(d, (W, H))
            d /= 255.0
            fg = float(np.percentile(d[votes >= 1], 60))
        core_d = cv2.dilate(core.astype(np.uint8), np.ones((9, 9), np.uint8))
        n, lab, st, _ = cv2.connectedComponentsWithStats(add.astype(np.uint8), 8)
        for c in range(1, n):
            if st[c, cv2.CC_STAT_AREA] < min_add:
                continue
            reg = (lab == c)
            touches = bool(core_d[reg].any())
            score_ok = float(best[reg].max()) >= add_score
            depth_ok = (d is None) or (float(np.median(d[reg])) >= fg - depth_margin)
            if touches or (score_ok and depth_ok):
                keep[reg] = 1
    return keep.astype(bool)


def _fill_holes_depth(m, d, fg, margin):
    """Fill enclosed holes in a SINGLE instance that sit at object depth (drop-outs); leave far holes."""
    H, W = m.shape
    inv = (~m).astype(np.uint8)
    n, lab = cv2.connectedComponents(inv, 8)
    border = set(np.unique(np.concatenate([lab[0, :], lab[-1, :], lab[:, 0], lab[:, -1]])))
    out = m.copy()
    for c in range(1, n):
        if c in border:
            continue
        reg = (lab == c)
        if d is None or float(np.median(d[reg])) >= fg - margin:
            out |= reg
    return out


def combine_instances(pooled, bgr, depth=None, iou_thr=0.45, min_area=600,
                      add_score=0.55, depth_margin=0.16):
    """Turn pooled multi-rep detections into DISTINCT OBJECTS, producing BOTH a binary union mask (for
    removal / 'select the crowd') and a labeled instance map (for per-object selection).

    Same-object detections from different representations are near-identical (high IoU) -> merged;
    two physically-overlapping-but-different people share only partial overlap (low IoU) -> kept separate.
    Junk (single-rep, low-score, far-depth) is dropped. Hole-fill is done PER INSTANCE so gaps between
    distinct objects are never bridged. Returns (binary bool, labels int32, instances [(mask,score)])."""
    H, W = bgr.shape[:2]
    items = sorted([(m, float(s)) for m, s in pooled if m.sum() >= min_area], key=lambda x: -x[1])
    clusters = []                                            # each: {"mask", "votes", "score"}
    for m, s in items:
        bi, bov = -1, 0.0
        for i, cl in enumerate(clusters):
            inter = int((m & cl["mask"]).sum())
            if inter == 0:
                continue
            iou = inter / max(int((m | cl["mask"]).sum()), 1)
            cont = inter / max(int(m.sum()), 1)             # a partial re-detection of the same object
            ov = max(iou, 0.85 * cont)
            if ov > bov:
                bov, bi = ov, i
        if bov >= iou_thr:
            clusters[bi]["mask"] |= m
            clusters[bi]["votes"] += 1
            clusters[bi]["score"] = max(clusters[bi]["score"], s)
        else:
            clusters.append({"mask": m.copy(), "votes": 1, "score": s})
    d = fg = None
    if depth is not None:
        d = (depth if depth.ndim == 2 else cv2.cvtColor(depth, cv2.COLOR_BGR2GRAY)).astype(np.float32)
        if d.shape != (H, W): d = cv2.resize(d, (W, H))
        d /= 255.0
        allm = np.zeros((H, W), bool)
        for cl in clusters: allm |= cl["mask"]
        fg = float(np.percentile(d[allm], 60)) if allm.any() else 0.0
    labels = np.zeros((H, W), np.int32)
    binm = np.zeros((H, W), bool)
    kept = []
    for cl in clusters:
        # junk reject: a lone (single-rep) low-confidence blob at background depth is a lift artifact
        if cl["votes"] < 2 and cl["score"] < add_score and d is not None \
                and float(np.median(d[cl["mask"]])) < fg - depth_margin:
            continue
        m = _fill_holes_depth(cl["mask"], d, fg, depth_margin)  # per-instance -> never bridges neighbours
        lid = len(kept) + 1
        labels[m] = lid
        binm |= m
        kept.append((m, cl["score"]))
    return binm, labels, kept


def clean(mask, frac=0.0005, min_px=64):
    """Drop tiny islands (< frac of frame) — removes SAM3's occasional spurious flecks."""
    H, W = mask.shape
    u8 = mask.astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(u8, 8)
    if n <= 1:
        return mask
    keep = np.zeros_like(u8)
    mn = max(min_px, int(frac * H * W))
    for c in range(1, n):
        if st[c, cv2.CC_STAT_AREA] >= mn:
            keep[lab == c] = 1
    return keep.astype(bool)


def carve_seethrough(mask, bgr, depth=None, ring=55, dE=19, min_hole=42, lum_guard=0.5, aggr=0.75,
                     depth_margin=0.14, texture=True, tex_win=7, tex_ratio=1.6, shadow_ab=10.0,
                     sky_L=170):
    """Remove in-mask pixels that show BACKGROUND through the object (see-through gaps: wheel spokes,
    railings, mesh) while keeping the object. Robust over BOTH bright sky AND dark building via three
    complementary signals:

      COLOUR  - kmeans (K<=6) models background colours from a ring OUTSIDE the mask; the object core
                colour comes from an eroded interior. A pixel is a gap candidate if it's clearly closer
                to a background cluster than to the object, and brighter than a luminance midpoint taken
                against its OWN NEAREST background cluster (per-pixel local guard: high over sky so dark
                spokes survive, low over a dark building so dark gaps carve while spoke metal is spared
                by colour).
      DEPTH   - (optional, near=bright) only carve pixels at clearly BACKGROUND/far depth. The thin
                spokes are too fine for monocular depth to resolve, so depth can't protect them, but it
                gates OUT near-object smooth regions that merely match a background colour.
      TEXTURE - a gap over a textured building has high local variance; smooth bike metal is low. A
                pixel needs background-like colour; texture is a soft extra vote that a candidate is a
                real see-through gap over a busy background, not object surface.
    """
    H, W = mask.shape
    m = mask.astype(np.uint8)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    core = cv2.erode(m, np.ones((21, 21), np.uint8))
    core = core if core.sum() > 200 else m
    obj = np.median(lab[core > 0], 0)
    outer = cv2.dilate(m, np.ones((ring, ring), np.uint8)) - m
    bgpx = lab[outer > 0].astype(np.float32)
    if len(bgpx) < 100:
        return mask
    K = int(min(6, max(2, len(bgpx) // 500)))
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 12, 1.0)
    _, _, cen = cv2.kmeans(bgpx, K, None, crit, 3, cv2.KMEANS_PP_CENTERS)
    ys, xs = np.where(mask)
    px = lab[ys, xs]
    d_obj = np.linalg.norm(px - obj, axis=1)
    dmat = np.linalg.norm(px[:, None, :] - cen[None, :, :], axis=2)   # [N, K]
    nn = np.argmin(dmat, axis=1)
    d_bg = dmat[np.arange(len(px)), nn]
    local_bg_L = cen[nn, 0]                                            # nearest background luminance, per pixel
    Lmid = obj[0] + lum_guard * (local_bg_L - obj[0])                  # guard scales with LOCAL bg brightness
    carve_px = (d_bg < dE) & (d_bg < d_obj * aggr) & (px[:, 0] > Lmid)
    # SHADOW GUARD (chroma): a shadowed part of the object keeps the object's a/b chroma and only drops
    # luminance, so it must NOT be carved even though it looks "dark like background". Protect any pixel
    # whose chroma matches the object AND whose nearest bg cluster has clearly different chroma (i.e. the
    # darkness is a shadow on the object, not a see-through gap). Neutral-grey backgrounds fall back to depth.
    d_ab_obj = np.linalg.norm(px[:, 1:3] - obj[1:3], axis=1)          # chroma distance to the object
    d_ab_bg = np.linalg.norm(cen[nn, 1:3] - obj[1:3], axis=1)         # chroma gap object<->its nearest bg
    shadow = (d_ab_obj < shadow_ab) & (d_ab_bg > shadow_ab)
    carve_px &= (~shadow)
    if depth is not None:                                             # DEPTH is the strong signal when present
        d = depth
        if d.ndim == 3: d = cv2.cvtColor(d, cv2.COLOR_BGR2GRAY)
        if d.shape != (H, W): d = cv2.resize(d, (W, H))
        d = d.astype(np.float32) / 255.0                             # near=bright -> object is high
        # LOCAL adaptive reference: nearest object depth in a neighbourhood (handles a subject that
        # spans a depth range, e.g. a rotating bike where the two wheels sit at different depths).
        dm = d * (mask > 0)
        ksz = max(21, int(0.06 * max(H, W)) | 1)                     # ~wheel-radius scale, odd
        near_ref = cv2.dilate(dm, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksz, ksz)))
        far = (d[ys, xs] < near_ref[ys, xs] - depth_margin)          # far relative to LOCAL near structure
        # depth carves the gaps; colour spares the spokes (they match the object, not the background)
        carve_px = (carve_px | (far & (d_bg < dE))) & far
    elif texture:                                                    # no depth -> texture fallback discriminator
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        mean = cv2.blur(gray, (tex_win, tex_win))
        var = cv2.blur(gray * gray, (tex_win, tex_win)) - mean * mean
        obj_var = float(np.median(var[core > 0])) + 1.0
        tv = var[ys, xs]
        textured = tv > tex_ratio * obj_var
        # A see-through gap is admitted by EITHER a positive texture vote (a gap over a busy
        # background) OR being a bright sky gap. The sky escape must be ABSOLUTE (sky-bright),
        # NOT "brighter than the nearest bg cluster": when that cluster is a dark neutral
        # (tree/shadow), `local_bg_L - 8` is tiny, so every dark SMOOTH OBJECT pixel (dark tire,
        # frame, clothing, shoe) cleared it and got carved -- forensics (2026-07-09,
        # /tmp/carve_forensics) showed this depthless texture fallback eating 96k object px
        # (dark L~40 surfaces at NEAR object depth) vs 3.3k for the depth path. Gating the escape
        # on an absolute sky brightness spares dark smooth object surface (it now needs a real
        # texture vote) while keeping bright-sky spoke gaps carveable; dark gaps over a textured
        # background still carve via `textured`. Cut eaten-object px ~96k -> ~9k on the BMX bench.
        carve_px &= (textured | (px[:, 0] > sky_L))
    cm = np.zeros((H, W), np.uint8)
    cm[ys[carve_px], xs[carve_px]] = 1
    cm = cv2.morphologyEx(cm, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))  # drop 1px specks
    n, cc, st, _ = cv2.connectedComponentsWithStats(cm, 8)
    holes = np.zeros_like(cm)
    for c in range(1, n):
        if st[c, cv2.CC_STAT_AREA] >= min_hole:
            holes[cc == c] = 1
    return mask & (~holes.astype(bool))


def carve_continuity(mask, bgr, depth, cont_thr=0.045, depth_margin=0.10, min_hole=40,
                     shadow_ab=10.0, dE=26, ring=55, edge_map=None):
    """See-through carve fusing DEPTH-continuity + COLOUR + CONNECTIVITY (needs a resolved/zoomed depth
    map, near=bright). The object is one/few surfaces CONTINUOUS in depth; a see-through gap is a
    sub-region separated by a depth CLIFF and either sitting FAR, or a BACKGROUND-coloured intruder
    connected to the outside (e.g. a tree branch behind the wheel, aligned with the spokes: it's near
    like a spoke so depth alone keeps it, but it's tree-coloured and touches the external tree -> carve).

    Per continuous sub-region (split at depth cliffs), carve if:
      * median depth is clearly FAR vs object foreground (a real see-through gap), OR
      * the region is BACKGROUND-COLOURED (closer to a bg colour cluster than to the object) AND touches
        the exterior of the mask (a background object poking through) AND not object-chroma.
    Object-coloured spokes/rim/frame and shadowed parts (object chroma, near, interior) survive.
    Optional `edge_map` (e.g. TEED) tightens cliffs where depth is ambiguous."""
    H, W = mask.shape
    m = mask.astype(bool)
    d = depth
    if d.ndim == 3: d = cv2.cvtColor(d, cv2.COLOR_BGR2GRAY)
    if d.shape != (H, W): d = cv2.resize(d, (W, H))
    d = d.astype(np.float32) / 255.0
    ds = cv2.GaussianBlur(d, (0, 0), 1.2)                              # denoise depth before gradients
    grad = cv2.magnitude(cv2.Sobel(ds, cv2.CV_32F, 1, 0, ksize=3),
                         cv2.Sobel(ds, cv2.CV_32F, 0, 1, ksize=3))
    barrier = (grad > cont_thr).astype(np.uint8)                      # depth cliffs = surface boundaries
    if edge_map is not None:                                          # optional TEED edges reinforce cliffs
        e = edge_map if edge_map.ndim == 2 else cv2.cvtColor(edge_map, cv2.COLOR_BGR2GRAY)
        if e.shape != (H, W): e = cv2.resize(e, (W, H))
        barrier = np.maximum(barrier, (e > 60).astype(np.uint8))
    inner = (m & (barrier == 0)).astype(np.uint8)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    core = cv2.erode(m.astype(np.uint8), np.ones((21, 21), np.uint8))
    core = core if core.sum() > 200 else m.astype(np.uint8)
    obj = np.median(lab[core > 0], 0)
    ext = (~m)                                                        # background colour clusters from outside
    outer = (cv2.dilate(m.astype(np.uint8), np.ones((ring, ring), np.uint8)) & ext.astype(np.uint8))
    bgpx = lab[outer > 0].astype(np.float32)
    cen = None
    if len(bgpx) >= 100:
        K = int(min(6, max(2, len(bgpx) // 500)))
        crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 12, 1.0)
        _, _, cen = cv2.kmeans(bgpx, K, None, crit, 3, cv2.KMEANS_PP_CENTERS)
    ext_u8 = ext.astype(np.uint8)
    fg = float(np.percentile(d[m], 65))                               # object foreground (nearer part)
    n, cc, st, _ = cv2.connectedComponentsWithStats(inner, 8)
    out = m.copy()
    for c in range(1, n):
        reg = (cc == c)
        if st[c, cv2.CC_STAT_AREA] < min_hole:
            continue
        med = float(np.median(d[reg]))
        col = np.median(lab[reg], 0)
        is_obj_chroma = np.linalg.norm(col[1:3] - obj[1:3]) < shadow_ab
        # (a) far continuous surface = see-through gap
        far = med < fg - depth_margin
        # (b) background-coloured intruder connected to the outside (the branch case)
        bg_intruder = False
        if cen is not None and not is_obj_chroma:
            d_bg = float(np.min(np.linalg.norm(col[None, :] - cen, axis=1)))
            d_obj = float(np.linalg.norm(col - obj))
            touches_ext = cv2.dilate(reg.astype(np.uint8), np.ones((3, 3), np.uint8))[ext_u8 > 0].any()
            bg_intruder = (d_bg < d_obj) and (d_bg < dE) and touches_ext and (med < fg - 0.3 * depth_margin)
        if far and is_obj_chroma and med > fg - 2 * depth_margin:
            continue                                                 # object-chroma near-ish bit: keep
        if far or bg_intruder:
            out[reg] = False
    return out


def mask(image, text=None, boxes=None, threshold=0.5, mask_threshold=0.5,
         carve=False, clean_frac=0.0005):
    """Union SAM3 mask for a concept or box, with optional fleck-clean and see-through carve.
    Returns (mask: HxW bool, scores: list[float])."""
    masks, scores = instances(image, text=text, boxes=boxes,
                              threshold=threshold, mask_threshold=mask_threshold)
    rgb = _to_rgb(image)
    H, W = rgb.shape[:2]
    u = np.zeros((H, W), bool)
    for mm in masks:
        u |= mm
    if clean_frac:
        u = clean(u, clean_frac)
    if carve and u.any():
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        u = carve_seethrough(u, bgr)
    return u, scores
