#!/usr/bin/env python3
"""
mask_c2f.py - coarse-to-fine semantic masking.

Pass 1 (GLOBAL/coarse): detect the target on the WHOLE image -> locates all instances
with full-scene context (recall). Cheap, catches everything including faint/distant ones.
Pass 2 (LOCAL/fine): cluster the coarse boxes into regions, crop each at HIGH res and
re-detect + SAM-segment there -> finds individuals a whole-frame pass blurs together and
produces pixel-crisp masks. Union all -> final mask. Optional ViTMatte edge refine.

Mirrors inpaint_c2f: global locate (recall) then local high-res segment (precision).

Usage:
  mask_c2f.py IMAGE --query "person. camera. tripod." --out mask.png
     [--roi ROI.png] [--box-threshold 0.22] [--text-threshold 0.18]
     [--dilate 12] [--min-box 8]
Runs on inferno (transformers Grounding-DINO + SAM).
"""
from __future__ import annotations
import argparse, os, sys
import numpy as np, cv2
from PIL import Image
sys.path.insert(0, os.path.expanduser("~/comfy"))
import grounded_sam as GS
from pelib import comfy as PC, matte as PM


def edge_preprocess(bgr):
    """Condition an image for edge detection. Crushed 8-bit shadows + JPEG noise make TEED emit
    weak, noisy edges; this denoises (so noise isn't mistaken for edges), lifts local contrast, and
    unsharps real boundaries. Returns a clean high-local-contrast BGR for the edge model."""
    # edge-preserving denoise: kills speckle that would become spurious edges, keeps true borders
    den = cv2.bilateralFilter(bgr, d=7, sigmaColor=60, sigmaSpace=60)
    # local contrast on luminance (LAB) so boundaries in dark regions become visible
    lab = cv2.cvtColor(den, cv2.COLOR_BGR2LAB)
    l, a_, b_ = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(l)
    out = cv2.cvtColor(cv2.merge([l, a_, b_]), cv2.COLOR_LAB2BGR)
    # unsharp mask: accentuate the boundaries the edge net keys on
    blur = cv2.GaussianBlur(out, (0, 0), 2.0)
    return cv2.addWeighted(out, 1.6, blur, -0.6, 0)


def edge_map(bgr, res=1024, preprocess=True):
    """Learned soft-edge map (TEED) via ComfyUI — object boundaries as an extra detection channel.
    TEED replaces HED here: lighter, modern, and the aux HEDPreprocessor build has a `safe` KeyError bug.
    `preprocess` conditions the input (denoise+local-contrast+unsharp) so edges pop on dark 8-bit frames."""
    if preprocess:
        bgr = edge_preprocess(bgr)
    H, W = bgr.shape[:2]
    name = PC.stage(bgr)
    wf = {
        "1": {"class_type": "LoadImage", "inputs": {"image": name}},
        "2": {"class_type": "TEEDPreprocessor", "inputs": {"image": ["1", 0], "resolution": res, "safe_steps": 2}},
        "3": {"class_type": "SaveImage", "inputs": {"images": ["2", 0], "filename_prefix": "c2f_edge"}},
    }
    out = PC.run(wf, timeout=120, cleanup_names=(name,))
    return cv2.resize(out, (W, H), interpolation=cv2.INTER_LINEAR)


def channel_variants(bgr):
    """Per-channel (R,G,B) + luminance, each as a 3ch gray image. Objects invisible in luminance can
    pop in a single colour channel (a red camera on dark bg is obvious in R)."""
    b, g, r = cv2.split(bgr)
    lum = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return [cv2.cvtColor(c, cv2.COLOR_GRAY2BGR) for c in (r, g, b, lum)]


def birefnet_mask(bgr):
    """Whole salient-subject mask via BiRefNet (ComfyUI) — wraps thin parts (spokes, frame) SAM misses."""
    H, W = bgr.shape[:2]
    s = min(1.0, 2048 / max(H, W))
    im = cv2.resize(bgr, (int(W * s), int(H * s)), interpolation=cv2.INTER_AREA) if s < 1 else bgr
    rw = max(32, round(im.shape[1] / 32) * 32); rh = max(32, round(im.shape[0] / 32) * 32)
    name = PC.stage(im)
    wf = {
        "1": {"class_type": "LoadImage", "inputs": {"image": name}},
        "2": {"class_type": "AutoDownloadBiRefNetModel", "inputs": {"model_name": "General-HR", "device": "AUTO"}},
        "3": {"class_type": "GetMaskByBiRefNet", "inputs": {"model": ["2", 0], "images": ["1", 0],
              "width": rw, "height": rh, "upscale_method": "bilinear", "mask_threshold": 0.5}},
        "5": {"class_type": "MaskToImage", "inputs": {"mask": ["3", 0]}},
        "6": {"class_type": "SaveImage", "inputs": {"images": ["5", 0], "filename_prefix": "c2f_bire"}},
    }
    out = PC.run(wf, timeout=180, cleanup_names=(name,))
    m = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY) if out.ndim == 3 else out
    return cv2.resize(m, (W, H), interpolation=cv2.INTER_LINEAR)


def enhance_for_detection(bgr, clahe=3.0, gamma=0.55, sharp=0.6, sat=1.25):
    """Reveal shadow-blended subjects for the DETECTOR only (mask still applies to the original).
    CLAHE local contrast + gamma shadow-lift + unsharp + a touch of saturation."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=clahe, tileGridSize=(8, 8)).apply(l)
    out = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR).astype(np.float32) / 255.0
    out = np.power(np.clip(out, 0, 1), gamma)                       # lift shadows (gamma<1)
    if sat != 1.0:
        hsv = cv2.cvtColor((out * 255).astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[..., 1] = np.clip(hsv[..., 1] * sat, 0, 255)
        out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32) / 255.0
    if sharp > 0:
        blur = cv2.GaussianBlur(out, (0, 0), 2.0)
        out = np.clip(out * (1 + sharp) - blur * sharp, 0, 1)
    return (out * 255).astype(np.uint8)


def cluster_boxes(boxes, gap):
    """Merge boxes whose expanded rects touch into region clusters (returns list of bbox)."""
    n = len(boxes)
    parent = list(range(n))
    def find(i):
        while parent[i] != i: parent[i] = parent[parent[i]]; i = parent[i]
        return i
    def rect(b): return (b[0]-gap, b[1]-gap, b[2]+gap, b[3]+gap)
    for i in range(n):
        for j in range(i+1, n):
            a, c = rect(boxes[i]), rect(boxes[j])
            if not (a[2] < c[0] or c[2] < a[0] or a[3] < c[1] or c[3] < a[1]):
                parent[find(i)] = find(j)
    groups = {}
    for i in range(n): groups.setdefault(find(i), []).append(i)
    out = []
    for idxs in groups.values():
        bs = boxes[idxs]
        out.append([bs[:,0].min(), bs[:,1].min(), bs[:,2].max(), bs[:,3].max()])
    return out


def zoom_depth(full, mask_bool, carve_depth="depthpro", depth_map_path=""):
    """Zoomed depth (uint8, near=bright) on the mask bbox — resolves thin structure. Depth Pro (sharp)
    with DA-V2 zoom fallback. Returns a full-size uint8 array, or None. Reused by the smart-combine,
    the depth-aware hole-fill, the see-through carve, and the adaptive feather (one pass)."""
    H, W = full.shape[:2]
    mys, mxs = np.where(mask_bool)
    if not len(mxs):
        return None
    pad = 60
    zy0, zy1 = max(0, mys.min()-pad), min(H, mys.max()+pad)
    zx0, zx1 = max(0, mxs.min()-pad), min(W, mxs.max()+pad)
    crop = full[zy0:zy1, zx0:zx1]
    from pelib import depth as PD
    dz = None
    if carve_depth == "depthpro":
        try:
            dz = PD.depth_pro(crop)
        except Exception as e:
            print("[depth] Depth Pro failed, falling back to DA-V2:", str(e)[:60])
    if dz is None:
        try:
            dz = PD.depth_map(crop, res=1024)
            if dz.ndim == 3: dz = cv2.cvtColor(dz, cv2.COLOR_BGR2GRAY)
        except Exception as e:
            print("[depth] zoom-depth failed:", str(e)[:60])
            if depth_map_path: dz = cv2.imread(depth_map_path, cv2.IMREAD_GRAYSCALE)
    if dz is None:
        return None
    Dc = np.zeros((H, W), np.uint8)
    Dc[zy0:zy1, zx0:zx1] = cv2.resize(dz, (zx1-zx0, zy1-zy0))
    return Dc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image"); ap.add_argument("--query", required=True); ap.add_argument("--out", default="mask_c2f.png")
    ap.add_argument("--roi", default="", help="optional coarse ROI to restrict the search")
    ap.add_argument("--det-images", default="", help="comma-separated exposure stack (RAW EV brackets); detections unioned")
    ap.add_argument("--box-threshold", type=float, default=0.22)
    ap.add_argument("--text-threshold", type=float, default=0.18)
    ap.add_argument("--global-side", type=int, default=1280)
    ap.add_argument("--local-side", type=int, default=1400, help="min long-side each region is upscaled to for the fine pass")
    ap.add_argument("--cluster-gap", type=int, default=40)
    ap.add_argument("--mode", choices=["precise", "removal"], default="precise",
                    help="precise = tight per-object silhouettes (selection/relight/grade); removal = filled region blob for inpaint")
    ap.add_argument("--protect-query", default="skateboard ramp. wooden ramp. halfpipe.",
                    help="semantic structures to KEEP (subtracted from the mask); empty to disable")
    ap.add_argument("--dilate", type=int, default=4)
    ap.add_argument("--birefnet-mode", choices=["auto", "on", "off"], default="auto",
                    help="BiRefNet whole-subject completion (auto = when few objects, i.e. a coherent subject)")
    ap.add_argument("--birefnet-max-objs", type=int, default=4)
    ap.add_argument("--birefnet-reach", type=int, default=50)
    ap.add_argument("--matte", action="store_true", default=True, help="ViTMatte crisp-edge refine of the final mask")
    ap.add_argument("--no-matte", dest="matte", action="store_false")
    ap.add_argument("--matte-band", type=int, default=20)
    ap.add_argument("--matte-refine", action="store_true", default=False,
                    help="run ViTMatte on the SAM3/multirep mask (any backend) and OUTPUT the soft alpha — "
                         "crisp edges + carves negative space between connected thin parts (frame/crank/hair). "
                         "Subject-gated (skips crowds, which it over-erodes); runs on the freest GPU.")
    ap.add_argument("--matte-refine-band", type=int, default=40, help="unknown-band half-width for matte-refine (px)")
    ap.add_argument("--matte-maxside", type=int, default=2048, help="matte working res cap (ViTMatte attention OOMs higher)")
    ap.add_argument("--matte-max-cc", type=int, default=8, help="max large mask blobs to still treat as a subject (not a crowd)")
    ap.add_argument("--zoom-carve", action="store_true", default=False,
                    help="depth-tiled carve: recover lattice structure (spokes/mesh) SAM3 fills solid, with "
                         "depth-continuity + image-independent depth-AA edges. Best-quality, ~8s. Selection/precise.")
    ap.add_argument("--zoom-tile", type=int, default=1024, help="zoom-carve tile size (px)")
    ap.add_argument("--multi-channel", action="store_true", default=False,
                    help="also detect on R/G/B/gray channels and union (catches colour-distinct objects)")
    ap.add_argument("--edges", action="store_true", default=False, help="add a learned HED edge map as a detection channel")
    ap.add_argument("--backend", choices=["sam2", "sam3", "both"], default="sam2",
                    help="segmentation core: sam2 = DINO->SAM2 coarse-to-fine (geometric); "
                         "sam3 = SAM3 concept segmentation (semantic, LLM-in-the-loop); "
                         "both = SAM3 finds instances, SAM2 tightens each edge")
    ap.add_argument("--sam3-threshold", type=float, default=0.5, help="SAM3 instance/presence score cutoff")
    ap.add_argument("--sam3-mask-threshold", type=float, default=0.5, help="SAM3 per-pixel mask cutoff (higher=tighter)")
    ap.add_argument("--sam3-carve", action="store_true", default=False,
                    help="carve see-through gaps (wheel spokes/mesh) as a final pass")
    ap.add_argument("--sam3-multirep", action="store_true", default=False,
                    help="run SAM3 on several image optimizations (raw/lift/CLAHE) and union the masks — "
                         "higher recall on dark/low-contrast subjects (~4x slower)")
    ap.add_argument("--sam3-parallel", action="store_true", default=False,
                    help="with --sam3-multirep: split the passes across both P100s (~1.75x faster)")
    ap.add_argument("--save-instances", default="",
                    help="with --sam3-multirep: also write a labeled instance map (per-object) — "
                         "16-bit label PNG here, plus a colourised _view.png next to it")
    ap.add_argument("--carve-depth", choices=["depthpro", "dav2"], default="depthpro",
                    help="depth model for the carve: depthpro (Apple Depth Pro, sharpest thin structures) "
                         "or dav2 (Depth Anything V2 zoomed). depthpro falls back to dav2 on failure.")
    ap.add_argument("--feather", choices=["adaptive", "none"], default="adaptive",
                    help="precise-mode border: adaptive = depth/defocus-aware soft alpha (tight on sharp "
                         "in-focus edges, wide where the background is far/out-of-focus); none = hard binary")
    ap.add_argument("--feather-max", type=float, default=20.0, help="max adaptive feather width (px)")
    ap.add_argument("--edge-snap", action="store_true", default=False,
                    help="final guided-filter pass: snap the mask alpha to real image edges (recovers thin "
                         "structures over busy backgrounds, kills matte-line halos). Works for removal+precise.")
    ap.add_argument("--edge-snap-radius", type=int, default=8, help="guided-filter edge neighbourhood (px)")
    ap.add_argument("--edge-snap-eps", type=float, default=1e-4, help="guided-filter eps on [0,1] (smaller=sharper)")
    ap.add_argument("--edge-snap-band", type=int, default=0,
                    help="only refine a ring N px around the boundary (keeps interior solid); 0=auto (3*radius)")
    ap.add_argument("--enhance", action="store_true", default=True, help="shadow-lift+CLAHE+sharpen for the detector only")
    ap.add_argument("--no-enhance", dest="enhance", action="store_false")
    ap.add_argument("--enh-gamma", type=float, default=0.55)
    ap.add_argument("--enh-clahe", type=float, default=3.0)
    ap.add_argument("--enh-sharp", type=float, default=0.6)
    ap.add_argument("--depth-map", default="", help="precomputed depth PNG (near=bright) to complete the crowd band")
    ap.add_argument("--depth-tol", type=float, default=0.12, help="depth-band half-width (0..1)")
    ap.add_argument("--depth-reach", type=int, default=90, help="px to grow the mask region for depth completion")
    ap.add_argument("--save-enh", default="")
    ap.add_argument("--save-vis", default="")
    a = ap.parse_args()

    full = cv2.imread(a.image, cv2.IMREAD_COLOR); H, W = full.shape[:2]
    if a.enhance:
        # auto-tune: darker/flatter images get a stronger shadow-lift + more local contrast so the detector can see
        g = cv2.cvtColor(full, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        med = float(np.median(g)); contrast = float(g.std())
        auto_gamma = float(np.clip(0.42 + med * 0.35, 0.42, a.enh_gamma))      # low brightness -> more lift (but not so much it washes out)
        auto_clahe = float(np.clip(a.enh_clahe + (0.20 - contrast) * 12, 2.0, 6.0))  # low contrast -> stronger CLAHE
        det = enhance_for_detection(full, auto_clahe, auto_gamma, a.enh_sharp)
        if a.save_enh: cv2.imwrite(a.save_enh, det)
        print(f"[enhance] auto gamma {auto_gamma:.2f} clahe {auto_clahe:.1f} (brightness {med:.2f}, contrast {contrast:.2f})")
    else:
        det = full
        print("[enhance] off")
    # detection stack: base + exposures + per-channel + edges; union detections for max recall
    det_stack = [det]
    for p in [x for x in a.det_images.split(",") if x.strip()]:
        im = cv2.imread(p.strip(), cv2.IMREAD_COLOR)
        if im is not None:
            if im.shape[:2] != (H, W): im = cv2.resize(im, (W, H))
            det_stack.append(im)
    if a.multi_channel:                                          # R / G / B / luminance variants
        det_stack += channel_variants(det)
        print("[multi-channel] +R +G +B +gray")
    if a.edges:                                                  # learned soft-edge map (TEED), on the enhanced frame
        try:
            det_stack.append(edge_map(det)); print("[edges] +TEED edge map (preprocessed)")
        except Exception as e:
            print("[edges] skipped:", str(e)[:80])
    print(f"[detect] union across {len(det_stack)} representation(s)")
    roi = None
    if a.roi:
        r = cv2.imread(a.roi, cv2.IMREAD_UNCHANGED)
        if r is not None:
            if r.ndim == 3: r = r[..., 3] if r.shape[2] == 4 else cv2.cvtColor(r, cv2.COLOR_BGR2GRAY)
            roi = cv2.resize(r, (W, H), interpolation=cv2.INTER_NEAREST) > 127

    # ---------- PASS 1: GLOBAL coarse detect (whole image) ----------
    gs = min(1.0, a.global_side / max(H, W))
    gW, gH = int(W*gs), int(H*gs)
    allb, alll = [], []
    if a.backend == "sam2":                                  # SAM3 does its own detection; skip DINO global pass
        for ei, dimg in enumerate(det_stack):                # union detections across the exposure stack
            g_img = Image.fromarray(cv2.cvtColor(cv2.resize(dimg, (gW, gH)), cv2.COLOR_BGR2RGB))
            b, l, sc = GS.detect(g_img, a.query, a.box_threshold, a.text_threshold)
            if len(b): allb.append(b / gs); alll += list(l)
            if len(det_stack) > 1: print(f"   exposure {ei+1}: {len(b)} boxes")
    gb = np.concatenate(allb, 0) if allb else np.zeros((0, 4))
    gl = alll
    det = det_stack[0]                                       # base enhanced image for the local/segment passes (not a channel)
    if roi is not None and len(gb):                          # keep boxes overlapping the ROI (center OR inside its bbox)
        rys, rxs = np.where(roi)
        rb = (rxs.min(), rys.min(), rxs.max(), rys.max()) if len(rxs) else (0, 0, W, H)
        keep = []
        for i, b in enumerate(gb):
            cx, cy = int((b[0]+b[2])/2), int((b[1]+b[3])/2)
            inside_bbox = rb[0] <= cx <= rb[2] and rb[1] <= cy <= rb[3]
            on_roi = 0 <= cy < H and 0 <= cx < W and roi[cy, cx]
            if on_roi or inside_bbox: keep.append(i)
        if keep:                                             # fallback: never zero out if boxes exist
            gb = gb[keep]; gl = [gl[i] for i in keep]
    print(f"[global] {W}x{H} -> {gW}x{gH} | detected {len(gb)}: " +
          ", ".join(sorted(set(gl))) if len(gb) else "[global] nothing")
    mask = np.zeros((H, W), np.uint8)
    total_fine = 0
    Dc_pre = None                                            # depth computed early for combine, reused later
    inst_labels = None                                       # labeled instance map (per-object) from multi-rep
    if a.backend in ("sam3", "both"):
        # ---------- SAM3 core: concept -> instance masks (semantic; pairs with the LLM/VLM loop) ----------
        from pelib import sam3 as S3
        concepts = [c.strip() for c in a.query.split(".") if c.strip()] or [a.query.strip()]
        inst = []                                            # list of (bool mask, score)
        if a.sam3_multirep:                                  # multi-representation: union over image optimizations
            inst = S3.multirep_instances(det, a.query, threshold=a.sam3_threshold,
                                         mask_threshold=a.sam3_mask_threshold, parallel=a.sam3_parallel)
            print(f"[sam3] multi-rep union{' (2-GPU)' if a.sam3_parallel else ''}: "
                  f"{len(inst)} pooled instance(s) over 4 representations")
        else:
            for c in concepts:
                try:
                    ms, ss = S3.instances(det, text=c, threshold=a.sam3_threshold,
                                          mask_threshold=a.sam3_mask_threshold)
                    inst += list(zip(ms, ss))
                    print(f"[sam3] '{c}': {len(ms)} instance(s)" + (f" scores {[round(s,2) for s in ss][:6]}" if ss else ""))
                except Exception as e:
                    print(f"[sam3] '{c}' failed:", str(e)[:80])
        if roi is not None and inst:                         # keep instances overlapping the ROI
            kept = [(m, s) for (m, s) in inst if (m & roi).sum() > 0]
            if kept: inst = kept
        if a.sam3_multirep and inst:
            # ---------- COMBINE into DISTINCT OBJECTS: binary union (removal) + labeled instances (per-object) ----
            raw = np.zeros((H, W), bool)
            for m, s in inst: raw |= m
            Dc_pre = zoom_depth(full, raw, a.carve_depth, a.depth_map)
            binm, inst_labels, kept = S3.combine_instances(inst, full, depth=Dc_pre)
            mask = binm.astype(np.uint8)
            total_fine = len(kept)
            print(f"[sam3] combine_instances: {len(kept)} distinct object(s), {100*mask.mean():.2f}% "
                  f"(from {len(inst)} pooled, raw union {100*raw.mean():.2f}%)")
        elif a.backend == "both" and inst:
            # ---------- SAM2 tightens each SAM3 instance via its bbox (geometric crisp edges) ----------
            refined = []
            for m, s in inst:
                ys, xs = np.where(m)
                if not len(xs): continue
                box = np.array([[float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]])
                try:
                    seg = GS.segment(Image.fromarray(cv2.cvtColor(full, cv2.COLOR_BGR2RGB)), box).astype(bool)
                    # keep SAM2's refine only if it agrees with SAM3 (avoid box grabbing neighbours)
                    inter = (seg & m).sum(); ref = seg if inter > 0.35 * max(1, m.sum()) else m
                    refined.append(ref)
                except Exception:
                    refined.append(m)
            print(f"[both] SAM2 tightened {len(refined)} SAM3 instance(s)")
            for m in refined: mask = np.maximum(mask, m.astype(np.uint8))
        else:
            for m, s in inst: mask = np.maximum(mask, m.astype(np.uint8))
        total_fine = len(inst)
        print(f"[{a.backend}] {total_fine} instance(s) from concepts {concepts}")
    elif len(gb):
        # ---------- PASS 2: LOCAL fine detect+segment per region cluster (SAM2 geometric core) ----------
        clusters = cluster_boxes(gb, a.cluster_gap)
        print(f"[local] {len(clusters)} region cluster(s)")
        for ci, cb in enumerate(clusters):
            pad = 40
            x0, y0 = max(0, int(cb[0]-pad)), max(0, int(cb[1]-pad))
            x1, y1 = min(W, int(cb[2]+pad)), min(H, int(cb[3]+pad))
            crop = det[y0:y1, x0:x1]; ch, cw = crop.shape[:2]
            us = max(1.0, a.local_side / max(ch, cw))         # upscale small regions for the fine pass
            cw2, ch2 = int(cw*us), int(ch*us)
            crop_u = cv2.resize(crop, (cw2, ch2), interpolation=cv2.INTER_CUBIC)
            cimg = Image.fromarray(cv2.cvtColor(crop_u, cv2.COLOR_BGR2RGB))
            lb, ll, lsc = GS.detect(cimg, a.query, max(0.18, a.box_threshold-0.04), max(0.15, a.text_threshold-0.03))
            if not len(lb): continue
            seg = GS.segment(cimg, lb).astype(np.uint8)       # crop-upscaled coords
            seg = cv2.resize(seg, (cw, ch), interpolation=cv2.INTER_NEAREST)
            mask[y0:y1, x0:x1] = np.maximum(mask[y0:y1, x0:x1], seg)
            total_fine += len(lb)
            print(f"   cluster {ci+1}: fine-detected {len(lb)} ({cw}x{ch} @x{us:.1f})")
    # ---------- DEPTH: gate (precise) or complete (removal) ----------
    if a.depth_map and mask.sum() > 500:
        D = cv2.imread(a.depth_map, cv2.IMREAD_GRAYSCALE)
        if D is not None:
            if D.shape != (H, W): D = cv2.resize(D, (W, H), interpolation=cv2.INTER_LINEAR)
            D = D.astype(np.float32) / 255.0
            pd = float(np.median(D[mask > 0]))
            band = (np.abs(D - pd) < a.depth_tol).astype(np.uint8)
            before = 100 * mask.mean()
            if a.mode == "precise":
                # AND-gate: keep only object pixels at the foreground depth (cuts SAM leaks into bg trees/wall)
                mask = ((mask > 0) & (band > 0)).astype(np.uint8)
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
                print(f"[depth] gate @ {pd:.2f}±{a.depth_tol} -> {before:.1f}% -> {100*mask.mean():.1f}% (tight)")
            else:
                region = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*a.depth_reach+1, 2*a.depth_reach+1)))
                if roi is not None: region = ((region > 0) | roi).astype(np.uint8)
                add = ((band > 0) & (region > 0) & (mask == 0)).astype(np.uint8)
                add = cv2.morphologyEx(add, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
                mask = ((mask > 0) | (add > 0)).astype(np.uint8)
                print(f"[depth] complete @ {pd:.2f}±{a.depth_tol} -> {before:.1f}% -> {100*mask.mean():.1f}% (region)")

    # ---------- SEMANTIC PROTECT: subtract structures to keep (ramp, etc.) — SAM2 path only ----------
    # (SAM3 masks only the named concepts, so it never grabs the ramp; the GroundingDINO subtract is
    #  redundant for sam3/both and just adds a load + cost.)
    if a.backend == "sam2" and a.protect_query.strip():
        pb, pl, psc = GS.detect(Image.fromarray(cv2.cvtColor(det, cv2.COLOR_BGR2RGB)),
                                 a.protect_query, 0.20, 0.15)
        if len(pb):
            pseg = GS.segment(Image.fromarray(cv2.cvtColor(det, cv2.COLOR_BGR2RGB)), pb).astype(np.uint8)
            pseg = cv2.dilate(pseg, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)))
            keep = mask.copy()
            mask = (mask & (1 - pseg)).astype(np.uint8)
            print(f"[protect] '{a.protect_query.split('.')[0]}' ({', '.join(sorted(set(pl)))}) shielded "
                  f"{100*((keep>0)&(pseg>0)).mean():.2f}% of frame")

    subject = 0 < total_fine <= a.birefnet_max_objs              # coherent subject (few objects) vs a crowd
    # BiRefNet/ViTMatte exist to clean up SAM1/SAM2 blobbiness; SAM3 masks are already precise, so these
    # only add cost + fatten the border -> skip them (and the dilate) on the sam3/both path.
    sam_refine = a.backend == "sam2"
    # ---------- BiRefNet: complete a coherent SUBJECT (thin parts SAM missed), gated to the detected region ----------
    if sam_refine and a.birefnet_mode != "off" and mask.sum() > 500:
        use_bire = a.birefnet_mode == "on" or (a.birefnet_mode == "auto" and subject)
        if use_bire:
            try:
                bm = birefnet_mask(full)
                region = cv2.dilate((mask > 0).astype(np.uint8),
                                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*a.birefnet_reach+1, 2*a.birefnet_reach+1)))
                add = ((bm > 127) & (region > 0)).astype(np.uint8)
                before = 100 * mask.mean()
                mask = ((mask > 0) | (add > 0)).astype(np.uint8)
                print(f"[birefnet] whole-subject union {before:.1f}% -> {100*mask.mean():.1f}% (wraps thin parts)")
            except Exception as e:
                print("[birefnet] skipped:", str(e)[:80])

    # ---------- ViTMatte: crisp edges — for a coherent SUBJECT only (over-erodes a crowd of tiny objects) ----------
    if sam_refine and a.matte and subject and mask.sum() > 500:
        try:
            alpha = PM.vitmatte(cv2.cvtColor(full, cv2.COLOR_BGR2RGB), (mask > 0).astype(np.float32),
                                band=a.matte_band)
            mask = (alpha > 0.5).astype(np.uint8)
            print(f"[matte] ViTMatte crisp edges -> coverage {100*mask.mean():.1f}%")
        except Exception as e:
            print("[matte] skipped:", str(e)[:80])

    dilate = a.dilate if a.backend == "sam2" else 0          # SAM3 is precise -> don't fatten the border
    if dilate > 0:
        mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*dilate+1, 2*dilate+1)))

    # ZOOMED depth on the mask bbox (Depth Pro / DA-V2): resolves thin see-through structure for the carve
    # AND drives the adaptive feather. Computed once, reused. Cropping tight = more px per thin feature.
    want_depth = (a.sam3_carve or a.feather == "adaptive" or a.backend in ("sam3", "both")) and mask.sum() > 500
    Dc = Dc_pre if (Dc_pre is not None and Dc_pre[mask > 0].any()) else None   # reuse smart-combine's depth
    if want_depth and Dc is None:
        Dc = zoom_depth(full, mask > 0, a.carve_depth, a.depth_map)

    # DEPTH-AWARE HOLE-FILL (sam3/both, not carving): an enclosed hole is a dark DROP-OUT on a solid figure
    # only if it sits at OBJECT depth (near) — then fill it. A hole at BACKGROUND depth (far) is a real gap
    # (between arms, background seen through the figure) — leave it. Replaces the dumb flood-fill-everything.
    if (not a.sam3_carve) and (not a.sam3_multirep) and a.backend in ("sam3", "both") and Dc is not None and mask.sum() > 500:
        d = Dc.astype(np.float32) / 255.0
        fg = float(np.percentile(d[mask > 0], 60))           # object foreground depth
        inv = (mask == 0).astype(np.uint8)
        nlab, lab = cv2.connectedComponents(inv, 8)
        border = set(np.unique(np.concatenate([lab[0, :], lab[-1, :], lab[:, 0], lab[:, -1]])))
        filled = 0
        for c in range(1, nlab):
            if c in border:
                continue                                     # touches image edge = real background, not a hole
            reg = (lab == c)
            if float(np.median(d[reg])) >= fg - 0.14:        # near = object drop-out -> fill; far = real gap -> keep
                mask[reg] = 1; filled += int(reg.sum())
        if filled:
            print(f"[fill] depth-aware: filled {filled} object drop-out px, kept far/background gaps")

    if a.sam3_carve and mask.sum() > 500:                    # final pass: open see-through gaps (spokes/mesh)
        from pelib import sam3 as S3
        before = 100 * mask.mean()
        if Dc is not None:
            mask = S3.carve_continuity((mask > 0), full, Dc).astype(np.uint8)
            method = "continuity"
        else:
            print("[carve] WARNING: no depth map -- falling back to the weaker depthless "
                  "heuristic carve (2026-07-09: safe since the sky_L fix, but continuity "
                  "with a depth map is markedly cleaner; check why Depth Pro was unavailable)")
            mask = S3.carve_seethrough((mask > 0), full).astype(np.uint8)
            method = "heuristic"
        print(f"[carve] see-through ({method}) {before:.1f}% -> {100*mask.mean():.1f}%")

    # OUTPUT border: precise/selection masks get a depth+defocus-aware SOFT alpha (tight on sharp in-focus
    # edges & across occluders, wide where the background is far/out-of-focus). Removal stays binary (the
    # fill/erase engine does its own occlusion-aware feather on the returned patch).
    zoom_done = False
    if a.zoom_carve and mask.sum() > 500:                       # depth-tiled carve of lattice structure (spokes/mesh)
        try:
            from pelib import zoomcarve as ZC, depth as PD
            PD.prewarm_dpro(("cuda:0", "cuda:1"))
            alpha = ZC.zoom_carve_tiled(
                full, (mask * 255).astype(np.uint8),
                depth_fn=lambda c, dev: PD.depth_pro_metric(c, dev),
                global_depth_fn=lambda c: PD.depth_pro_metric(c, "cuda:0", max_side=1024),
                depth_batch_fn=lambda cs, dev: PD.depth_pro_metric_batch(cs, dev),
                tile=a.zoom_tile, overlap=0.30, parallel=True)
            out_mask = np.clip(alpha * 255, 0, 255).astype(np.uint8)
            zoom_done = True
            print(f"[zoom-carve] depth-tiled carve {100*mask.mean():.1f}% -> {100*(alpha>0.5).mean():.1f}%")
        except Exception as e:
            print("[zoom-carve] skipped:", str(e)[:100])
    matte_done = False
    matte_ok = False
    if (not zoom_done) and a.matte_refine and mask.sum() > 500:  # gate on mask COHERENCE (few connected blobs), not
        n_cc, cc_lbl, cc_stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), 8)
        big = [i for i in range(1, n_cc) if cc_stats[i, cv2.CC_STAT_AREA] > 0.02 * mask.sum()]
        matte_ok = 0 < len(big) <= a.matte_max_cc              # SAM3 instance count (a subject fragments into many)
        if not matte_ok:
            print(f"[matte-refine] skipped: {len(big)} large blobs > matte-max-cc {a.matte_max_cc} (crowd, would over-erode)")
    if matte_ok:                                                # ViTMatte soft-alpha refine (any backend), free GPU
        try:
            alpha = PM.vitmatte(cv2.cvtColor(full, cv2.COLOR_BGR2RGB), (mask > 0).astype(np.float32),
                                band=a.matte_refine_band, maxside=a.matte_maxside, firm=1.0)
            out_mask = np.clip(alpha * 255, 0, 255).astype(np.uint8)
            matte_done = True
            print(f"[matte-refine] ViTMatte soft alpha (band {a.matte_refine_band}, maxside {a.matte_maxside})")
        except Exception as e:
            print("[matte-refine] skipped:", str(e)[:80])
    if not matte_done and not zoom_done:
        if a.feather == "adaptive" and a.mode == "precise" and Dc is not None and mask.sum() > 500:
            from pelib.imaging import adaptive_feather
            alpha = adaptive_feather(full, mask * 255, depth=Dc, max_f=a.feather_max)
            out_mask = np.clip(alpha * 255, 0, 255).astype(np.uint8)
            print(f"[feather] adaptive depth/defocus border (max {a.feather_max:.0f}px)")
        else:
            out_mask = (mask * 255).astype(np.uint8)
        if a.edge_snap and mask.sum() > 500:                    # snap the (binary or feathered) alpha to real edges
            from pelib.imaging import guided_edge_snap
            asnap = guided_edge_snap(full, out_mask, radius=a.edge_snap_radius, eps=a.edge_snap_eps,
                                     band=a.edge_snap_band)
            out_mask = np.clip(asnap * 255, 0, 255).astype(np.uint8)
            print(f"[edge-snap] guided-filter edge refine (r={a.edge_snap_radius}, eps={a.edge_snap_eps:g})")
    cv2.imwrite(a.out, out_mask)
    print(f"done [{a.mode}] -> {a.out} | fine instances {total_fine} | coverage {100*mask.mean():.1f}%")
    if a.save_vis:
        vis = cv2.convertScaleAbs(full, alpha=2.4, beta=12)
        vis[mask>0] = (0.4*vis[mask>0] + 0.6*np.array([0,0,255])).astype(np.uint8)
        cv2.imwrite(a.save_vis, vis)
    if a.save_instances and inst_labels is not None:           # per-object: 16-bit label PNG + colour view
        cv2.imwrite(a.save_instances, inst_labels.astype(np.uint16))
        nlab = int(inst_labels.max())
        rng = np.random.default_rng(0)
        colors = np.zeros((nlab + 1, 3), np.uint8)
        colors[1:] = rng.integers(60, 256, (nlab, 3))
        view = cv2.convertScaleAbs(full, alpha=1.6, beta=8)
        col = colors[inst_labels]
        m3 = (inst_labels > 0)[..., None]
        view = np.where(m3, (0.45 * view + 0.55 * col).astype(np.uint8), view)
        cv2.imwrite(a.save_instances.rsplit(".", 1)[0] + "_view.png", view)
        print(f"[instances] {nlab} distinct object(s) -> {a.save_instances} (+_view.png)")


if __name__ == "__main__":
    main()
