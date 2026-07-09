#!/usr/bin/env python3
"""
vitmatte_refine.py - alpha-perfect edge refinement of a coarse BiRefNet mask using
ViTMatte trimap matting, at full resolution.

Pipeline: coarse mask -> trimap (FG/unknown/BG bands) -> crop to subject bbox (so full-res
matting fits VRAM) -> ViTMatte alpha -> paste back into full-res canvas.

Usage: vitmatte_refine.py FULLRES_IMAGE COARSE_MASK OUT_ALPHA [--model ~/models/vitmatte-small]
       [--band 24] [--maxside 2048]
"""
from __future__ import annotations
import argparse, os
import numpy as np, cv2, torch
from PIL import Image
from transformers import VitMatteForImageMatting, VitMatteImageProcessor

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image"); ap.add_argument("mask"); ap.add_argument("out")
    ap.add_argument("--model", default=os.path.expanduser("~/models/vitmatte-small"))
    ap.add_argument("--band", type=int, default=24)      # unknown-band half-width (px, at working res)
    ap.add_argument("--maxside", type=int, default=2048) # cap the crop long side (VRAM guard)
    ap.add_argument("--pad", type=int, default=64)
    a = ap.parse_args()

    img = cv2.cvtColor(cv2.imread(a.image, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    H, W = img.shape[:2]
    m = cv2.imread(a.mask, cv2.IMREAD_GRAYSCALE)
    if m.shape != (H, W): m = cv2.resize(m, (W, H), interpolation=cv2.INTER_LINEAR)
    mb = (m > 128).astype(np.uint8)

    # subject bbox (+pad) so ViTMatte runs on the region only
    ys, xs = np.where(mb > 0)
    if len(xs) == 0: raise SystemExit("empty coarse mask")
    x0, x1 = max(0, xs.min()-a.pad), min(W, xs.max()+a.pad)
    y0, y1 = max(0, ys.min()-a.pad), min(H, ys.max()+a.pad)
    crop = img[y0:y1, x0:x1]; cm = m[y0:y1, x0:x1]
    ch, cw = crop.shape[:2]

    # downscale crop if it exceeds the VRAM cap, remember scale to upsample alpha back
    s = min(1.0, a.maxside/max(ch, cw))
    if s < 1.0:
        crop_s = cv2.resize(crop, (int(cw*s), int(ch*s)), interpolation=cv2.INTER_AREA)
        cm_s   = cv2.resize(cm,   (int(cw*s), int(ch*s)), interpolation=cv2.INTER_LINEAR)
    else:
        crop_s, cm_s = crop, cm

    # trimap: FG=1 (eroded), BG=0 (outside dilated), unknown=0.5 (band around edge)
    b = a.band
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*b+1, 2*b+1))
    cmb = (cm_s > 128).astype(np.uint8)
    fg = cv2.erode(cmb, k); bgo = cv2.dilate(cmb, k)
    tri = np.full(cm_s.shape, 0.5, np.float32)
    tri[fg > 0] = 1.0; tri[bgo == 0] = 0.0

    proc = VitMatteImageProcessor.from_pretrained(a.model)
    model = VitMatteForImageMatting.from_pretrained(a.model, torch_dtype=torch.float32).to("cuda").eval()
    inputs = proc(images=Image.fromarray(crop_s), trimaps=Image.fromarray((tri*255).astype(np.uint8)), return_tensors="pt")
    inputs = {kk: v.to("cuda") for kk, v in inputs.items()}
    with torch.no_grad():
        alpha = model(**inputs).alphas[0, 0].float().cpu().numpy()
    alpha = alpha[:crop_s.shape[0], :crop_s.shape[1]]      # processor may pad

    # --- defringe: kill bright halo on a dark subject (bg bleeding through edges) ---
    lum = (crop_s.astype(np.float32) @ np.array([0.2126,0.7152,0.0722],np.float32)) / 255.0
    fg_core = alpha > 0.9
    if fg_core.sum() > 50:
        fg_lum = float(np.median(lum[fg_core]))
        edge = (alpha > 0.03) & (alpha < 0.97)
        excess = np.clip(lum - (fg_lum + 0.12), 0, 1)      # how much brighter than FG
        supp = 1.0 - np.clip(excess * 4.0, 0, 0.85)         # bright edge -> reduce alpha
        alpha = np.where(edge, alpha * supp, alpha)
    # firm up the mid-alpha so edges are decisive, not mushy
    alpha = np.clip((alpha - 0.5) * 1.25 + 0.5, 0, 1)

    if s < 1.0:
        alpha = cv2.resize(alpha, (cw, ch), interpolation=cv2.INTER_LINEAR)

    # paste alpha back into full-res canvas
    out = np.zeros((H, W), np.float32)
    out[y0:y1, x0:x1] = np.clip(alpha, 0, 1)
    cv2.imwrite(a.out, (out*255+0.5).astype(np.uint8))
    print("vitmatte alpha ->", a.out, "| crop", (cw, ch), "scale %.2f" % s, "| fg%% %.1f" % (100*(out > 0.5).mean()))

if __name__ == "__main__":
    main()
