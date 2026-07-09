#!/usr/bin/env python3
"""
grounded_sam.py - semantic, object-aware masking: text query -> boxes (Grounding DINO)
-> precise masks (SAM). Optionally refine each mask edge with ViTMatte.

Only masks the named objects, so see-through background (e.g. a spectator behind a wheel)
is never included. Also prints what was detected (image understanding).

Usage:
  python grounded_sam.py IMAGE --query "bicycle. person." --out mask.png
      [--box-threshold 0.3] [--text-threshold 0.25] [--matte]
"""
from __future__ import annotations
import argparse, sys, os
import numpy as np, cv2, torch
from PIL import Image
from transformers import (AutoProcessor, GroundingDinoForObjectDetection,
                          SamModel, SamProcessor)

GD_ID = "IDEA-Research/grounding-dino-base"
SAM_ID = "facebook/sam-vit-large"
SAM2_ID = os.environ.get("SAM2_ID", "facebook/sam2.1-hiera-large")
# SAM backend: 'sam2' (cleaner masks, default) or 'sam1' (legacy sam-vit-large fallback)
SAM_BACKEND = os.environ.get("MASK_SAM", "sam2").lower()
DEV = "cuda" if torch.cuda.is_available() else "cpu"
_SAM2 = {}  # lazy singleton cache (SAM2-large is heavy; segment() is called many times per image)

def detect(image, query, box_thr, text_thr):
    proc = AutoProcessor.from_pretrained(GD_ID)
    model = GroundingDinoForObjectDetection.from_pretrained(GD_ID).to(DEV).eval()
    q = query if query.strip().endswith(".") else query.strip() + "."
    inp = proc(images=image, text=q, return_tensors="pt").to(DEV)
    with torch.no_grad(): out = model(**inp)
    res = proc.post_process_grounded_object_detection(
        out, inp.input_ids, threshold=box_thr, text_threshold=text_thr,
        target_sizes=[image.size[::-1]])[0]
    del model; torch.cuda.empty_cache()
    return res["boxes"].cpu().numpy(), res["labels"], res["scores"].cpu().numpy()

def _segment_sam1(image, boxes):
    proc = SamProcessor.from_pretrained(SAM_ID)
    model = SamModel.from_pretrained(SAM_ID).to(DEV).eval()
    inp = proc(image, input_boxes=[[b.tolist() for b in boxes]], return_tensors="pt").to(DEV)
    with torch.no_grad(): out = model(**inp)
    masks = proc.image_processor.post_process_masks(
        out.pred_masks.cpu(), inp["original_sizes"].cpu(), inp["reshaped_input_sizes"].cpu())[0]
    scores = out.iou_scores.cpu().numpy()[0]           # [n_boxes, 3]
    del model; torch.cuda.empty_cache()
    # pick best of the 3 proposals per box, union them
    W, H = image.size; union = np.zeros((H, W), bool)
    for i in range(masks.shape[0]):
        best = int(scores[i].argmax())
        union |= masks[i, best].numpy().astype(bool)
    return union


SAM2_CKPT = os.path.expanduser(os.environ.get("SAM2_CKPT", "~/tracking/models/sam2.1_hiera_large.pt"))
SAM2_CFG = os.environ.get("SAM2_CFG", "configs/sam2.1/sam2.1_hiera_l.yaml")


def _segment_sam2(image, boxes):
    """SAM 2.1 segmenter via the NATIVE sam2 SAM2ImagePredictor (not transformers — the HF video
    checkpoint corrupts the image model and speckles the mask). Sharper edges, fewer holes, better
    thin parts. Same box-prompt contract as SAM1. Predictor cached; set_image() re-run per frame."""
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    if "pred" not in _SAM2:
        _SAM2["pred"] = SAM2ImagePredictor(build_sam2(SAM2_CFG, SAM2_CKPT, device=DEV))
    pred = _SAM2["pred"]
    rgb = np.array(image.convert("RGB")); H, W = rgb.shape[:2]
    pred.set_image(rgb)
    box_arr = np.asarray([b.tolist() for b in boxes], dtype=np.float32)  # Nx4 xyxy
    with torch.no_grad():
        masks, scores, _ = pred.predict(box=box_arr, multimask_output=False)
    mm = masks if masks.ndim == 3 else masks[:, 0]     # -> [N,H,W]
    union = np.zeros((H, W), bool)
    for i in range(mm.shape[0]):
        union |= mm[i].astype(bool)
    # speckle cleanup: SAM2's sharp response traces amplified shadow-noise into flecks. Drop tiny
    # islands (< 0.02% of frame) and close pinholes so the mask is contiguous like SAM1's fill.
    u8 = union.astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(u8, 8)
    if n > 1:
        min_area = max(64, int(0.0002 * H * W))
        keep = np.zeros_like(u8)
        for c in range(1, n):
            if stats[c, cv2.CC_STAT_AREA] >= min_area:
                keep[lab == c] = 1
        u8 = keep
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    u8 = cv2.morphologyEx(u8, cv2.MORPH_CLOSE, k)      # fill pinholes
    return u8.astype(bool)


def segment(image, boxes):
    """Box-prompted segmentation. Dispatches to SAM2 (default) or SAM1 via MASK_SAM env.
    Falls back to SAM1 if SAM2 raises (e.g. weights unavailable)."""
    if SAM_BACKEND == "sam2":
        try:
            return _segment_sam2(image, boxes)
        except Exception as e:
            print(f"[segment] SAM2 failed ({str(e)[:80]}), falling back to SAM1")
    return _segment_sam1(image, boxes)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image"); ap.add_argument("--query", required=True)
    ap.add_argument("--out", default="grounded_mask.png")
    ap.add_argument("--box-threshold", type=float, default=0.3)
    ap.add_argument("--text-threshold", type=float, default=0.25)
    ap.add_argument("--matte", action="store_true", help="refine edges with vitmatte_refine")
    ap.add_argument("--main-subject", action="store_true",
                    help="keep only the primary object cluster (anchor + overlapping boxes); drops background crowds")
    a = ap.parse_args()
    image = Image.open(a.image).convert("RGB")
    boxes, labels, scores = detect(image, a.query, a.box_threshold, a.text_threshold)
    print(f"[understanding] detected {len(boxes)} objects:",
          ", ".join(f"{l}({s:.2f})" for l, s in zip(labels, scores)) or "none")
    if len(boxes) == 0: sys.exit("no objects matched the query")

    if a.main_subject and len(boxes) > 1:
        def inter(a1, b1):  # intersection area / min-box area
            x0=max(a1[0],b1[0]); y0=max(a1[1],b1[1]); x1=min(a1[2],b1[2]); y1=min(a1[3],b1[3])
            iw=max(0,x1-x0); ih=max(0,y1-y0); ia=iw*ih
            aa=(a1[2]-a1[0])*(a1[3]-a1[1]); bb=(b1[2]-b1[0])*(b1[3]-b1[1])
            return ia/max(1,min(aa,bb))
        areas=[(b[2]-b[0])*(b[3]-b[1]) for b in boxes]
        anchor=int(np.argmax(areas))                       # biggest object = the subject cluster
        keep=[i for i in range(len(boxes)) if i==anchor or inter(boxes[anchor],boxes[i])>0.05]
        boxes=boxes[keep]; labels=[labels[i] for i in keep]
        print(f"[main-subject] kept {len(boxes)}:", ", ".join(labels))
    mask = segment(image, boxes)
    cv2.imwrite(a.out, (mask*255).astype(np.uint8))
    print("mask ->", a.out, "| coverage %.1f%%" % (100*mask.mean()))
    if a.matte:
        os.system(f"~/comfy/vitmatte_refine.py {a.image} {a.out} {a.out.replace('.png','_alpha.png')}")

if __name__ == "__main__":
    main()
