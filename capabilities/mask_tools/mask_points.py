#!/usr/bin/env python3
"""mask_points.py - SAM2/SAM3 point-prompt masking for the rr-ai-gateway.
Runs in the ComfyUI venv on inferno; imports the SAM2 predictor plumbing from
grounded_sam.py in --tools-dir (default ~/comfy).

Usage:
  mask_points.py IMAGE --points '[[x,y,1],[x,y,0]]' --backend sam2 --out mask.png
Point labels: 1 = foreground, 0 = background. Coordinates in image pixels.
"""
import argparse, json, os, sys
import numpy as np
from PIL import Image

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--points", required=True, help="JSON [[x,y,label],...]")
    ap.add_argument("--backend", choices=["sam2", "sam3"], default="sam2")
    ap.add_argument("--tools-dir", default=os.path.expanduser("~/comfy"))
    ap.add_argument("--out", default="mask.png")
    args = ap.parse_args()

    if args.backend == "sam3":
        print("warning: sam3 backend not wired for point prompts; using sam2")

    sys.path.insert(0, args.tools_dir)
    import grounded_sam as GS  # provides the cached SAM2 predictor plumbing

    image = Image.open(args.image).convert("RGB")
    pts = json.loads(args.points)
    coords = np.array([[p[0], p[1]] for p in pts], dtype=np.float32)
    labels = np.array([int(p[2]) for p in pts], dtype=np.int32)

    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    pred = SAM2ImagePredictor(build_sam2(GS.SAM2_CFG, GS.SAM2_CKPT, device=dev))
    pred.set_image(np.array(image))
    with torch.no_grad():
        masks, scores, _ = pred.predict(point_coords=coords, point_labels=labels,
                                        multimask_output=True)
    best = int(np.argmax(scores))
    mask = (masks[best].astype(np.float32) * 255).astype(np.uint8)
    Image.fromarray(mask, mode="L").save(args.out)
    print(f"detected: point-selection {float(scores[best]):.2f}")

if __name__ == "__main__":
    main()
