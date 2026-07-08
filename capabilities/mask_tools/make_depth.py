#!/usr/bin/env python3
"""make_depth.py - Depth Pro depth map for the rr-ai-gateway carve path.
Runs in the ComfyUI venv; imports pelib.depth from --tools-dir.

Usage:
  make_depth.py IMAGE --out depth.png [--tools-dir ~/comfy]
Writes a near=bright uint8 PNG at image resolution.
"""
import argparse, os, sys
import cv2

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--tools-dir", default=os.path.expanduser("~/comfy"))
    ap.add_argument("--out", default="depth.png")
    a = ap.parse_args()

    sys.path.insert(0, a.tools_dir)
    from pelib import depth as D

    bgr = cv2.imread(a.image)
    if bgr is None:
        raise SystemExit(f"cannot read {a.image}")
    d = D.depth_pro(bgr)
    cv2.imwrite(a.out, d)
    print(f"depth: {d.shape[1]}x{d.shape[0]} -> {a.out}")

if __name__ == "__main__":
    main()
