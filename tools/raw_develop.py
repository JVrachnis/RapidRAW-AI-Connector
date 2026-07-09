#!/usr/bin/env python3
"""
raw_develop.py - develop a camera RAW into a multi-EV exposure stack (+ EXIF sidecar).

One develop to LINEAR 16-bit, then apply EV multipliers -> a bracket from a SINGLE RAW.
14-bit sensor latitude recovers shadow detail an 8-bit JPEG can't -- so a +3/+4 EV frame
reveals shadow-blended subjects for the detector, while 0/-2 EV keep highlights. Feeds:
  - masking: union detections across the stack (max recall)
  - inpaint: true HDR/linear context + correct grain/tone
Also dumps camera EXIF (aperture/focal/ISO/focus) for the realism finish.

Usage: raw_develop.py RAW.ARW OUTDIR [--evs -2,0,2,4] [--wb camera|auto] [--maxside 0]
Requires rawpy (libraw). Run where the RAW lives (Fedora).
"""
from __future__ import annotations
import argparse, os, json, subprocess
import numpy as np, cv2, rawpy


def srgb_encode(x):
    x = np.clip(x, 0, 1)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(x, 1 / 2.4) - 0.055)


def develop_stack(path, evs, wb="camera", maxside=0):
    raw = rawpy.imread(path)
    rgb = raw.postprocess(use_camera_wb=(wb == "camera"), use_auto_wb=(wb == "auto"),
                          no_auto_bright=True, output_bps=16, gamma=(1, 1),
                          output_color=rawpy.ColorSpace.sRGB)
    lin = rgb.astype(np.float32) / 65535.0
    if maxside and max(lin.shape[:2]) > maxside:
        s = maxside / max(lin.shape[:2])
        lin = cv2.resize(lin, (int(lin.shape[1]*s), int(lin.shape[0]*s)), interpolation=cv2.INTER_AREA)
    out = {}
    for ev in evs:
        bgr = cv2.cvtColor((srgb_encode(lin * (2.0 ** ev)) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        out[ev] = bgr
    return out


def read_exif(path):
    keys = ["Model", "LensModel", "FNumber", "ApertureValue", "FocalLength",
            "FocalLengthIn35mmFormat", "ISO", "ExposureTime", "FocusDistance",
            "HyperfocalDistance", "DateTimeOriginal"]
    try:
        r = subprocess.run(["exiftool", "-json", *[f"-{k}" for k in keys], path],
                           capture_output=True, text=True, timeout=30)
        return json.loads(r.stdout)[0] if r.stdout.strip() else {}
    except Exception:
        return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raw"); ap.add_argument("outdir")
    ap.add_argument("--evs", default="-2,0,2,4")
    ap.add_argument("--wb", default="camera")
    ap.add_argument("--maxside", type=int, default=0)
    ap.add_argument("--montage", action="store_true", default=True)
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    evs = [float(x) for x in a.evs.split(",")]
    stack = develop_stack(a.raw, evs, a.wb, a.maxside)
    base = os.path.splitext(os.path.basename(a.raw))[0]
    paths = {}
    for ev, img in stack.items():
        p = os.path.join(a.outdir, f"{base}_EV{ev:+g}.jpg")
        cv2.imwrite(p, img, [cv2.IMWRITE_JPEG_QUALITY, 94]); paths[ev] = p
    exif = read_exif(a.raw)
    with open(os.path.join(a.outdir, f"{base}_exif.json"), "w") as f:
        json.dump(exif, f, indent=2)
    if a.montage:
        thumbs = [cv2.resize(stack[ev], (0, 0), fx=520/stack[ev].shape[1], fy=520/stack[ev].shape[1]) for ev in evs]
        h = min(t.shape[0] for t in thumbs)
        row = np.hstack([t[:h] for t in thumbs])
        for i, ev in enumerate(evs):
            cv2.putText(row, f"EV{ev:+g}", (i*520+12, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 255), 2)
        cv2.imwrite(os.path.join(a.outdir, f"{base}_stack.jpg"), row, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print("developed", len(evs), "EV ->", a.outdir)
    print("exif:", {k: exif.get(k) for k in ("Model", "LensModel", "FNumber", "FocalLength", "ISO", "ExposureTime")})


if __name__ == "__main__":
    main()
