#!/usr/bin/env python3
"""
mask_agentic.py - self-correcting semantic masking with an LLM + VLM loop, over any backend.

Pipeline:
  1. LLM (LFM2.5) expands a natural-language TARGET ("the camera gear") into concrete detector
     concepts ("camera. tripod. monitor.") — solves SAM3's "only masks what you name" limitation.
  2. mask_c2f runs with the chosen --backend (sam2 | sam3 | both) on those concepts.
  3. VLM (Qwen2.5-VL) SEES the mask overlay and reports missing_objects / wrong_objects / too_dark.
  4. The loop re-prompts: add missing concepts, lower the detection threshold, protect wrong things,
     brighten harder if too dark — then retries until the VLM is satisfied (or max iters).

Usage:
  mask_agentic.py IMAGE --target "the biker and the camera gear" --backend sam3 --out mask.png
     [--query "seed. concepts."] [--roi ROI.png] [--depth-map D.png] [--mode precise|removal]
     [--carve] [--max-iter 3]
"""
from __future__ import annotations
import argparse, os, sys, subprocess, uuid, json, re, urllib.request
import numpy as np, cv2
sys.path.insert(0, os.path.expanduser("~/comfy"))
from pelib import vlm

PY = os.path.expanduser("~/comfy/ComfyUI/.venv/bin/python")
C2F = os.path.expanduser("~/comfy/mask_c2f.py")
LLM_URL = os.environ.get("LLM_URL", "http://127.0.0.1:8080/v1/chat/completions")
LLM_MODEL = os.environ.get("INTENT_LLM", "gemma4-12b-ablit")  # non-reasoning: clean list output (lfm2.5 dumps reasoning)

_STOP = {"the", "a", "an", "and", "or", "of", "with", "target", "concepts", "concept", "objects",
         "object", "list", "etc", "output", "reply", "photo", "image", "parts", "part", "here"}


def llm_expand(target):
    """LFM2.5 -> a period-separated list of concrete singular concept nouns covering the target+parts."""
    sysmsg = ("List every distinct physical object AND sub-part a segmentation model must select to fully "
              "cover this photo-editing target. Use common singular nouns an open-vocabulary detector knows "
              "(person, bicycle, wheel, helmet, camera, tripod, monitor, light stand, cable). Include parts "
              "that are easy to miss. Reply with ONLY a period-separated list of 3-8 nouns, no explanations. "
              "Example: 'the biker and camera rig' -> person. bicycle. helmet. camera. tripod. monitor.")
    body = {"model": LLM_MODEL, "temperature": 0, "max_tokens": 120,
            "messages": [{"role": "system", "content": sysmsg}, {"role": "user", "content": target}],
            "chat_template_kwargs": {"enable_thinking": False}}
    try:
        r = json.load(urllib.request.urlopen(urllib.request.Request(
            LLM_URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}), timeout=60))
        m = r["choices"][0]["message"]
        txt = (m.get("content") or "") + " " + (m.get("reasoning_content") or "")
    except Exception as e:
        print("[llm] expand failed:", str(e)[:60]); return _fallback(target)
    concepts = _extract(txt)
    return (". ".join(concepts) + ".") if concepts else _fallback(target)


def _extract(txt):
    cand = re.split(r"[.,\n;:•\-]", txt)
    out = []
    for c in cand:
        c = c.strip().lower()
        w = c.split()
        if 1 <= len(w) <= 3 and c.isascii() and any(ch.isalpha() for ch in c) and len(c) > 2:
            if c not in _STOP and not (len(w) == 1 and w[0] in _STOP):
                out.append(c)
    return list(dict.fromkeys(out))[:8]


def _fallback(target):
    ex = _extract(target)
    return (". ".join(ex) + ".") if ex else target.strip().rstrip(".") + "."


def shadow_lift(bgr, gamma, clahe=2.5):
    """Shadow lift: gamma curve on L (reveals crushed shadows) + CLAHE (local contrast). Denoise at strong
    lift so amplified 8-bit noise doesn't swamp the detail. gamma<1 lifts; smaller gamma = stronger."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    L = np.clip(255.0 * (L.astype(np.float32) / 255.0) ** gamma, 0, 255).astype(np.uint8)
    L = cv2.createCLAHE(clipLimit=clahe, tileGridSize=(8, 8)).apply(L)
    out = cv2.cvtColor(cv2.merge([L, A, B]), cv2.COLOR_LAB2BGR)
    return cv2.fastNlMeansDenoisingColored(out, None, 3, 3, 7, 15) if gamma <= 0.38 else out


def _crop_outline(img, roi_mask, pad_frac=0.15):
    """Crop to the ROI bbox (+context) and draw the selection outline in green. Returns (crop, bbox)."""
    ys, xs = np.where(roi_mask > 127)
    if not len(xs):
        return None, None
    H, W = img.shape[:2]
    pad = int(pad_frac * max(xs.max() - xs.min(), ys.max() - ys.min()) + 30)
    x0, x1 = max(0, xs.min() - pad), min(W, xs.max() + pad)
    y0, y1 = max(0, ys.min() - pad), min(H, ys.max() + pad)
    crop = img[y0:y1, x0:x1].copy()
    edge = cv2.morphologyEx((roi_mask[y0:y1, x0:x1] > 127).astype(np.uint8),
                            cv2.MORPH_GRADIENT, np.ones((7, 7), np.uint8))
    crop[edge > 0] = (0, 255, 0)
    return crop, (x0, y0, x1, y1)


def vlm_autolift(img, roi_mask, gammas=(0.75, 0.55, 0.40, 0.30), max_iter=4):
    """VLM-DRIVEN adaptive lift: escalate the shadow-lift and ask the VLM what it can see at each level.
    ACCUMULATE the objects across ALL levels (different brightnesses reveal different things — a dim
    tripod may only show at one lift), stop once the VLM is confident. Returns (chosen_gamma,
    lifted_full_image, unioned_objects) so the lift the VLM settled on is what SAM3 gets fed."""
    seen, best = [], (gammas[0], img)
    for g in gammas[:max_iter]:
        lifted = shadow_lift(img, g)
        crop, _ = _crop_outline(lifted, roi_mask)
        if crop is None:
            break
        q = ("A user selected the green-outlined region. List EVERY distinct physical object you can make "
             "out inside it (even faint ones), as concrete singular nouns; and say if it's clear enough. "
             'Reply ONE JSON object only: {"clear": true/false, "objects": "comma-separated nouns"}')
        v = vlm._json(vlm.ask(crop, q))
        objs = str(v.get("objects", "")).strip()
        for o in re.split(r"[,.;]", objs):
            o = o.strip().lower()
            if len(o) > 2 and o.isascii() and o not in seen and not (len(o.split()) == 1 and o in _STOP):
                seen.append(o)
        print(f"[autolift] gamma {g:.2f}: clear={v.get('clear')} objects='{objs}' | union={seen}")
        best = (g, lifted)
        if v.get("clear") is True and seen:
            break
    return best[0], best[1], ", ".join(seen)


def plan_concepts(vlm_desc, user_prompt):
    """LLM (gemma) fuses the VLM's view of the selection with the user's prompt (if any) into concrete
    SAM3 concepts. The user prompt STEERS (e.g. 'just the cameras' narrows; empty = the prominent objects)."""
    sysmsg = ("You decide what a segmentation model should select. A vision model listed the objects it "
              "sees inside the user's selection; the user may add an instruction that narrows or clarifies "
              "it. Output ONLY a period-separated list of concrete singular concept nouns to segment "
              "(and their visible parts). No explanations.")
    up = user_prompt.strip() if user_prompt else ""
    usermsg = (f"Vision model sees in the selection: {vlm_desc or '(unclear)'}\n"
               f"User instruction: {up or '(none — select the prominent selected objects)'}")
    body = {"model": LLM_MODEL, "temperature": 0, "max_tokens": 120,
            "messages": [{"role": "system", "content": sysmsg}, {"role": "user", "content": usermsg}],
            "chat_template_kwargs": {"enable_thinking": False}}
    try:
        r = json.load(urllib.request.urlopen(urllib.request.Request(
            LLM_URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}), timeout=60))
        m = r["choices"][0]["message"]
        txt = (m.get("content") or "") + " " + (m.get("reasoning_content") or "")
    except Exception as e:
        print("[llm] plan failed:", str(e)[:60])
        return _fallback(vlm_desc + " " + up)
    ex = _extract(txt)
    return (". ".join(ex) + ".") if ex else _fallback(vlm_desc + " " + up)


def run_c2f(image, query, roi, depth, mode, backend, out, sam3_thr, box_thr, text_thr, enh_gamma, protect, carve, no_enhance=False, multirep=False):
    cmd = [PY, C2F, image, "--query", query, "--mode", mode, "--backend", backend, "--out", out,
           "--sam3-threshold", str(sam3_thr), "--box-threshold", str(box_thr),
           "--text-threshold", str(text_thr), "--enh-gamma", str(enh_gamma), "--protect-query", protect]
    if no_enhance:                                            # image is already VLM-lifted -> don't double-lift
        cmd += ["--no-enhance"]
    if multirep and backend in ("sam3", "both"):             # SAM3 over multiple lifts + instance-aware combine
        cmd += ["--sam3-multirep", "--sam3-parallel"]
    if carve:
        cmd += ["--sam3-carve"]
    if roi:
        cmd += ["--roi", roi]
    if depth:
        cmd += ["--depth-map", depth]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=800)
    return (r.returncode == 0 and os.path.exists(out)), r.stdout


def score(v):
    return (int(v.get("fully_covered", False)) - int(v.get("missed_parts", False))
            - int(v.get("covers_wrong_things", False)) - int(v.get("too_dark_to_tell", False)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--target", default="", help="natural-language target; LLM expands it to concepts")
    ap.add_argument("--query", default="", help="seed concept query (skips LLM expansion)")
    ap.add_argument("--backend", choices=["sam2", "sam3", "both"], default="sam3")
    ap.add_argument("--roi", default=""); ap.add_argument("--depth-map", default="")
    ap.add_argument("--mode", default="precise"); ap.add_argument("--out", default="mask.png")
    ap.add_argument("--carve", action="store_true", default=False)
    ap.add_argument("--max-iter", type=int, default=3)
    ap.add_argument("--save-vis", default="")
    ap.add_argument("--describe", action="store_true", default=False,
                    help="with --roi: VLM looks at the selection, LLM fuses it with the prompt into concepts")
    ap.add_argument("--multirep", action="store_true", default=False,
                    help="SAM3 over multiple lifts + instance-aware combine (recall on dark/occluded objects)")
    a = ap.parse_args()

    img = cv2.imread(a.image, cv2.IMREAD_COLOR)
    target = a.target or a.query or "the selected objects"
    c2f_image, c2f_no_enh = a.image, False                           # image fed to SAM3 (may be VLM-lifted)
    if a.query:                                                      # explicit concepts win
        query = a.query
        print(f"[query] explicit: '{query}'")
    elif a.roi and (a.describe or not a.target):                     # SELECTION-DRIVEN: VLM autolift -> LLM plans
        roi_mask = cv2.imread(a.roi, cv2.IMREAD_GRAYSCALE)
        g, lifted, objs = vlm_autolift(img, roi_mask) if roi_mask is not None else (0, img, "")
        print(f"[autolift] chose gamma {g:.2f}; VLM sees (union): '{objs}'")
        # NOTE: autolift only gathers CONCEPTS. SAM3 keeps mask_c2f's validated enhance_for_detection on
        # the ORIGINAL image (my custom lift's denoise/CLAHE over-smooths figures -> patchier SAM3 masks).
        query = plan_concepts(objs, a.target)
        print(f"[llm] plan (+prompt '{a.target}'): '{query}'")
    else:                                                            # prompt-only: LLM expands the target
        query = llm_expand(target)
        print(f"[llm] target: '{target}' -> concepts: '{query}'")

    sam3_thr, box_thr, text_thr, enh_gamma = 0.40, 0.22, 0.18, 0.55
    protect_terms = set()
    best_mask, best_v, best_s = None, None, -99
    tmp = f"/tmp/ma_{uuid.uuid4().hex[:8]}.png"

    for it in range(a.max_iter):
        protect = ". ".join(sorted(protect_terms)) + ("." if protect_terms else "")
        print(f"\n=== iter {it+1}/{a.max_iter} | backend {a.backend} | sam3_thr {sam3_thr:.2f} "
              f"box {box_thr:.2f} gamma {enh_gamma:.2f} | query='{query}' protect='{protect}' ===")
        ok, _ = run_c2f(c2f_image, query, a.roi, a.depth_map, a.mode, a.backend, tmp,
                        sam3_thr, box_thr, text_thr, enh_gamma, protect, a.carve,
                        no_enhance=c2f_no_enh, multirep=a.multirep)
        if not ok:
            print("   mask_c2f failed"); continue
        mask = cv2.imread(tmp, cv2.IMREAD_GRAYSCALE)
        v = vlm.judge_mask(img, mask, target)
        s = score(v)
        print(f"   VLM: {v}")
        if s > best_s:
            best_s, best_v, best_mask = s, v, mask.copy()
        if v.get("fully_covered") and not v.get("missed_parts") \
                and not v.get("covers_wrong_things") and not v.get("too_dark_to_tell"):
            print("   accepted"); break
        # --- map the VLM verdict to concrete retries ---
        if v.get("too_dark_to_tell"):
            enh_gamma = max(0.30, enh_gamma - 0.10)                       # brighten harder
        miss = [str(x).strip() for x in v.get("missing_objects", []) if str(x).strip()]
        if miss:
            query = query.rstrip(". ") + ". " + ". ".join(miss) + "."     # feed missing concepts back
        if v.get("missed_parts") or not v.get("fully_covered"):
            sam3_thr = max(0.25, sam3_thr - 0.08)                         # SAM3: catch fainter instances
            box_thr = max(0.15, box_thr - 0.05); text_thr = max(0.12, text_thr - 0.04)  # SAM2 knobs
        for w in v.get("wrong_objects", []):
            if str(w).strip():
                protect_terms.add(str(w).strip())
        if v.get("covers_wrong_things"):
            sam3_thr = min(0.6, sam3_thr + 0.05)                          # tighten if grabbing junk

    if best_mask is None:
        sys.exit("no mask produced")
    cv2.imwrite(a.out, best_mask)
    if a.save_vis:
        vis = cv2.convertScaleAbs(img, alpha=2.2, beta=12)
        vis[best_mask > 127] = (0.4 * vis[best_mask > 127] + 0.6 * np.array([0, 0, 255])).astype(np.uint8)
        cv2.imwrite(a.save_vis, vis)
    try:
        os.remove(tmp)
    except Exception:
        pass
    print(f"\ndone [{a.backend}] -> {a.out} | best verdict {best_v}")


if __name__ == "__main__":
    main()
