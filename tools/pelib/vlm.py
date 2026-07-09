"""pelib.vlm - Qwen2.5-VL vision judge via llama-swap. Lets the pipeline SEE its own output and self-correct."""
import os, json, base64, re
import numpy as np, cv2
import urllib.request

VLM_URL = os.environ.get("VLM_URL", "http://127.0.0.1:8080/v1/chat/completions")
VLM_MODEL = os.environ.get("VLM_MODEL", "qwen2.5-vl")


def _b64(bgr, maxside=768):
    f = min(1.0, maxside / max(bgr.shape[:2]))
    if f < 1:
        bgr = cv2.resize(bgr, (0, 0), fx=f, fy=f, interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf).decode()


def ask(bgr, prompt, maxside=768, max_tokens=220, timeout=200):
    """Ask the VLM about an image (BGR). Returns the text reply, or '' on failure."""
    body = {"model": VLM_MODEL, "temperature": 0, "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + _b64(bgr, maxside)}}]}]}
    try:
        r = json.load(urllib.request.urlopen(urllib.request.Request(
            VLM_URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}), timeout=timeout))
        m = r["choices"][0]["message"]
        return (m.get("content") or "") + "\n" + (m.get("reasoning_content") or "")
    except Exception as e:
        return ""


def _json(text):
    for m in re.findall(r"\{[^{}]*\}", text, re.S):
        try:
            return json.loads(m)
        except Exception:
            pass
    return {}


def judge_mask(image_bgr, mask, targets):
    """VLM verdict on a mask overlay. Returns dict with fully_covered/missed_parts/covers_wrong/too_dark/notes.
    Crops to the mask region (+context) and brightens so the VLM can actually see it on dark shots."""
    H, W = image_bgr.shape[:2]
    ys, xs = np.where(mask > 127)
    if len(xs):
        pad = int(0.35 * max(xs.max() - xs.min(), ys.max() - ys.min()) + 40)
        x0, x1 = max(0, xs.min() - pad), min(W, xs.max() + pad)
        y0, y1 = max(0, ys.min() - pad), min(H, ys.max() + pad)
        image_bgr = image_bgr[y0:y1, x0:x1]; mask = mask[y0:y1, x0:x1]
    # auto-brighten to the region's own exposure so dark subjects are visible
    med = max(1.0, float(np.median(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY))))
    vis = cv2.convertScaleAbs(image_bgr, alpha=float(np.clip(120.0 / med, 1.2, 4.0)), beta=8)
    red = vis.copy(); red[mask > 127] = (0.4 * red[mask > 127] + 0.6 * np.array([0, 0, 255])).astype(np.uint8)
    q = (f"The red overlay is a selection mask that should cover exactly: {targets}. "
         "Judge it. Reply with ONE JSON object only: "
         '{"fully_covered":bool, "missed_parts":bool, "covers_wrong_things":bool, '
         '"too_dark_to_tell":bool, "missing_objects":["concrete detector words for anything NOT yet covered"], '
         '"wrong_objects":["things wrongly covered"], "notes":"one short sentence"}')
    return _json(ask(red, q))


def judge_fill(result_bgr, mask, targets):
    """VLM verdict on a removal/fill. Reply: looks_natural/has_ghosts/discoloured/seam_visible/notes."""
    q = (f"This photo had '{targets}' removed and the area filled in. Judge the filled area. "
         "Reply with ONE JSON object only: "
         '{"looks_natural":bool, "has_ghosts":bool, "discoloured":bool, "seam_visible":bool, "notes":"one short sentence"}')
    return _json(ask(result_bgr, q))
