"""pelib.comfy - minimal ComfyUI API client (stage input, submit graph, read+clean output)."""
import os, json, time, uuid, urllib.request
import cv2

COMFY = os.environ.get("COMFY_URL", "http://127.0.0.1:8188")
CIN = os.path.expanduser("~/comfy/ComfyUI/input")
COUT = os.path.expanduser("~/comfy/ComfyUI/output")


def stage(img_bgr, ext=".png"):
    """Write an image into ComfyUI/input, return its filename (for LoadImage)."""
    name = f"pe_{uuid.uuid4().hex[:10]}{ext}"
    cv2.imwrite(os.path.join(CIN, name), img_bgr)
    return name


def submit(workflow, timeout=1200, poll=2.0):
    """Queue a prompt graph, poll /history, return the last output image filename. Raises on error/timeout."""
    pid = json.load(urllib.request.urlopen(urllib.request.Request(
        COMFY + "/prompt", data=json.dumps({"prompt": workflow}).encode(),
        headers={"Content-Type": "application/json"}), timeout=30))["prompt_id"]
    for _ in range(max(1, int(timeout / poll))):
        time.sleep(poll)
        try:
            h = json.load(urllib.request.urlopen(COMFY + "/history/" + pid, timeout=12))
        except Exception:
            continue
        if h:
            hv = list(h.values())[0]
            st = hv.get("status", {})
            if st.get("status_str") == "error":
                raise RuntimeError("ComfyUI error: " + json.dumps(st)[:400])
            out = None
            for v in hv.get("outputs", {}).values():
                if "images" in v:
                    out = v["images"][-1]["filename"]
            if out:
                return out
    raise RuntimeError("ComfyUI timed out")


def read_output(name, cleanup=True):
    """Load a SaveImage result from ComfyUI/output; delete it afterwards so output/ doesn't fill up."""
    img = cv2.imread(os.path.join(COUT, name), cv2.IMREAD_COLOR)
    if cleanup:
        _rm(os.path.join(COUT, name))
    return img


def cleanup_inputs(*names):
    for n in names:
        _rm(os.path.join(CIN, n))


def run(workflow, timeout=1200, cleanup_names=()):
    """Submit -> read result BGR -> clean the staged inputs + the output file."""
    out = submit(workflow, timeout=timeout)
    img = read_output(out, cleanup=True)
    cleanup_inputs(*cleanup_names)
    return img


def _rm(path):
    try:
        os.remove(path)
    except Exception:
        pass
