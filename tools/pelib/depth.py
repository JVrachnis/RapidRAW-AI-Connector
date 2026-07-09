"""pelib.depth - monocular depth (near=bright uint8). Depth Anything V2 via ComfyUI, and Apple Depth Pro
via transformers (sharper thin-structure boundaries — spokes/mesh — for the see-through carve)."""
import os
import threading
import cv2
import numpy as np
from . import comfy

_DPRO = {}  # lazy Depth Pro cache (processor once + one model copy per device)
_DPRO_LOCK = threading.Lock()  # concurrent lazy loads race -> half-init model; guard cache population


def _load_dpro(device, model_id):
    from transformers import DepthProForDepthEstimation, DepthProImageProcessor
    import torch
    key = f"m::{device}"
    with _DPRO_LOCK:
        if "p" not in _DPRO:
            _DPRO["p"] = DepthProImageProcessor.from_pretrained(model_id)
        if key not in _DPRO:
            _DPRO[key] = (DepthProForDepthEstimation.from_pretrained(
                model_id, torch_dtype=torch.float16, attn_implementation="eager").to(device).eval())
    return _DPRO["p"], _DPRO[key]


def depth_map(source_bgr_or_path, ckpt="depth_anything_v2_vitl.pth", res=1024, timeout=180):
    """Depth Anything V2 depth map (uint8, near=bright) for an image (path or BGR array)."""
    img = cv2.imread(source_bgr_or_path) if isinstance(source_bgr_or_path, str) else source_bgr_or_path
    name = comfy.stage(img)
    wf = {
        "1": {"class_type": "LoadImage", "inputs": {"image": name}},
        "2": {"class_type": "DepthAnythingV2Preprocessor",
              "inputs": {"image": ["1", 0], "ckpt_name": ckpt, "resolution": res}},
        "3": {"class_type": "SaveImage", "inputs": {"images": ["2", 0], "filename_prefix": "pe_depth"}},
    }
    out = comfy.run(wf, timeout=timeout, cleanup_names=(name,))
    return cv2.cvtColor(out, cv2.COLOR_BGR2GRAY) if out is not None and out.ndim == 3 else out


def depth_pro(bgr, model_id=os.environ.get("DEPTHPRO_ID", "apple/DepthPro-hf")):
    """Apple Depth Pro depth map (uint8, near=bright disparity) for a BGR array. Sharper boundaries on
    thin structures (spokes/hair) than Depth Anything V2. fp16 + eager -> runs on the Pascal P100 (~3.5s).
    Returns the inverse-depth (disparity) normalised to 0..255 so near=bright, matching depth_map()."""
    import torch
    from PIL import Image
    if "m" not in _DPRO:
        from transformers import DepthProForDepthEstimation, DepthProImageProcessor
        _DPRO["p"] = DepthProImageProcessor.from_pretrained(model_id)
        _DPRO["m"] = (DepthProForDepthEstimation.from_pretrained(
            model_id, torch_dtype=torch.float16, attn_implementation="eager").to("cuda").eval())
    proc, model = _DPRO["p"], _DPRO["m"]
    H, W = bgr.shape[:2]
    pil = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    inp = proc(images=pil, return_tensors="pt").to("cuda", torch.float16)
    with torch.no_grad():
        out = model(**inp)
    dm = proc.post_process_depth_estimation(out, target_sizes=[(H, W)])[0]["predicted_depth"]
    dm = dm.float().cpu().numpy()                      # metric metres, near=small
    disp = 1.0 / np.clip(dm, 0.1, None)                # inverse depth -> near=bright
    return cv2.normalize(disp, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def depth_pro_metric(bgr, device="cuda:0", max_side=None, model_id=os.environ.get("DEPTHPRO_ID", "apple/DepthPro-hf")):
    """Like depth_pro but returns the UNNORMALISED metric inverse-depth (disparity, near=large) as float32.
    Because it's a consistent metric scale (not per-crop min-max), a global full-frame pass and per-tile
    zoomed passes are COMPARABLE -> the object/sky depth valley can be found globally and applied locally.
    `device` selects a per-GPU model copy so tiles can run in parallel across both P100s. `max_side`
    downscales the input (and the returned map) -> avoids post-processing depth back up to a 24MP frame for
    the coarse global pass (big time saver)."""
    import torch
    from PIL import Image
    if max_side and max(bgr.shape[:2]) > max_side:
        s = max_side / max(bgr.shape[:2])
        bgr = cv2.resize(bgr, (int(bgr.shape[1] * s), int(bgr.shape[0] * s)), interpolation=cv2.INTER_AREA)
    proc, model = _load_dpro(device, model_id)
    H, W = bgr.shape[:2]
    pil = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    inp = proc(images=pil, return_tensors="pt").to(device, torch.float16)
    with torch.no_grad():
        out = model(**inp)
    dm = proc.post_process_depth_estimation(out, target_sizes=[(H, W)])[0]["predicted_depth"]
    dm = dm.float().cpu().numpy()
    return (1.0 / np.clip(dm, 0.1, None)).astype(np.float32)     # disparity, near=large, metric scale


def depth_pro_metric_batch(bgr_list, device="cuda:0", model_id=os.environ.get("DEPTHPRO_ID", "apple/DepthPro-hf")):
    """One batched Depth Pro forward for several equal-cost tiles -> amortises launch/pre/post overhead vs
    N separate calls. Returns a list of metric-disparity float32 maps (each at its input size)."""
    import torch
    from PIL import Image
    proc, model = _load_dpro(device, model_id)
    sizes = [b.shape[:2] for b in bgr_list]
    pils = [Image.fromarray(cv2.cvtColor(b, cv2.COLOR_BGR2RGB)) for b in bgr_list]
    inp = proc(images=pils, return_tensors="pt").to(device, torch.float16)
    with torch.no_grad():
        out = model(**inp)
    res = proc.post_process_depth_estimation(out, target_sizes=sizes)
    return [(1.0 / np.clip(r["predicted_depth"].float().cpu().numpy(), 0.1, None)).astype(np.float32) for r in res]


def prewarm_dpro(devices=("cuda:0", "cuda:1"), model_id=os.environ.get("DEPTHPRO_ID", "apple/DepthPro-hf")):
    """Load a Depth Pro copy on each device in the MAIN thread (concurrent lazy load races -> dtype corruption)."""
    for d in devices:
        try:
            _load_dpro(d, model_id)
        except Exception:
            pass
