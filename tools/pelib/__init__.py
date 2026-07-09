"""
pelib - shared building blocks for the photo edit/removal stack.

Modules:
  comfy    - talk to the ComfyUI API (stage inputs, submit a graph, read+clean output)
  imaging  - mask loading, /8 alignment, mask-context crop box, feather paste, long-side fit
  sdxl     - RealVisXL inpaint workflow builder
  matte    - ViTMatte trimap alpha refine (bbox-crop)
  enhance  - shadow-lift/CLAHE/unsharp for detection
  depth    - Depth Anything V2 depth map via ComfyUI

Import from anywhere that has ~/comfy on sys.path (the scripts already add it).
"""
from . import comfy, imaging  # noqa: F401
__all__ = ["comfy", "imaging", "sdxl", "matte", "enhance", "depth"]
