import base64
import time
import aiofiles
from engine import ComfyClient, ImageProcessor, build_workflow, save_inputs_for_debug
from gateway.registry import Capability, register

PARAMS_SCHEMA = {
    "type": "object",
    "required": ["prompt", "mask_image_base64"],
    "properties": {
        "prompt": {"type": "string"},
        "negative_prompt": {"type": "string",
                            "default": "blur, low quality, distortion, watermark"},
        "mask_image_base64": {"type": "string"},
        "seed": {"type": "integer"},
    },
}


class ComfyDown(Exception):
    kind = "comfyui_down"


async def handle(ctx) -> dict:
    if ctx.source is None:
        raise FileNotFoundError("source not available")
    mask_bytes = base64.b64decode(ctx.params["mask_image_base64"])
    processed = ImageProcessor.process_mask_for_comfyui(mask_bytes)

    await save_inputs_for_debug(ctx.source.path, processed)

    mask_path = ctx.workdir / "mask.png"
    async with aiofiles.open(mask_path, "wb") as f:
        await f.write(processed)
    seed = ctx.params.get("seed") or int(time.time())
    workflow = build_workflow(
        str(ctx.source.path.absolute()), str(mask_path.absolute()),
        ctx.params["prompt"],
        ctx.params.get("negative_prompt", "blur, low quality, distortion, watermark"),
        seed)
    try:
        result_bytes = await ComfyClient().execute(workflow)
    except ConnectionError as e:
        raise ComfyDown(str(e)) from e
    return ImageProcessor.crop_and_pack(result_bytes, mask_bytes)


register(Capability(id="inpaint", title="Generative inpaint (ComfyUI)",
                    params_schema=PARAMS_SCHEMA, handler=handle))
