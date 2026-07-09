"""pelib.sdxl - RealVisXL Lightning inpaint workflow builder (ControlNet-free)."""
CKPT = "RealVisXL_V5.0_Lightning_fp16.safetensors"


def inpaint_wf(img_name, mask_name, prompt, neg, seed, steps=6, cfg=1.8,
               sampler="dpmpp_sde", scheduler="karras", denoise=1.0, prefix="pe_sdxl"):
    """SDXL inpaint graph: VAEEncode -> SetLatentNoiseMask -> KSampler -> VAEDecode -> SaveImage.
    img_name/mask_name are files already staged in ComfyUI/input (mask read from its red channel)."""
    return {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": CKPT}},
        "7": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 1], "text": prompt}},
        "8": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 1], "text": neg}},
        "10": {"class_type": "LoadImage", "inputs": {"image": img_name}},
        "11": {"class_type": "VAEEncode", "inputs": {"pixels": ["10", 0], "vae": ["1", 2]}},
        "12": {"class_type": "LoadImage", "inputs": {"image": mask_name}},
        "13": {"class_type": "ImageToMask", "inputs": {"image": ["12", 0], "channel": "red"}},
        "14": {"class_type": "SetLatentNoiseMask", "inputs": {"samples": ["11", 0], "mask": ["13", 0]}},
        "15": {"class_type": "KSampler",
               "inputs": {"model": ["1", 0], "positive": ["7", 0], "negative": ["8", 0],
                          "latent_image": ["14", 0], "seed": seed, "steps": steps, "cfg": cfg,
                          "sampler_name": sampler, "scheduler": scheduler, "denoise": denoise}},
        "16": {"class_type": "VAEDecode", "inputs": {"samples": ["15", 0], "vae": ["1", 2]}},
        "17": {"class_type": "SaveImage", "inputs": {"images": ["16", 0], "filename_prefix": prefix}},
    }
