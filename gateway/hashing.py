import os
import blake3

RAW_EXTS = {".arw", ".cr2", ".cr3", ".nef", ".dng", ".raf", ".orf", ".rw2", ".pef", ".srw"}
TIFF_EXTS = {".tif", ".tiff"}

def content_id(data: bytes) -> str:
    return blake3.blake3(data).hexdigest()

def detect_kind(filename: str) -> str:
    ext = os.path.splitext(filename)[1].lower()
    if ext in RAW_EXTS:
        return "raw"
    if ext in TIFF_EXTS:
        return "tiff"
    return "std"
