"""Image-to-data-URL conversion utilities.

Extracted from ``caliball.utils.web_interaction`` for reuse across web
service modules.
"""

from __future__ import annotations

import base64
import io

import numpy as np
from PIL import Image


def image_to_data_url(image_rgb: np.ndarray) -> str:
    arr = np.asarray(image_rgb)
    if arr.dtype != np.uint8:
        arr = (arr * 255).astype(np.uint8) if arr.max() <= 1.0 else arr.astype(np.uint8)
    if arr.ndim == 2:
        pil = Image.fromarray(arr, mode="L").convert("RGB")
    else:
        pil = Image.fromarray(arr)
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    b64 = base64.standard_b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def image_to_thumb_url(image_rgb: np.ndarray, max_w: int = 120) -> str:
    """生成紧凑 JPEG 缩略图 data URL。"""
    arr = np.asarray(image_rgb)
    if arr.dtype != np.uint8:
        arr = (arr * 255).astype(np.uint8) if arr.max() <= 1.0 else arr.astype(np.uint8)
    if arr.ndim == 2:
        pil = Image.fromarray(arr, mode="L").convert("RGB")
    else:
        pil = Image.fromarray(arr)
    w, h = pil.size
    if w > max_w:
        pil = pil.resize((max_w, int(h * max_w / w)), Image.LANCZOS)
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=55)
    b64 = base64.standard_b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"
