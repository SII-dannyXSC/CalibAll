"""Save pipeline results: masks, tracking point visualization, config JSON."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
from PIL import Image, ImageDraw


def save_masks(
    masks: list[np.ndarray],
    mask_frame_idxs: list[int],
    video: np.ndarray,
    camera_key: str,
    result_dir: Path,
    manual_label_dir: Path,
    filename_prefix: str,
    overlay_fn=None,
) -> list[str]:
    """Save mask arrays and overlay images. Returns list of mask save paths."""
    # overlay function
    if overlay_fn is None:
        from caliball.utils.image import overlay_mask
        overlay_fn = overlay_mask

    mask_output_dir = result_dir / "masks"
    mask_output_dir.mkdir(parents=True, exist_ok=True)
    manual_label_dir.mkdir(parents=True, exist_ok=True)

    mask_save_paths = []
    for mi, (mfi, msk) in enumerate(zip(mask_frame_idxs, masks)):
        if msk is None:
            continue
        # Save to result dir
        result_mask_path = mask_output_dir / f"{camera_key}_{mfi:06d}.npy"
        result_overlay_path = mask_output_dir / f"{camera_key}_{mfi:06d}_overlay.png"
        np.save(result_mask_path, msk.astype(np.uint8))
        Image.fromarray(overlay_fn(video[mfi], msk)).save(result_overlay_path)

        # Save to manual_label dir
        suffix = f".mask_{mfi}" if len(mask_frame_idxs) > 1 else ".mask"
        manual_mask_path = manual_label_dir / f"{filename_prefix}{suffix}.npy"
        mask_overlay_path = manual_label_dir / f"{filename_prefix}{suffix}_overlay.png"
        np.save(manual_mask_path, msk.astype(np.uint8))
        Image.fromarray(overlay_fn(video[mfi], msk)).save(mask_overlay_path)
        mask_save_paths.append(str(manual_mask_path))

    return mask_save_paths


def save_tracking_point_vis(
    frame: np.ndarray,
    tracking_point: tuple[float, float],
    save_path: Path,
):
    """Save tracking point visualization image."""
    if frame.max() <= 1.0:
        frame = (frame * 255).astype(np.uint8)
    vis_img = Image.fromarray(frame)
    draw = ImageDraw.Draw(vis_img)
    r = 7
    x, y = tracking_point
    draw.ellipse([x - r, y - r, x + r, y + r], outline=(255, 0, 0), width=3, fill=(255, 0, 0))
    vis_img.save(save_path)


def save_config(config: dict, save_path: Path):
    """Save pipeline config as JSON."""
    def _serialize(v: Any) -> Any:
        if isinstance(v, Path):
            return str(v)
        if isinstance(v, dict):
            return {k: _serialize(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [_serialize(x) for x in v]
        if isinstance(v, (str, int, float, bool)) or v is None:
            return v
        return str(v)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(_serialize(config), f, indent=2, ensure_ascii=False)
