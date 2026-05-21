"""Frame I/O and visualization utilities.

Extracted from ``scripts/extrinsic_detection.py`` for reuse across the
codebase.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Tuple

import cv2
import numpy as np
from PIL import Image


def ensure_dir(path: Path | str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_video_frames(
    video: np.ndarray,
    output_dir: Path,
    start_idx: int = 0,
    end_idx: Optional[int] = None,
) -> Path:
    output_dir = ensure_dir(output_dir)
    if end_idx is None:
        end_idx = len(video) - 1
    for idx in range(start_idx, end_idx + 1):
        Image.fromarray(video[idx]).save(output_dir / f"frame_{idx:06d}.png")
    print(f"保存帧完成: {output_dir} [{start_idx}, {end_idx}]")
    return output_dir


def exported_frames_complete(output_dir: Path, expected_count: int) -> bool:
    """导出目录是否已有与视频等长的非空 PNG（frame_000000.png …）。"""
    if expected_count <= 0:
        return False
    for idx in range(expected_count):
        p = output_dir / f"frame_{idx:06d}.png"
        if not p.is_file() or p.stat().st_size == 0:
            return False
    return True


def verify_exported_frames(output_dir: Path, expected_count: int) -> None:
    """确认已写入与视频长度一致的 PNG，否则中止。"""
    if expected_count <= 0:
        raise RuntimeError(f"视频长度为 0，无法校验导出帧: {output_dir}")
    missing = []
    for idx in range(expected_count):
        p = output_dir / f"frame_{idx:06d}.png"
        if not p.is_file() or p.stat().st_size == 0:
            missing.append(str(p))
    if missing:
        raise RuntimeError(
            f"帧导出不完整（期望 {expected_count} 张），缺 {len(missing)} 个，例如: {missing[:3]}"
        )


def overlay_mask(image_rgb: np.ndarray, mask: np.ndarray, color: Tuple[int, int, int] = (0, 255, 0), alpha: float = 0.45) -> np.ndarray:
    canvas = np.asarray(image_rgb).copy()
    overlay = canvas.copy()
    m = np.asarray(mask) > 0
    overlay[m] = color
    return cv2.addWeighted(overlay, alpha, canvas, 1 - alpha, 0)


def json_serialize(v: Any) -> Any:
    if isinstance(v, Path):
        return str(v)
    if isinstance(v, dict):
        return {k: json_serialize(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [json_serialize(x) for x in v]
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    return str(v)
