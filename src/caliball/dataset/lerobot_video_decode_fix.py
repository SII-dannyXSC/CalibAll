"""
修复 torchvision VideoReader + pyav：frame['pts'] 可能为 fractions.Fraction，
直接 torch.tensor(loaded_ts) 会在 torch.cdist 等处触发 TypeError: Tensor / Fraction。

对 lerobot 的 decode_video_frames_torchvision 做 monkey-patch（与上游逻辑一致，仅统一为 float）。
"""

from __future__ import annotations

import logging

import lerobot.common.datasets.video_utils as vu
import torch
import torchvision


def _decode_video_frames_torchvision_fixed(
    video_path: Path | str,
    timestamps: list[float],
    tolerance_s: float,
    backend: str = "pyav",
    log_loaded_timestamps: bool = False,
) -> torch.Tensor:
    video_path = str(video_path)
    timestamps = [float(t) for t in timestamps]

    keyframes_only = False
    torchvision.set_video_backend(backend)
    if backend == "pyav":
        keyframes_only = True

    reader = torchvision.io.VideoReader(video_path, "video")

    first_ts = min(timestamps)
    last_ts = max(timestamps)
    reader.seek(first_ts, keyframes_only=keyframes_only)

    loaded_frames = []
    loaded_ts = []
    for frame in reader:
        current_ts = float(frame["pts"])
        if log_loaded_timestamps:
            logging.info(f"frame loaded at timestamp={current_ts:.4f}")
        loaded_frames.append(frame["data"])
        loaded_ts.append(current_ts)
        if current_ts >= last_ts:
            break

    if backend == "pyav":
        reader.container.close()

    reader = None

    query_ts = torch.as_tensor(timestamps, dtype=torch.float64)
    loaded_ts_t = torch.as_tensor(loaded_ts, dtype=torch.float64)

    dist = torch.cdist(query_ts[:, None], loaded_ts_t[:, None], p=1)
    min_, argmin_ = dist.min(1)

    is_within_tol = min_ < tolerance_s
    assert is_within_tol.all(), (
        f"One or several query timestamps unexpectedly violate the tolerance ({min_[~is_within_tol]} > {tolerance_s=})."
        "It means that the closest frame that can be loaded from the video is too far away in time."
        "This might be due to synchronization issues with timestamps during data collection."
        "To be safe, we advise to ignore this item during training."
        f"\nqueried timestamps: {query_ts}"
        f"\nloaded timestamps: {loaded_ts_t}"
        f"\nvideo: {video_path}"
        f"\nbackend: {backend}"
    )

    closest_frames = torch.stack([loaded_frames[idx] for idx in argmin_])

    closest_frames = closest_frames.type(torch.float32) / 255

    assert len(timestamps) == len(closest_frames)
    return closest_frames


def apply_lerobot_pyav_pts_fraction_fix() -> None:
    """在 import decode_video_frames 之前调用，替换 torchvision 解码实现。"""
    vu.decode_video_frames_torchvision = _decode_video_frames_torchvision_fixed
