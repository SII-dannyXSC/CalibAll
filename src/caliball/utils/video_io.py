"""视频读写工具。"""

from __future__ import annotations

import subprocess

import torch
import torchvision
import numpy as np


def get_ffmpeg_exe() -> str:
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return "ffmpeg"


_FFMPEG = get_ffmpeg_exe()


class FfmpegVideoReader:
    """基于 PyAV 的视频帧读取器。"""

    def __init__(self, path: str):
        import av as _av
        self._container = _av.open(path)
        self._stream = self._container.streams.video[0]
        self.width = self._stream.width
        self.height = self._stream.height
        self.n_frames = self._stream.frames if self._stream.frames else -1
        self._iter = self._container.decode(self._stream)

    def read(self):
        try:
            frame = next(self._iter)
            return True, frame.to_ndarray(format="bgr24")
        except StopIteration:
            return False, None

    def release(self):
        self._container.close()


class FfmpegVideoWriter:
    """基于 FFmpeg 子进程的视频写入器（libx264）。"""

    def __init__(self, path: str, fps: int, width: int, height: int):
        self._proc = subprocess.Popen(
            [_FFMPEG, "-y",
             "-f", "rawvideo", "-vcodec", "rawvideo",
             "-pix_fmt", "bgr24", "-s", f"{width}x{height}",
             "-r", str(fps), "-i", "pipe:0",
             "-vcodec", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
             path],
            stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )

    def write(self, frame):
        self._proc.stdin.write(frame.tobytes())

    def release(self):
        if self._proc.stdin:
            self._proc.stdin.close()
        self._proc.wait()


def decode_video_frames(
    video_path: str,
    timestamps: list,
    tolerance_s: float = 0.04,
    backend: str = "pyav",
) -> np.ndarray:
    """解码指定时间戳的视频帧，返回 (T, H, W, 3) uint8 numpy 数组。"""
    torchvision.set_video_backend(backend)
    reader = torchvision.io.VideoReader(str(video_path), "video")

    timestamps = [float(t) for t in timestamps]
    first_ts, last_ts = min(timestamps), max(timestamps)
    reader.seek(first_ts, keyframes_only=(backend == "pyav"))

    loaded_frames, loaded_ts = [], []
    for frame in reader:
        current_ts = float(frame["pts"])
        loaded_frames.append(frame["data"])
        loaded_ts.append(current_ts)
        if current_ts >= last_ts:
            break

    if backend == "pyav":
        try:
            reader.container.close()
        except Exception:
            pass

    query_ts = torch.as_tensor(timestamps, dtype=torch.float64)
    loaded_ts_t = torch.as_tensor(loaded_ts, dtype=torch.float64)
    dist = torch.cdist(query_ts[:, None], loaded_ts_t[:, None], p=1)
    min_, argmin_ = dist.min(1)

    assert (min_ < tolerance_s).all(), (
        f"Timestamps violate tolerance ({min_[min_ >= tolerance_s]} > {tolerance_s}). "
        f"video: {video_path}"
    )

    frames = torch.stack([loaded_frames[idx] for idx in argmin_])
    # (T, C, H, W) float → (T, H, W, C) uint8
    arr = frames.permute(0, 2, 3, 1).numpy()
    if arr.dtype != np.uint8:
        arr = (arr.astype(np.float32) * 255).clip(0, 255).astype(np.uint8)
    return arr
