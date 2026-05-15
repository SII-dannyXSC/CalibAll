"""视频读写工具，基于 PyAV (读) 和 FFmpeg 子进程 (写)。"""

from __future__ import annotations

import subprocess


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
