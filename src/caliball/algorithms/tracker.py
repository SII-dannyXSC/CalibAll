"""Point tracker — CoTracker wrapper for 2-D keypoint tracking."""

import logging
import os
import subprocess
from pathlib import Path

import cv2
import numpy as np
import torch

from cotracker.predictor import CoTrackerPredictor
from cotracker.utils.visualizer import Visualizer

logger = logging.getLogger(__name__)


class Tracker:
    def __init__(self, repo_dir="facebookresearch/co-tracker", model_id="cotracker3_offline", local_ckpt_path=None, device=None) -> None:
        self.device = device

        if local_ckpt_path is not None:
            self.cotracker = CoTrackerPredictor(checkpoint=local_ckpt_path, window_len=60, v2=False)
        else:
            self.cotracker = torch.hub.load(repo_dir, model_id)
        self.cotracker.eval()

        if device is not None:
            self.cotracker.to(device)

    def to(self, device):
        self.device = device
        self.cotracker.to(device)
        return self

    def preprocess_video(self, video):
        video = torch.tensor(video).to(device=self.device, dtype=torch.float32)
        video = video.permute(0, 3, 1, 2)  # T C H W
        video = video.unsqueeze(0)          # B T C H W
        return video

    def track(self, video, uv, img_idx=0):
        video = self.preprocess_video(video)

        u, v = uv
        queries = [[img_idx, u, v]]
        queries = torch.tensor(queries).to(self.device, dtype=torch.float32)

        pred_tracks, pred_visibility = self.cotracker(video, queries=queries[None])  # B T N 2,  B T N 1

        points_2d = pred_tracks[0].permute(1, 0, 2)
        points_2d = points_2d.detach().cpu().numpy()
        points_2d = points_2d[0]    # only one query here

        return points_2d, pred_tracks, pred_visibility

    def visualize(self, video, pred_tracks, pred_visibility, path, pad_value=100):
        """Save tracking overlay video.

        CoTracker's default imageio+ffmpeg writer often hits Broken pipe when
        frame count is small or dimensions need padding. Use OpenCV instead.
        """
        out_path = path if str(path).endswith(".mp4") else f"{path}.mp4"
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

        try:
            video_t = self.preprocess_video(video)
            vis = Visualizer(save_dir=".", pad_value=pad_value, fps=10)
            res = vis.visualize(
                video=video_t,
                tracks=pred_tracks,
                visibility=pred_visibility,
                filename="tracking",
                save_video=False,
            )
            frames = res[0].permute(0, 2, 3, 1).contiguous().cpu().numpy()
            if frames.shape[0] < 1:
                logger.warning("No frames for tracking video")
                return
            self._write_mp4_cv2(frames, out_path, fps=vis.fps)
            self._try_h264_transcode(out_path)
            logger.info("Tracking video saved to %s", out_path)
        except Exception as exc:
            logger.warning("Tracking visualization failed (%s), using fallback", exc)
            self._write_tracking_fallback(video, pred_tracks, out_path, pad_value)

    @staticmethod
    def _write_mp4_cv2(frames_rgb: np.ndarray, out_path: str, fps: float = 10.0) -> None:
        """Write RGB frames with OpenCV (even dimensions for codec compatibility)."""
        frames_rgb = np.ascontiguousarray(frames_rgb, dtype=np.uint8)
        t, h, w = frames_rgb.shape[:3]
        w2, h2 = w - (w % 2), h - (h % 2)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, float(fps), (w2, h2))
        if not writer.isOpened():
            raise RuntimeError(f"cv2.VideoWriter failed to open {out_path}")
        try:
            for i in range(t):
                frame = frames_rgb[i, :h2, :w2]
                if frame.shape[2] == 3:
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                writer.write(frame)
        finally:
            writer.release()

    @staticmethod
    def _try_h264_transcode(path: str) -> None:
        """Optional H.264 transcode for browser playback; keep mp4v on failure."""
        from caliball.utils.video_io import get_ffmpeg_exe

        src = Path(path)
        tmp = src.with_name(src.stem + "_h264.mp4")
        try:
            subprocess.run(
                [
                    get_ffmpeg_exe(), "-y", "-i", str(src),
                    "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                    "-pix_fmt", "yuv420p", str(tmp),
                ],
                check=True,
                capture_output=True,
                timeout=180,
            )
            tmp.replace(src)
        except Exception:
            if tmp.exists():
                tmp.unlink()

    def _write_tracking_fallback(
        self, video, pred_tracks, out_path: str, pad_value: int,
    ) -> None:
        """Minimal overlay when CoTracker Visualizer fails."""
        frames = np.asarray(video)
        if frames.ndim != 4 or frames.shape[0] < 1:
            return
        tracks = pred_tracks[0].detach().cpu().numpy()
        pad = int(pad_value)
        t, h, w = frames.shape[:3]
        w2, h2 = w - (w % 2), h - (h % 2)
        out_w, out_h = w2 + 2 * pad, h2 + 2 * pad
        writer = cv2.VideoWriter(
            out_path, cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (out_w, out_h),
        )
        if not writer.isOpened():
            return
        try:
            for i in range(t):
                frame = np.ascontiguousarray(frames[i, :h2, :w2], dtype=np.uint8)
                if pad:
                    frame = cv2.copyMakeBorder(
                        frame, pad, pad, pad, pad,
                        cv2.BORDER_CONSTANT, value=(255, 255, 255),
                    )
                u = int(tracks[i, 0, 0] + pad)
                v = int(tracks[i, 0, 1] + pad)
                cv2.circle(frame, (u, v), 5, (0, 255, 0), -1)
                if frame.shape[2] == 3:
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                writer.write(frame)
        finally:
            writer.release()


def build_tracker(config, device=None):
    tracker = Tracker(repo_dir=config.tracker_repo_dir, model_id=config.tracker_id, local_ckpt_path=config.tracker_ckpt_path, device=device)
    return tracker
