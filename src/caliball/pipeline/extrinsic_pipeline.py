"""Extrinsic calibration pipeline: tracking -> coarse -> refinement -> visualization."""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np


class ExtrinsicPipeline:
    """Encapsulates the full extrinsic calibration pipeline.

    Args:
        coarse_init: CoarseInit model instance
        refinement: Refinement model instance
        device: torch device string
        max_steps: maximum refinement iterations
    """

    def __init__(self, coarse_init, refinement, device: str = "cuda", max_steps: int = 10000):
        self._coarse = coarse_init
        self._refinement = refinement
        self._device = device
        self._max_steps = max_steps

    def update_robot(self, robot, mesh_paths=None):
        """Update FK models when robot type changes.

        Args:
            robot: new robot instance (from ``build_robot``)
            mesh_paths: mesh file paths for the new robot (defaults to ``robot.MESH_PATHS``)
        """
        if mesh_paths is None:
            mesh_paths = robot.MESH_PATHS
        self._coarse.update_robot(robot)
        self._refinement.update_robot(robot, mesh_paths)

    def reset_intrinsic(self):
        """Reset cached intrinsic (for new camera/resolution)."""
        self._coarse.reset_intrinsic()

    def run(
        self,
        clip: np.ndarray,
        joint_angles: np.ndarray,
        tracking_point: tuple[float, float],
        masks: list[np.ndarray],
        mask_ids: list[int],
        save_dir: str | Path,
        progress_fn: Optional[Callable] = None,
        stop_check: Optional[Callable] = None,
        arm_index: int = 0,
        full_video: Optional[np.ndarray] = None,
        full_joint_angles: Optional[np.ndarray] = None,
        tracking_img_idx: int = 0,
    ) -> dict:
        """Run the full pipeline.

        Args:
            clip: video clip for tracking/coarse/refine
            joint_angles: joint angles for clip
            full_video: complete video for visualization (if None, uses clip)
            full_joint_angles: complete joint angles for visualization (if None, uses joint_angles)

        Returns dict with: extrinsic_coarse, extrinsic_refined, intrinsic,
                          anno_video_path, loss_dict
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        def _report(stage, message, **kw):
            if progress_fn:
                progress_fn(stage=stage, message=message, **kw)

        # Stage 1: Tracking + Coarse (只用 tracking 范围的帧)
        _report("tracking", "正在追踪…")
        # tracking_img_idx 是 tracking point 帧在 clip 中的偏移
        # tracking 只用 clip 中对应 start:end 范围的帧
        track_clip = clip[tracking_img_idx:]  # 从 tracking point 帧开始到 clip 末尾
        track_joints = joint_angles[tracking_img_idx:]

        def _coarse_progress(msg):
            _report("tracking", msg)

        extrinsic, K, details = self._coarse.get_extrinsic(
            video=track_clip, joint_angles=track_joints,
            tracking_point=list(tracking_point), img_idx=0,
            save_path=str(save_dir), return_details=True,
            arm_index=arm_index, progress_fn=_coarse_progress,
        )

        # Tracking visualization
        pts_2d = details["points_2d"]
        pts_3d = details["points_3d"]
        vis = clip[0].copy()
        for pt in pts_2d:
            cv2.circle(vis, (int(pt[0]), int(pt[1])), 3, (0, 255, 0), -1)

        # 检查 3D 轨迹质量
        tracking_video = save_dir / "tracking.mp4"
        _report("coarse", "Coarse extrinsic estimated",
                image=vis,
                tracking_video_path=str(tracking_video) if tracking_video.exists() else None)

        # Free CoarseInit GPU memory (only drop local ref, like the original pipeline_fn)
        import torch
        gc.collect()
        torch.cuda.empty_cache()

        # Stage 2: Refinement
        _report("refine", "开始 Refinement…", step=0, max_steps=self._max_steps)

        def _refine_cb(step, max_steps, loss, overlay, overlays=None, mask_ids=None, **kwargs):
            _report(
                "refine",
                f"step {step}/{max_steps}  loss={loss:.6f}",
                step=step, max_steps=max_steps,
                image=overlay, overlays=overlays, mask_ids=mask_ids,
            )

        output, loss_dict = self._refinement.refine(
            clip, joint_angles, K, extrinsic, str(save_dir),
            mask=masks, mask_id=mask_ids, max_steps=self._max_steps,
            progress_callback=_refine_cb,
            stop_check=stop_check or (lambda: False),
        )

        refined_tsfm = output["tsfm"].detach().cpu().numpy()

        # Save matrices
        np.save(str(save_dir / "extrinsic_coarse.npy"), extrinsic)
        np.save(str(save_dir / "extrinsic_refined.npy"), refined_tsfm)
        np.save(str(save_dir / "intrinsic.npy"), K)

        # Stage 3: Visualization video (use full video if provided)
        _report("visualize", "生成可视化视频…")
        vis_video = full_video if full_video is not None else clip
        vis_joints = full_joint_angles if full_joint_angles is not None else joint_angles
        anno_path = self._render_video(
            vis_video, vis_joints, refined_tsfm, K, save_dir, _report,
        )

        tracking_video = save_dir / "tracking.mp4"

        return {
            "extrinsic_coarse": extrinsic,
            "extrinsic_refined": refined_tsfm,
            "intrinsic": K,
            "anno_video_path": str(anno_path),
            "tracking_video_path": str(tracking_video) if tracking_video.exists() else None,
            "loss_dict": loss_dict,
            "save_dir": str(save_dir),
        }

    def _render_video(self, clip, joint_angles, tsfm, K, save_dir, report_fn) -> Path:
        """Render annotation overlay video."""
        import torch as _torch
        from caliball.algorithms.rendering_optimizer import RBSolver
        from caliball.utils.image import add_mask as _add_mask

        H, W = clip.shape[1:3]
        vis_solver = RBSolver(self._refinement.mesh_paths, H, W, tsfm, device=self._device)
        vis_solver.to(self._device)
        vis_solver.eval()

        K_t = self._refinement._to_float_tensor(K).unsqueeze(0)

        anno_path = save_dir / "anno_video.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        vw = cv2.VideoWriter(str(anno_path), fourcc, 15, (W, H))

        last_overlay_rgb = None
        for fi in range(len(clip)):
            link_poses_i = self._refinement._prepare_link_poses(joint_angles[fi:fi + 1])
            dps = {
                "global_step": 0,
                "mask": _torch.zeros((1, H, W), device=self._device),
                "link_poses": link_poses_i,
                "K": K_t,
            }
            with _torch.no_grad():
                out_v, _ = vis_solver.forward(dps)
            rmask = out_v["rendered_masks"][0].detach().cpu().numpy()
            rmask = (rmask > 0).astype(np.uint8)
            frame_bgr = clip[fi][:, :, ::-1].copy()
            overlay_bgr = _add_mask(frame_bgr, rmask, color=[0, 255, 0], alpha=0.5)
            vw.write(overlay_bgr)
            last_overlay_rgb = overlay_bgr[:, :, ::-1]
            if fi % 5 == 0:
                report_fn("visualize", f"可视化 {fi + 1}/{len(clip)}", image=last_overlay_rgb)

        vw.release()

        # Transcode to H.264
        h264_path = save_dir / "anno_video_h264.mp4"
        try:
            import subprocess
            try:
                import imageio_ffmpeg
                ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
            except ImportError:
                ffmpeg_exe = "ffmpeg"
            subprocess.run(
                [ffmpeg_exe, "-y", "-i", str(anno_path), "-c:v", "libx264",
                 "-preset", "fast", "-crf", "23", "-pix_fmt", "yuv420p", str(h264_path)],
                check=True, capture_output=True,
            )
            anno_path = h264_path
        except Exception:
            pass

        return anno_path
