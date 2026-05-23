"""Coarse extrinsic initialisation via feature matching + PnP.

Dependency-injection version: all heavy models (recognizer, tracker, etc.)
are constructed externally and passed in.
"""

from PIL import Image
import cv2
import numpy as np
import os

from caliball.algorithms.intrinsic_estimator import build_intrinsic_estimator


class CoarseInit:
    def __init__(self, robot, recognizer, tracker, pnp_solver, intrinsic_estimator=None):
        self.robot_tf = robot
        self.recognizer = recognizer
        self.point_tracker = tracker
        self.pnp_solver = pnp_solver
        self.intrinsic_estimator = intrinsic_estimator
        self._intrinsic = None

    def update_robot(self, robot):
        """Switch the FK model (does not reload DINOv2/CoTracker)."""
        self.robot_tf = robot

    def reset_intrinsic(self):
        """Reset the cached intrinsic so it is re-estimated on next call."""
        self._intrinsic = None

    def to(self, device):
        self.recognizer.to(device)
        self.point_tracker.to(device)
        if self.intrinsic_estimator is not None:
            self.intrinsic_estimator.to(device)

    def _init_recognizer(self, given_img_pil, given_p):
        self.recognizer.reset(img_pil=given_img_pil, p=given_p)

    def _init_intrinsic(self, intrinsic=None):
        self._intrinsic = intrinsic
        if intrinsic is None and self.intrinsic_estimator is None:
            self.intrinsic_estimator = build_intrinsic_estimator()

    def _get_intrinsic(self, img_pil):
        if self._intrinsic is None:
            if self.intrinsic_estimator is None:
                self.intrinsic_estimator = build_intrinsic_estimator()
            self._intrinsic, origin_width, origin_height = self.intrinsic_estimator.estimate(img_pil=img_pil)
            width, height = img_pil.size

            self._intrinsic[0, :3] *= 1.0 * width / origin_width
            self._intrinsic[1, :3] *= 1.0 * height / origin_height

        return self._intrinsic

    def get_extrinsic(self, video, joint_angles, tracking_point=None, img_idx=0,
                      method=cv2.SOLVEPNP_EPNP, save_path=None, init_w2c=None,
                      return_details=False, arm_index=0):
        img_pil = Image.fromarray(video[img_idx])

        if tracking_point is not None:
            u, v = tracking_point
        else:
            u, v = self.recognizer.get_uv(target_img_pil=img_pil)
        points_2d, pred_tracks, pred_visibility = self.point_tracker.track(video=video, uv=(u, v), img_idx=img_idx)

        if save_path is not None:
            self.point_tracker.visualize(video, pred_tracks=pred_tracks, pred_visibility=pred_visibility, path=os.path.join(save_path, "tracking"))

        K = self._get_intrinsic(img_pil)
        hom = np.array([self.robot_tf.fkine_flange(q)[0] for q in joint_angles])  # (T, 4, 4) or (T, 2, 4, 4)
        if hom.ndim == 4:
            hom = hom[:, arm_index, ...]
        points_3d = hom[:, :3, 3]
        extrinsic = self.pnp_solver(
            points_3d=points_3d, points_2d=points_2d,
            camera_matrix=K, method=method, init_w2c=init_w2c,
        )
        if return_details:
            return extrinsic, K, {"points_2d": points_2d, "points_3d": points_3d}
        return extrinsic, K
