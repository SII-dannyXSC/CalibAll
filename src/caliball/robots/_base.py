"""BaseTF -- abstract base class for all robot TF implementations.

Adds shared static helpers (``build_mesh_adjust``, ``build_joint_idx_map``)
that were previously duplicated across 5+ files.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


class BaseTF(ABC):
    """Robot TF base class.  Single-frame input q: (n_joints,).

    - Single arm: n_arms=1, output (1, 4, 4)
    - Dual arm:   n_arms=2, order [left, right], output (2, 4, 4)
    """

    n_arms: int = 1

    # Subclasses may override these with concrete values.
    LINK_NAMES: list[str] = []
    MESH_PATHS: list[str] = []
    TCP_NAME: str = ""

    grasp_point_R_align: Optional[np.ndarray] = None  # (3,3) or None

    # ------------------------------------------------------------------
    # Shared static helpers (previously duplicated across many files)
    # ------------------------------------------------------------------

    @staticmethod
    def build_mesh_adjust(robot, name_list: list[str]) -> dict[str, np.ndarray]:
        """Build mesh alignment matrices from URDF geometry.

        For each link whose name is in *name_list* and that carries at least
        one geometry primitive, compute T @ S where T encodes the geometry
        origin (translation + quaternion rotation) and S encodes the scale.
        """
        from scipy.spatial.transform import Rotation

        adjust: dict[str, np.ndarray] = {name: np.eye(4) for name in name_list}
        for link in robot.links:
            if link is None or link.name not in adjust or len(link.geometry) == 0:
                continue
            geo = link.geometry[0]
            T = np.eye(4)
            T[:3, 3] = geo._wT[:3, 3]
            T[:3, :3] = Rotation.from_quat(geo._wq).as_matrix()
            S = np.eye(4)
            S[:3, :3] = np.diag(geo.scale)
            adjust[link.name] = T @ S
        return adjust

    @staticmethod
    def build_joint_idx_map(robot) -> dict[str, int]:
        """Build joint name -> index mapping (by link traversal order)."""
        result: dict[str, int] = {}
        idx = 0
        for link in robot.links:
            if link.isjoint:
                result[link.name] = idx
                idx += 1
        return result

    # ------------------------------------------------------------------
    # grasp_point_rotation_align helper
    # ------------------------------------------------------------------

    @staticmethod
    def _build_grasp_point_R_align(val) -> Optional[np.ndarray]:
        """Convert grasp_point_rotation_align config to (3,3) rotation matrix.

        Accepts None, [rx, ry, rz] (degrees, euler_xyz), or 3x3 nested list.
        """
        if val is None:
            return None
        from scipy.spatial.transform import Rotation

        arr = np.array(val, dtype=np.float64)
        if arr.shape == (3,):
            return Rotation.from_euler("xyz", arr, degrees=True).as_matrix()
        if arr.shape == (3, 3):
            return arr
        raise ValueError(
            f"grasp_point_rotation_align shape should be (3,) or (3,3), got {arr.shape}"
        )

    # ------------------------------------------------------------------
    # Abstract / default FK interface
    # ------------------------------------------------------------------

    @abstractmethod
    def fkine_flange(self, q: np.ndarray) -> np.ndarray:
        """Arm end-effector pose (no gripper offset).

        Args:
            q: (n_joints,)
        Returns:
            (n_arms, 4, 4)
        """

    @abstractmethod
    def _fkine_tcp_raw(self, q: np.ndarray) -> np.ndarray:
        """Raw gripper FK (before rotation alignment). Subclass implements.

        Args:
            q: (n_joints,)
        Returns:
            (n_arms, 4, 4)
        """

    def fkine_tcp(self, q: np.ndarray) -> np.ndarray:
        """Gripper-closed end-effector pose; equals fkine_flange if no gripper.
        Automatically applies grasp_point_R_align if set.

        Args:
            q: (n_joints,)
        Returns:
            (n_arms, 4, 4)
        """
        result = self._fkine_tcp_raw(q)
        if self.grasp_point_R_align is not None:
            result = result.copy()
            result[:, :3, :3] = result[:, :3, :3] @ self.grasp_point_R_align
        return result

    @abstractmethod
    def fkine_all(self, q: np.ndarray) -> np.ndarray:
        """All link transforms.

        Args:
            q: (n_joints,)
        Returns:
            (n_arms, n_links, 4, 4)
        """

    def fkine(self, q: np.ndarray) -> np.ndarray:
        """Alias for fkine_tcp."""
        return self.fkine_tcp(q)

    def gripper_scalar(self, q: np.ndarray) -> float:
        """Extract gripper open/close scalar from joint vector (default: last)."""
        return float(np.asarray(q, dtype=np.float64)[-1])

    def gripper_scalars(self, q: np.ndarray) -> np.ndarray:
        """Per-arm gripper scalars, shape (n_arms,)."""
        return np.array([self.gripper_scalar(q)], dtype=np.float64)
