"""RoboticsToolBoxTF -- base class for arms that use roboticstoolbox FK."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from caliball.robots._base import BaseTF


class RoboticsToolBoxTF(BaseTF, ABC):
    """Base class for single-arm robots using roboticstoolbox.

    Pure arm: fkine_tcp = fkine_flange.
    """

    def __init__(self, name_list: list[str], tcp_name: str, **kwargs):
        self.name_list = list(name_list)
        self.tcp_name = tcp_name
        self.mesh_adjust: dict[str, np.ndarray] = {}

    @property
    @abstractmethod
    def robot(self):
        """Return the underlying roboticstoolbox Robot instance."""

    def _init_urdf(self) -> None:
        """Call after self.robot is available to build mesh_adjust dict."""
        self.mesh_adjust = BaseTF.build_mesh_adjust(self.robot, self.name_list)

    def fkine_flange(self, q: np.ndarray) -> np.ndarray:
        """Arm end-effector pose, shape (1, 4, 4)."""
        q = np.asarray(q, dtype=np.float64)
        eef_pose = self.robot.fkine(q, end=self.tcp_name).A
        return eef_pose[np.newaxis]  # (1, 4, 4)

    def _fkine_tcp_raw(self, q: np.ndarray) -> np.ndarray:
        """Pure arm has no gripper, same as fkine_flange, shape (1, 4, 4)."""
        return self.fkine_flange(q)

    def fkine_all(self, q: np.ndarray) -> np.ndarray:
        """All link transforms (with mesh alignment), shape (1, n_links, 4, 4)."""
        q = np.asarray(q, dtype=np.float64)
        all_tf = [self.robot.fkine(q, end=name).A for name in self.name_list]
        result = np.array(all_tf)[np.newaxis]  # (1, n_links, 4, 4)
        # Apply mesh alignment if available
        if self.mesh_adjust:
            for idx, name in enumerate(self.name_list):
                if name in self.mesh_adjust:
                    result[0, idx] = result[0, idx] @ self.mesh_adjust[name]
        return result
