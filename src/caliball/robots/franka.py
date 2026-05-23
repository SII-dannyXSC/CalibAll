"""Franka Panda arm (pure arm, no URDF file -- uses rtb.models.Panda)."""

from __future__ import annotations

import numpy as np
import roboticstoolbox as rtb

from caliball.robots._base import BaseTF
from caliball.robots._registry import register_robot


@register_robot("franka")
class FrankaTF(BaseTF):
    """Franka Panda arm (pure arm). q: (7,), fkine_tcp = fkine_flange.

    Uses rtb.models.Panda() directly; FK uses hardcoded fkine_all indices.
    """

    JOINT_NUM = 7

    LINK_NAMES: list[str] = []
    TCP_NAME = ""
    MESH_PATHS = [
        "third_party/urdf/franka_description/meshes/visual/link0.dae",
        "third_party/urdf/franka_description/meshes/visual/link1.dae",
        "third_party/urdf/franka_description/meshes/visual/link2.dae",
        "third_party/urdf/franka_description/meshes/visual/link3.dae",
        "third_party/urdf/franka_description/meshes/visual/link4.dae",
        "third_party/urdf/franka_description/meshes/visual/link5.dae",
        "third_party/urdf/franka_description/meshes/visual/link6.dae",
        "third_party/urdf/franka_description/meshes/visual/link7.dae",
    ]

    def __init__(self, **kwargs):
        self._robot = rtb.models.Panda()

    def fkine_flange(self, q: np.ndarray) -> np.ndarray:
        """panda_link8 pose, shape (1, 4, 4)."""
        q = np.asarray(q, dtype=np.float64)[: self.JOINT_NUM]
        all_tf = self._robot.fkine_all(q)
        eef_pose = all_tf[9].A
        return eef_pose[np.newaxis]  # (1, 4, 4)

    def _fkine_tcp_raw(self, q: np.ndarray) -> np.ndarray:
        """Pure arm, same as fkine_flange, shape (1, 4, 4)."""
        return self.fkine_flange(q)

    def fkine_all(self, q: np.ndarray) -> np.ndarray:
        """Arm links (link1-link8 + panda_hand), shape (1, 9, 4, 4)."""
        q = np.asarray(q, dtype=np.float64)[: self.JOINT_NUM]
        all_tf = self._robot.fkine_all(q)
        link8 = all_tf[9]
        arm_links = all_tf[1:9]
        tf_hand = link8 @ self._robot.grippers[0].links[0].A()
        tfs = np.array([item.A for item in arm_links] + [tf_hand.A])
        return tfs[np.newaxis]  # (1, 9, 4, 4)
