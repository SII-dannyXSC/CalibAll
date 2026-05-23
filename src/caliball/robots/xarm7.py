"""xArm7 pure arm (menagerie URDF)."""

from __future__ import annotations

import numpy as np
from roboticstoolbox.robot.Robot import Robot

from caliball.robots._registry import register_robot
from caliball.robots._rtb import RoboticsToolBoxTF


# ---------------------------------------------------------------------------
# URDF loader (shared with xarm7_xarmgripper.py)
# ---------------------------------------------------------------------------

class XArm7URDF(Robot):
    """xArm7 + gripper (mujoco_menagerie ufactory_xarm7)."""

    def __init__(self):
        links, name, urdf_string, urdf_filepath = self.URDF_read(
            file_path="xarm7.urdf",
            tld="./third_party/urdf/xarm7",
        )
        super().__init__(
            links,
            name=name,
            manufacturer="UFACTORY",
            urdf_string=urdf_string,
            urdf_filepath=urdf_filepath,
        )


# ---------------------------------------------------------------------------
# Pure arm TF
# ---------------------------------------------------------------------------

@register_robot("xarm7")
class XArm7ArmTF(RoboticsToolBoxTF):
    """xArm7 arm chain (link_base...link7), only first 7 joint angles;
    remaining joints padded to 0.
    """

    ARM_NAMES = [
        "link_base", "link1", "link2", "link3",
        "link4", "link5", "link6", "link7",
    ]
    ARM_TCP_NAME = "link7"
    MESH_PATHS = [
        "third_party/urdf/xarm7/assets/link_base.stl",
        "third_party/urdf/xarm7/assets/link1.stl",
        "third_party/urdf/xarm7/assets/link2.stl",
        "third_party/urdf/xarm7/assets/link3.stl",
        "third_party/urdf/xarm7/assets/link4.stl",
        "third_party/urdf/xarm7/assets/link5.stl",
        "third_party/urdf/xarm7/assets/link6.stl",
        "third_party/urdf/xarm7/assets/end_tool.stl",
    ]

    def __init__(self, name_list=None, tcp_name=None, **kwargs):
        super().__init__(list(name_list or self.ARM_NAMES), tcp_name or self.ARM_TCP_NAME)
        self._robot = XArm7URDF()

    @property
    def robot(self):
        return self._robot

    def _pad(self, q_arm: np.ndarray) -> np.ndarray:
        q_arm = np.asarray(q_arm, dtype=np.float64)
        q = np.zeros(self._robot.n, dtype=np.float64)
        q[:7] = q_arm[:7]
        return q

    def fkine_flange(self, q: np.ndarray) -> np.ndarray:
        return self.robot.fkine(self._pad(q), end=self.tcp_name).A[np.newaxis]

    def _fkine_tcp_raw(self, q: np.ndarray) -> np.ndarray:
        return self.fkine_flange(q)

    def fkine_all(self, q: np.ndarray) -> np.ndarray:
        q_pad = self._pad(q)
        tfs = [self.robot.fkine(q_pad, end=name).A for name in self.name_list]
        return np.array(tfs)[np.newaxis]
