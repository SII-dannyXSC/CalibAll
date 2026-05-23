"""UR5e arm (pure arm, uses URDF)."""

from __future__ import annotations

import numpy as np
from roboticstoolbox.robot.Robot import Robot

from caliball.robots._base import BaseTF
from caliball.robots._registry import register_robot
from caliball.robots._rtb import RoboticsToolBoxTF


# ---------------------------------------------------------------------------
# URDF loader (inlined from robot/urdf/ur5e.py)
# ---------------------------------------------------------------------------

class _Ur5e(Robot):
    def __init__(self):
        links, name, urdf_string, urdf_filepath = self.URDF_read(
            file_path="ur_description/urdf/ur5e.urdf",
            tld="./third_party/urdf/universal_robots",
        )
        super().__init__(
            links,
            name=name,
            manufacturer="ur5e",
            urdf_string=urdf_string,
            urdf_filepath=urdf_filepath,
        )


# ---------------------------------------------------------------------------
# TF implementation
# ---------------------------------------------------------------------------

@register_robot("ur5e")
class Ur5eTF(RoboticsToolBoxTF):
    """UR5e arm (pure arm). q: (6,), fkine_tcp = fkine_flange."""

    ARM_NAMES = [
        "base_link", "shoulder_link", "upper_arm_link", "forearm_link",
        "wrist_1_link", "wrist_2_link", "wrist_3_link",
    ]
    ARM_TCP_NAME = "wrist_3_link"

    LINK_NAMES = ARM_NAMES
    TCP_NAME = ARM_TCP_NAME
    MESH_PATHS = [
        "third_party/urdf/universal_robots/ur_description/meshes/ur5e/visual/base.dae",
        "third_party/urdf/universal_robots/ur_description/meshes/ur5e/visual/shoulder.dae",
        "third_party/urdf/universal_robots/ur_description/meshes/ur5e/visual/upperarm.dae",
        "third_party/urdf/universal_robots/ur_description/meshes/ur5e/visual/forearm.dae",
        "third_party/urdf/universal_robots/ur_description/meshes/ur5e/visual/wrist1.dae",
        "third_party/urdf/universal_robots/ur_description/meshes/ur5e/visual/wrist2.dae",
        "third_party/urdf/universal_robots/ur_description/meshes/ur5e/visual/wrist3.dae",
    ]

    def __init__(self, name_list=None, tcp_name=None, **kwargs):
        super().__init__(name_list or self.ARM_NAMES, tcp_name or self.ARM_TCP_NAME)
        self._robot = _Ur5e()
        self._init_urdf()

    @property
    def robot(self):
        return self._robot
