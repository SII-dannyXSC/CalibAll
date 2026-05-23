"""ARX5 dual-arm robot (Agilex_Cobot_Magic_robotwin)."""

from __future__ import annotations

from typing import Literal, Sequence, Union

import numpy as np
from roboticstoolbox.robot.Robot import Robot

from caliball.robots._base import BaseTF
from caliball.robots._dual_arm import DualArmTF
from caliball.robots._registry import register_robot


# ---------------------------------------------------------------------------
# URDF loader (inlined from robot/urdf/arx5_robotwin.py)
# ---------------------------------------------------------------------------

class _Arx5Robotwin(Robot):
    """ARX5 dual-arm robot (front-left + front-right)."""

    def __init__(self):
        links, name, urdf_string, urdf_filepath = self.URDF_read(
            file_path="urdf/arx5_dual_arm.urdf",
            tld="./third_party/urdf/arx5",
        )
        super().__init__(
            links,
            name=name,
            manufacturer="arx5_robotwin",
            urdf_string=urdf_string,
            urdf_filepath=urdf_filepath,
        )


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PREFIX = {"left": "fl_", "right": "fr_"}
_DEFAULT_TCP = {"left": "fl_link6", "right": "fr_link6"}
_TIP_LINK = {"left": "fl_link8", "right": "fr_link8"}

# link8.dae mesh furthest vertex (meters)
LINK8_MESH_TIP_OFFSET_HOM = np.array(
    [0.07099997, 0.0244944, 0.005103468, 1.0], dtype=np.float64
)


def _hom_with_mesh_tip(T: np.ndarray, tip: np.ndarray) -> np.ndarray:
    out = np.array(T, dtype=np.float64, copy=True)
    out[:3, 3] = (T @ tip)[:3]
    return out


# ---------------------------------------------------------------------------
# Single-arm TF
# ---------------------------------------------------------------------------

class _Arx5ArmTF(BaseTF):
    """ARX5 single arm. q: (7,) = 6 arm joints + 1 gripper distance [0, 0.1].

    Holds the full dual-arm robot (can be shared); side determines joint prefix (fl_ / fr_).
    Gripper internally expanded as two co-directional prismatic: joint7 = joint8 = g/2.

    fkine_flange:     wrist (tcp_name) pose, shape (1, 4, 4).
    fkine_tcp: fingertip (link8 + mesh furthest point offset), shape (1, 4, 4).
    fkine_all:     this arm all links (with mesh alignment), shape (1, n_links, 4, 4).
    """

    def __init__(
        self,
        side: Literal["left", "right"],
        name_list,
        tcp_name: str,
        robot=None,
        **kwargs,
    ):
        self.side = side
        self.name_list = list(name_list)
        self.tcp_name = tcp_name or _DEFAULT_TCP[side]
        self.tip_link = _TIP_LINK[side]
        self._prefix = _PREFIX[side]

        self._robot = robot if robot is not None else _Arx5Robotwin()
        self._jmap = BaseTF.build_joint_idx_map(self._robot)

        self.mesh_adjust = BaseTF.build_mesh_adjust(self._robot, self.name_list)

    def _expand_q(self, q: np.ndarray) -> np.ndarray:
        """q: (7,) -> full robot joint vector (nq,), other joints set to 0.

        URDF joints named by link name (fl_link1 ... fl_link8).
        """
        q = np.asarray(q, dtype=np.float64)
        p = self._prefix
        idx = self._jmap
        out = np.zeros(self._robot.n, dtype=np.float64)
        for i in range(6):
            jname = f"{p}link{i + 1}"  # fl_link1 ... fl_link6
            if jname in idx:
                out[idx[jname]] = q[i]
        gripper = float(q[6])
        if f"{p}link7" in idx:
            out[idx[f"{p}link7"]] = gripper / 2
        if f"{p}link8" in idx:
            out[idx[f"{p}link8"]] = gripper / 2  # co-directional open
        return out

    def fkine_flange(self, q: np.ndarray) -> np.ndarray:
        T = self._robot.fkine(self._expand_q(q), end=self.tcp_name).A
        return T[np.newaxis]  # (1, 4, 4)

    def _fkine_tcp_raw(self, q: np.ndarray) -> np.ndarray:
        # Grip point independent of open/close: force q[6]=0 (closed) for UV stability
        q_closed = np.array(q, dtype=np.float64)
        if q_closed.shape[0] > 6:
            q_closed[6] = 0.0
        T = np.asarray(
            self._robot.fkine(self._expand_q(q_closed), end=self.tip_link).A,
            dtype=np.float64,
        )
        return _hom_with_mesh_tip(T, LINK8_MESH_TIP_OFFSET_HOM)[np.newaxis]  # (1, 4, 4)

    def fkine_all(self, q: np.ndarray) -> np.ndarray:
        q_full = self._expand_q(q)
        tfs = np.array(
            [self._robot.fkine(q_full, end=name).A for name in self.name_list]
        )
        for i, name in enumerate(self.name_list):
            tfs[i] = tfs[i] @ self.mesh_adjust[name]
        return tfs[np.newaxis]  # (1, n_links, 4, 4)


# ---------------------------------------------------------------------------
# Dual-arm composite
# ---------------------------------------------------------------------------

@register_robot("arx5_robotwin")
class Arx5RobotwinTF(DualArmTF):
    """ARX5 dual-arm. q: (14,) = left arm 7 + right arm 7, order [left, right].

    names: all link names (fl_* auto-assigned left, fr_* auto-assigned right).
    tcp_name: str or [left_tcp, right_tcp], default fl_link6 / fr_link6.
    """

    ALL_NAMES = [
        # left arm
        "fl_base_link", "fl_link1", "fl_link2", "fl_link3",
        "fl_link4", "fl_link5", "fl_link6", "fl_link7", "fl_link8",
        # right arm
        "fr_base_link", "fr_link1", "fr_link2", "fr_link3",
        "fr_link4", "fr_link5", "fr_link6", "fr_link7", "fr_link8",
    ]
    DEFAULT_TCP = ["fl_link8", "fr_link8"]

    LINK_NAMES = ALL_NAMES
    TCP_NAME = "fl_link8"
    MESH_PATHS = [
        # left arm
        "third_party/urdf/arx5/urdf/aloha_maniskill_sim/meshes/base_arm.dae",
        "third_party/urdf/arx5/urdf/aloha_maniskill_sim/meshes/link1.dae",
        "third_party/urdf/arx5/urdf/aloha_maniskill_sim/meshes/link2.dae",
        "third_party/urdf/arx5/urdf/aloha_maniskill_sim/meshes/link3.dae",
        "third_party/urdf/arx5/urdf/aloha_maniskill_sim/meshes/link4.dae",
        "third_party/urdf/arx5/urdf/aloha_maniskill_sim/meshes/link5.dae",
        "third_party/urdf/arx5/urdf/aloha_maniskill_sim/meshes/link6.dae",
        "third_party/urdf/arx5/urdf/aloha_maniskill_sim/meshes/link7.dae",
        "third_party/urdf/arx5/urdf/aloha_maniskill_sim/meshes/link8.dae",
        # right arm
        "third_party/urdf/arx5/urdf/aloha_maniskill_sim/meshes/base_arm.dae",
        "third_party/urdf/arx5/urdf/aloha_maniskill_sim/meshes/link1.dae",
        "third_party/urdf/arx5/urdf/aloha_maniskill_sim/meshes/link2.dae",
        "third_party/urdf/arx5/urdf/aloha_maniskill_sim/meshes/link3.dae",
        "third_party/urdf/arx5/urdf/aloha_maniskill_sim/meshes/link4.dae",
        "third_party/urdf/arx5/urdf/aloha_maniskill_sim/meshes/link5.dae",
        "third_party/urdf/arx5/urdf/aloha_maniskill_sim/meshes/link6.dae",
        "third_party/urdf/arx5/urdf/aloha_maniskill_sim/meshes/link7.dae",
        "third_party/urdf/arx5/urdf/aloha_maniskill_sim/meshes/link8.dae",
    ]

    def __init__(
        self,
        names=None,
        tcp_name=None,
        grasp_point_rotation_align=None,
        **kwargs,
    ):
        all_names = list(names or self.ALL_NAMES)
        left_names = [n for n in all_names if n.startswith("fl_")]
        right_names = [n for n in all_names if n.startswith("fr_")]

        if tcp_name is None:
            tcp_name = self.DEFAULT_TCP

        if isinstance(tcp_name, str):
            left_tcp, right_tcp = _DEFAULT_TCP["left"], tcp_name
        else:
            seq = list(tcp_name)
            left_tcp = seq[0] if len(seq) >= 1 else _DEFAULT_TCP["left"]
            right_tcp = seq[1] if len(seq) >= 2 else _DEFAULT_TCP["right"]

        _robot = _Arx5Robotwin()  # loaded once, shared by both arms
        super().__init__(
            left=_Arx5ArmTF("left", left_names, left_tcp, robot=_robot),
            right=_Arx5ArmTF("right", right_names, right_tcp, robot=_robot),
            n_left_joints=7,
        )
        self.grasp_point_R_align = self._build_grasp_point_R_align(
            grasp_point_rotation_align
        )
