"""ALOHA dual-arm robot (front-left + front-right)."""

from __future__ import annotations

from typing import Literal, Sequence, Union

import numpy as np
from scipy.spatial.transform import Rotation as R
from roboticstoolbox.robot.Robot import Robot

from caliball.robots._base import BaseTF
from caliball.robots._dual_arm import DualArmTF
from caliball.robots._registry import register_robot


# ---------------------------------------------------------------------------
# URDF loader (inlined from robot/urdf/aloha.py)
# ---------------------------------------------------------------------------

class _AlohaCobotMagic(Robot):
    """ALOHA dual-arm robot (front-left + front-right only)."""

    def __init__(self):
        links, name, urdf_string, urdf_filepath = self.URDF_read(
            file_path="aloha_new_description/urdf/aloha_new.urdf",
            tld="./third_party/urdf/aloha",
        )
        super().__init__(
            links,
            name=name,
            manufacturer="aloha",
            urdf_string=urdf_string,
            urdf_filepath=urdf_filepath,
        )


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PREFIX = {"left": "fl_", "right": "fr_"}
_DEFAULT_TCP = {"left": "fl_link6", "right": "fr_link6"}
_TIP_LINK = {"left": "fl_link8", "right": "fr_link8"}


def _build_aloha_joint_idx_map(robot):
    """Robot joint name -> q vector index (robust to jindex vs link order)."""
    pairs = []
    for link in robot.links:
        j = getattr(link, "joint", None)
        if j is None or not getattr(j, "isjoint", True):
            continue
        name = (getattr(j, "name", None) or "").strip()
        ji = getattr(j, "jindex", None)
        if not name or ji is None:
            continue
        pairs.append((int(ji), name))
    if pairs:
        pairs.sort(key=lambda x: x[0])
        if len(pairs) == int(getattr(robot, "n", len(pairs))):
            return {nm: i for i, nm in pairs}
    # Fallback: enumerate by links order
    result = []
    for link in robot.links:
        j = getattr(link, "joint", None)
        if j is not None and getattr(j, "isjoint", True):
            result.append(
                (getattr(j, "jindex", len(result)), getattr(j, "name", ""))
            )
    result.sort(key=lambda x: x[0])
    return {nm: i for i, nm in result}


# ---------------------------------------------------------------------------
# Single-arm TF
# ---------------------------------------------------------------------------

class _AlohaArmTF(BaseTF):
    """ALOHA single arm. q: (7,) = 6 arm joints + 1 gripper distance [0, 0.1].

    Holds the full dual-arm robot (can be shared); side determines joint prefix (fl_ / fr_).
    Gripper internally expanded as two symmetric prismatic: joint7 = +g/2, joint8 = -g/2.

    fkine_flange:     wrist (tcp_name) pose, shape (1, 4, 4).
    fkine_tcp: fingertip (link8) pose, shape (1, 4, 4).
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

        self._robot = robot if robot is not None else _AlohaCobotMagic()
        self._jmap = _build_aloha_joint_idx_map(self._robot)

        self.mesh_adjust = BaseTF.build_mesh_adjust(self._robot, self.name_list)

    def _expand_q(self, q: np.ndarray) -> np.ndarray:
        """q: (7,) -> full robot joint vector (nq,), other joints set to 0."""
        q = np.asarray(q, dtype=np.float64)
        p = self._prefix
        idx = self._jmap
        out = np.zeros(self._robot.n, dtype=np.float64)
        for i in range(6):
            jname = f"{p}joint{i + 1}"
            if jname in idx:
                out[idx[jname]] = q[i]
        gripper = float(q[6])
        if f"{p}joint7" in idx:
            out[idx[f"{p}joint7"]] = gripper / 2
        if f"{p}joint8" in idx:
            out[idx[f"{p}joint8"]] = -gripper / 2  # symmetric open
        return out

    def fkine_flange(self, q: np.ndarray) -> np.ndarray:
        T = self._robot.fkine(self._expand_q(q), end=self.tcp_name).A
        return T[np.newaxis]  # (1, 4, 4)

    def _fkine_tcp_raw(self, q: np.ndarray) -> np.ndarray:
        T = self._robot.fkine(self._expand_q(q), end=self.tip_link).A
        return T[np.newaxis]  # (1, 4, 4)

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

# @register_robot("aloha_cobot_magic")
# @register_robot("aloha")
class AlohaCobotMagicTF(DualArmTF):
    """ALOHA dual-arm. q: (14,) = left arm 7 + right arm 7, order [left, right].

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
    DEFAULT_TCP = "fl_link8"

    LINK_NAMES = ALL_NAMES
    TCP_NAME = "fl_link8"
    MESH_PATHS = [
        # left arm
        "third_party/urdf/aloha/aloha_new_description/meshes/base_link.dae",
        "third_party/urdf/aloha/aloha_new_description/meshes/link1.dae",
        "third_party/urdf/aloha/aloha_new_description/meshes/link2.dae",
        "third_party/urdf/aloha/aloha_new_description/meshes/link3.dae",
        "third_party/urdf/aloha/aloha_new_description/meshes/link4.dae",
        "third_party/urdf/aloha/aloha_new_description/meshes/link5.dae",
        "third_party/urdf/aloha/aloha_new_description/meshes/piper_slave_meshes/link6.dae",
        "third_party/urdf/aloha/aloha_new_description/meshes/piper_slave_meshes/link7.dae",
        "third_party/urdf/aloha/aloha_new_description/meshes/piper_slave_meshes/link8.dae",
        # right arm
        "third_party/urdf/aloha/aloha_new_description/meshes/base_link.dae",
        "third_party/urdf/aloha/aloha_new_description/meshes/link1.dae",
        "third_party/urdf/aloha/aloha_new_description/meshes/link2.dae",
        "third_party/urdf/aloha/aloha_new_description/meshes/link3.dae",
        "third_party/urdf/aloha/aloha_new_description/meshes/link4.dae",
        "third_party/urdf/aloha/aloha_new_description/meshes/link5.dae",
        "third_party/urdf/aloha/aloha_new_description/meshes/piper_slave_meshes/link6.dae",
        "third_party/urdf/aloha/aloha_new_description/meshes/piper_slave_meshes/link7.dae",
        "third_party/urdf/aloha/aloha_new_description/meshes/piper_slave_meshes/link8.dae",
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

        _robot = _AlohaCobotMagic()  # loaded once, shared by both arms
        super().__init__(
            left=_AlohaArmTF("left", left_names, left_tcp, robot=_robot),
            right=_AlohaArmTF("right", right_names, right_tcp, robot=_robot),
            n_left_joints=7,
        )
        self.grasp_point_R_align = self._build_grasp_point_R_align(
            grasp_point_rotation_align
        )
