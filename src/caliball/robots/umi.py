"""UMI handheld gripper URDF wrapper."""

from __future__ import annotations

import numpy as np
from roboticstoolbox.robot.Robot import Robot

from caliball.robots._base import BaseTF
from caliball.robots._registry import register_robot


class _UmiGripper(Robot):
    """SolidWorks-exported UMI gripper assembly."""

    def __init__(self):
        links, name, urdf_string, urdf_filepath = self.URDF_read(
            file_path="urdf/手持夹爪总成_0.8-urdf.urdf",
            tld="./third_party/urdf/umi",
        )
        super().__init__(
            links,
            name=name,
            manufacturer="umi",
            urdf_string=urdf_string,
            urdf_filepath=urdf_filepath,
        )


@register_robot("umi")
class UmiTF(BaseTF):
    """UMI handheld gripper.

    Runtime state can be either:
      - one scalar gripper opening in meters, for URDF smoke tests
      - one eef block: ``[x, y, z, rot6d, gripper]``

    For eef blocks this class is a direct pose adapter: no arm FK is applied.
    """

    n_arms = 1
    LINK_NAMES = ["base_link", "link1", "link2"]
    TCP_NAME = "base_link"
    MESH_PATHS = [
        "third_party/urdf/umi/meshes/base_link.STL",
        "third_party/urdf/umi/meshes/link1.STL",
        "third_party/urdf/umi/meshes/link2.STL",
    ]

    MAX_GRIPPER_OPENING = 0.08
    EEF_DIM = 10
    TOP_TO_URDF_BASE_VERTICAL_M = 0.24
    TOP_BRACKET_TILT_DEG = 36.0

    _tilt = np.deg2rad(TOP_BRACKET_TILT_DEG)
    _sin_tilt = float(np.sin(_tilt))
    _cos_tilt = float(np.cos(_tilt))

    # Rendering-only transform from URDF frame to recorded UMI eef frame.
    #
    # URDF axes: x = forward, y = left, z = up.
    # UMI eef axes: x = left, y = forward/up along the top bracket,
    # z = forward/down. The top frame is about 240 mm above the URDF base.
    RENDER_ALIGN_T = np.array(
        [
            [0.0, 1.0, 0.0, 0.0],
            [
                _sin_tilt,
                0.0,
                _cos_tilt,
                -TOP_TO_URDF_BASE_VERTICAL_M * _cos_tilt,
            ],
            [
                _cos_tilt,
                0.0,
                -_sin_tilt,
                TOP_TO_URDF_BASE_VERTICAL_M * _sin_tilt,
            ],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    def __init__(self, name_list=None, tcp_name=None, **kwargs):
        self.name_list = list(name_list or self.LINK_NAMES)
        self.tcp_name = tcp_name or self.TCP_NAME
        self._robot = _UmiGripper()
        self._jmap = BaseTF.build_joint_idx_map(self._robot)
        self.mesh_adjust = BaseTF.build_mesh_adjust(self._robot, self.name_list)

    @property
    def robot(self):
        return self._robot

    @staticmethod
    def _rot6d_to_matrix(rot6d: np.ndarray) -> np.ndarray:
        a1 = np.asarray(rot6d[:3], dtype=np.float64)
        a2 = np.asarray(rot6d[3:6], dtype=np.float64)
        b1 = a1 / max(np.linalg.norm(a1), 1e-12)
        a2_orth = a2 - np.dot(b1, a2) * b1
        b2 = a2_orth / max(np.linalg.norm(a2_orth), 1e-12)
        b3 = np.cross(b1, b2)
        return np.stack([b1, b2, b3], axis=1)

    @classmethod
    def _eef_to_hom(cls, block: np.ndarray) -> np.ndarray:
        block = np.asarray(block, dtype=np.float64)
        T = np.eye(4, dtype=np.float64)
        T[:3, 3] = block[:3]
        T[:3, :3] = cls._rot6d_to_matrix(block[3:9])
        return T

    def _eef_block(self, q: np.ndarray) -> np.ndarray | None:
        q = np.asarray(q, dtype=np.float64).reshape(-1)
        if q.size >= self.EEF_DIM:
            return q[: self.EEF_DIM]
        return None

    def _expand_q(self, q: np.ndarray) -> np.ndarray:
        q = np.asarray(q, dtype=np.float64).reshape(-1)
        eef_block = self._eef_block(q)
        if eef_block is not None:
            opening = float(eef_block[-1])
        else:
            opening = float(q[0]) if q.size else 0.0
        opening = min(max(opening, 0.0), self.MAX_GRIPPER_OPENING)

        out = np.zeros(self._robot.n, dtype=np.float64)
        if "link1" in self._jmap:
            out[self._jmap["link1"]] = opening / 2.0
        if "link2" in self._jmap:
            out[self._jmap["link2"]] = -opening / 2.0
        return out

    def fkine_flange(self, q: np.ndarray) -> np.ndarray:
        eef_block = self._eef_block(q)
        if eef_block is not None:
            return self._eef_to_hom(eef_block)[np.newaxis]
        T = self.robot.fkine(self._expand_q(q), end=self.tcp_name).A
        return T[np.newaxis]

    def _fkine_tcp_raw(self, q: np.ndarray) -> np.ndarray:
        return self.fkine_flange(q)

    def fkine_all(self, q: np.ndarray) -> np.ndarray:
        eef_block = self._eef_block(q)
        if eef_block is None:
            q_full = self._expand_q(q)
            tfs = np.array([self.robot.fkine(q_full, end=name).A for name in self.name_list])
            for i, name in enumerate(self.name_list):
                tfs[i] = tfs[i] @ self.mesh_adjust[name]
            return (self.RENDER_ALIGN_T[np.newaxis] @ tfs)[np.newaxis]

        q_full = self._expand_q(eef_block)
        local = np.array([self.robot.fkine(q_full, end=name).A for name in self.name_list])
        for i, name in enumerate(self.name_list):
            local[i] = local[i] @ self.mesh_adjust[name]
        T_mount = self._eef_to_hom(eef_block) @ self.RENDER_ALIGN_T
        return (T_mount[np.newaxis] @ local)[np.newaxis]

    def gripper_scalar(self, q: np.ndarray) -> float:
        q = np.asarray(q, dtype=np.float64).reshape(-1)
        eef_block = self._eef_block(q)
        if eef_block is not None:
            return float(eef_block[-1])
        return float(q[0]) if q.size else 0.0
