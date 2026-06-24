"""ARX X5A single-arm robot (ARX-X5 SolidWorks URDF)."""

from __future__ import annotations

import numpy as np
from roboticstoolbox.robot.Robot import Robot

from caliball.robots._base import BaseTF
from caliball.robots._registry import register_robot


class _ArxX5A(Robot):
    """ARX X5A arm with two prismatic gripper fingers."""

    def __init__(self):
        links, name, urdf_string, urdf_filepath = self.URDF_read(
            file_path="urdf/X5A.urdf",
            tld="./third_party/urdf/ARX-X5",
        )
        super().__init__(
            links,
            name=name,
            manufacturer="ARX",
            urdf_string=urdf_string,
            urdf_filepath=urdf_filepath,
        )


@register_robot("arx_x5")
class ArxX5TF(BaseTF):
    """ARX X5A TF model.

    Input q accepts either:
    - (7,): 6 arm joints + one gripper opening scalar.
    - (8,): raw URDF joints joint1..joint8.

    For the 7D form, the gripper scalar is split symmetrically into joint7
    and joint8, each clamped to the URDF limit [0, 0.044].
    """

    ARM_NAMES = [
        "base_link",
        "link1",
        "link2",
        "link3",
        "link4",
        "link5",
        "link6",
        "link7",
        "link8",
    ]
    ARM_TCP_NAME = "link6"
    FINGER_NAMES = ("link7", "link8")
    GRIPPER_CLOSED = 0.0
    GRIPPER_MAX = 0.088
    # Midpoint of link7/link8 foremost finger vertices in the closed gripper
    # state, expressed in link6/flange frame.
    TCP_IN_LINK6_HOM = np.array(
        [0.15757000232458115, -2.0003725290299165e-06, 0.000999998375232211, 1.0],
        dtype=np.float64,
    )

    LINK_NAMES = ARM_NAMES
    TCP_NAME = ARM_TCP_NAME
    MESH_PATHS = [
        "third_party/urdf/ARX-X5/meshes/base_link.STL",
        "third_party/urdf/ARX-X5/meshes/link1.STL",
        "third_party/urdf/ARX-X5/meshes/link2.STL",
        "third_party/urdf/ARX-X5/meshes/link3.STL",
        "third_party/urdf/ARX-X5/meshes/link4.STL",
        "third_party/urdf/ARX-X5/meshes/link5.STL",
        "third_party/urdf/ARX-X5/meshes/link6.STL",
        "third_party/urdf/ARX-X5/meshes/link7.STL",
        "third_party/urdf/ARX-X5/meshes/link8.STL",
    ]

    def __init__(
        self,
        name_list=None,
        tcp_name=None,
        grasp_point_rotation_align=None,
        **kwargs,
    ):
        self.name_list = list(name_list or self.ARM_NAMES)
        self.tcp_name = tcp_name or self.ARM_TCP_NAME
        self._robot = _ArxX5A()
        self.mesh_adjust = BaseTF.build_mesh_adjust(self._robot, self.name_list)
        self.grasp_point_R_align = self._build_grasp_point_R_align(
            grasp_point_rotation_align
        )

    @property
    def robot(self):
        return self._robot

    def _expand_q(self, q: np.ndarray) -> np.ndarray:
        q = np.asarray(q, dtype=np.float64)
        out = np.zeros(self._robot.n, dtype=np.float64)
        if q.shape[0] >= self._robot.n:
            out[:] = q[: self._robot.n]
            return out
        if q.shape[0] < 7:
            raise ValueError(
                f"ARX X5A expects 7D compact q or 8D raw q, got {q.shape[0]}D"
            )

        out[:6] = q[:6]
        finger = min(max(float(q[6]) / 2.0, 0.0), 0.044)
        out[6] = finger
        out[7] = finger
        return out

    def fkine_flange(self, q: np.ndarray) -> np.ndarray:
        return self._robot.fkine(self._expand_q(q), end=self.tcp_name).A[np.newaxis]

    def _fkine_tcp_raw(self, q: np.ndarray) -> np.ndarray:
        flange = self.fkine_flange(q)[0]
        tcp = flange.copy()
        tcp[:3, 3] = (flange @ self.TCP_IN_LINK6_HOM)[:3]
        return tcp[np.newaxis]

    def fkine_all(self, q: np.ndarray) -> np.ndarray:
        q_full = self._expand_q(q)
        tfs = np.array(
            [self._robot.fkine(q_full, end=name).A for name in self.name_list]
        )
        for i, name in enumerate(self.name_list):
            tfs[i] = tfs[i] @ self.mesh_adjust[name]
        return tfs[np.newaxis]

    def gripper_scalar(self, q: np.ndarray) -> float:
        q = np.asarray(q, dtype=np.float64)
        if q.shape[0] >= 8:
            return float(q[6] + q[7])
        if q.shape[0] >= 7:
            return float(q[6])
        return 0.0
