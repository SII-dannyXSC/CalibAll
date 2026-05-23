"""FR3 arm + FR3 panda hand (original factory gripper) composite."""

from __future__ import annotations

import numpy as np
from roboticstoolbox.robot.Robot import Robot

from caliball.robots._base import BaseTF
from caliball.robots._composite import ArmGripperCompositeTF
from caliball.robots._registry import register_robot
from caliball.robots._rtb import RoboticsToolBoxTF


# ---------------------------------------------------------------------------
# URDF loaders (inlined)
# ---------------------------------------------------------------------------

class _Fr3Arm(Robot):
    """FR3 arm (no hand), fr3.urdf, 7 revolute joints."""

    def __init__(self):
        links, name, urdf_string, urdf_filepath = self.URDF_read(
            file_path="robots/fr3/fr3.urdf",
            tld="./third_party/urdf/franka_description",
        )
        super().__init__(
            links,
            name=name,
            manufacturer="Franka",
            urdf_string=urdf_string,
            urdf_filepath=urdf_filepath,
        )


class _Fr3Hand(Robot):
    """fr3_hand.urdf (7 arm + 2 finger). Sub-chain FK in fr3_link8 flange frame."""

    def __init__(self):
        links, name, urdf_string, urdf_filepath = self.URDF_read(
            file_path="robots/fr3/fr3_hand.urdf",
            tld="./third_party/urdf/franka_description",
        )
        super().__init__(
            links,
            name=name,
            manufacturer="Franka",
            urdf_string=urdf_string,
            urdf_filepath=urdf_filepath,
        )


# ---------------------------------------------------------------------------
# FR3 Arm TF (internal, used as sub-component)
# ---------------------------------------------------------------------------

class _Fr3ArmTF(RoboticsToolBoxTF):
    """FR3 arm (pure arm). q: (7,), fkine_tcp = fkine_flange."""

    ARM_NAMES = [
        "fr3_link0", "fr3_link1", "fr3_link2", "fr3_link3",
        "fr3_link4", "fr3_link5", "fr3_link6", "fr3_link7",
    ]
    ARM_TCP_NAME = "fr3_link8"

    def __init__(self, name_list=None, tcp_name=None, **kwargs):
        super().__init__(name_list or self.ARM_NAMES, tcp_name or self.ARM_TCP_NAME)
        self._robot = _Fr3Arm()
        self._init_urdf()

    @property
    def robot(self):
        return self._robot


# ---------------------------------------------------------------------------
# FR3 Panda Gripper TF (internal, gripper sub-chain)
# ---------------------------------------------------------------------------

class _Fr3PandaGripperTF(BaseTF):
    """FR3 factory hand sub-chain (fr3_hand.urdf). Poses in fr3_link8 (flange) frame.

    q: (1,) single scalar [0, FINGER_MAX], duplicated to two prismatic fingers.
    fkine_flange: eef link pose in flange frame.
    fkine_tcp: same as fkine_flange (fingertip = eef).
    """

    ARM_DOF = 7
    FLANGE_LINK = "fr3_link8"
    FINGER_CLOSED = 0.04

    GRIPPER_NAMES = ["fr3_hand", "fr3_leftfinger", "fr3_rightfinger"]
    GRIPPER_TCP_NAME = "fr3_hand_tcp"

    def __init__(self, name_list=None, tcp_name=None, **kwargs):
        self.name_list = list(name_list or self.GRIPPER_NAMES)
        self.tcp_name = tcp_name or self.GRIPPER_TCP_NAME
        self._robot = _Fr3Hand()

        self.mesh_adjust = BaseTF.build_mesh_adjust(self._robot, self.name_list)

        q0 = np.zeros(9)
        T_fl = self._robot.fkine(q0, end=self.FLANGE_LINK).A
        self._T_fl_inv_zero = np.linalg.inv(T_fl)

    def _q_full(self, g: float) -> np.ndarray:
        return np.concatenate([np.zeros(self.ARM_DOF), [g, g]])

    def fkine_flange(self, q: np.ndarray) -> np.ndarray:
        """eef link pose in flange frame, shape (1, 4, 4)."""
        q = np.asarray(q, dtype=np.float64)
        g = float(q[0])
        qf = self._q_full(g)
        T_w = self._robot.fkine(qf, end=self.tcp_name).A
        return (self._T_fl_inv_zero @ T_w)[np.newaxis]  # (1, 4, 4)

    def _fkine_tcp_raw(self, q: np.ndarray) -> np.ndarray:
        """Same as fkine_flange, shape (1, 4, 4)."""
        return self.fkine_flange(q)

    def fkine_all(self, q: np.ndarray) -> np.ndarray:
        """Gripper links in flange frame, shape (1, n_links, 4, 4)."""
        q = np.asarray(q, dtype=np.float64)
        g = float(q[0])
        qf = self._q_full(g)
        row = []
        for name in self.name_list:
            T_w = self._robot.fkine(qf, end=name).A
            T_rel = self._T_fl_inv_zero @ T_w
            row.append(T_rel @ self.mesh_adjust[name])
        return np.array(row)[np.newaxis]  # (1, n_links, 4, 4)


# ---------------------------------------------------------------------------
# Composite: FR3 arm + panda hand
# ---------------------------------------------------------------------------

@register_robot("franka_panda_hand")
class FrankaPandaHandTF(ArmGripperCompositeTF):
    """FR3 arm (fr3.urdf) + factory hand (fr3_hand sub-chain).

    q: (8,) = 7 arm joints + 1 finger open/close (duplicated to both fingers).
    fkine_flange: fr3 flange pose.
    fkine_tcp: flange pose @ closed TCP.
    """

    ARM_NAMES = _Fr3ArmTF.ARM_NAMES
    ARM_TCP_NAME = _Fr3ArmTF.ARM_TCP_NAME
    GRIPPER_NAMES = _Fr3PandaGripperTF.GRIPPER_NAMES
    GRIPPER_TCP_NAME = _Fr3PandaGripperTF.GRIPPER_TCP_NAME

    LINK_NAMES = [
        # arm (fr3)
        "fr3_link0", "fr3_link1", "fr3_link2", "fr3_link3",
        "fr3_link4", "fr3_link5", "fr3_link6", "fr3_link7",
        # gripper (panda hand)
        "fr3_hand", "fr3_leftfinger", "fr3_rightfinger",
    ]
    TCP_NAME = "fr3_hand_tcp"
    MESH_PATHS = [
        # arm (fr3)
        "third_party/urdf/franka_description/meshes/visual/link0.dae",
        "third_party/urdf/franka_description/meshes/visual/link1.dae",
        "third_party/urdf/franka_description/meshes/visual/link2.dae",
        "third_party/urdf/franka_description/meshes/visual/link3.dae",
        "third_party/urdf/franka_description/meshes/visual/link4.dae",
        "third_party/urdf/franka_description/meshes/visual/link5.dae",
        "third_party/urdf/franka_description/meshes/visual/link6.dae",
        "third_party/urdf/franka_description/meshes/visual/link7.dae",
        # gripper (panda hand)
        "third_party/urdf/franka_description/meshes/visual/hand.dae",
        "third_party/urdf/franka_description/meshes/visual/finger.dae",
        "third_party/urdf/franka_description/meshes/visual/finger.dae",
    ]

    def __init__(
        self,
        arm_names=None,
        arm_tcp_name=None,
        gripper_names=None,
        gripper_tcp_name=None,
        *,
        gripper_mount_yaw_deg: float = 0.0,
        grasp_point_rotation_align=None,
        **kwargs,
    ):
        super().__init__(
            arm_joint_num=7,
            gripper_closed_q=float(_Fr3PandaGripperTF.FINGER_CLOSED),
            gripper_mount_yaw_deg=gripper_mount_yaw_deg,
        )
        arm_names = arm_names or self.ARM_NAMES
        arm_tcp_name = arm_tcp_name or self.ARM_TCP_NAME
        gripper_names = gripper_names or self.GRIPPER_NAMES
        gripper_tcp_name = gripper_tcp_name or self.GRIPPER_TCP_NAME
        self.arm = _Fr3ArmTF(list(arm_names), arm_tcp_name)
        self.gripper = _Fr3PandaGripperTF(list(gripper_names), gripper_tcp_name)
        self.grasp_point_R_align = self._build_grasp_point_R_align(
            grasp_point_rotation_align
        )

    def _mount_T_for_tcp(self, q_arm: np.ndarray) -> np.ndarray:
        return self.arm.fkine_flange(q_arm)  # (1, 4, 4)

    def _mount_T_for_gripper_meshes(
        self, arm_tfs: np.ndarray, q_arm: np.ndarray
    ) -> np.ndarray:
        return self.arm.fkine_flange(q_arm)  # (1, 4, 4)
