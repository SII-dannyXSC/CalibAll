"""Franka + Robotiq 2F-85 composite."""

from __future__ import annotations

import numpy as np

from caliball.robots._composite import ArmGripperCompositeTF
from caliball.robots._registry import register_robot
from caliball.robots.franka import FrankaTF
from caliball.robots.ur5e_robotiq import _RobotiqGripperTF


@register_robot("franka_robotiq")
class FrankaRobotiqTF(ArmGripperCompositeTF):
    """Franka + Robotiq 2F-85. q: (8,) = 7 arm joints + 1 gripper [0, 0.8].

    fkine_flange: panda_link8 pose.
    fkine_tcp: closed-gripper fingertip pose in world frame.
    fkine_all: arm 8 links + gripper links (skip meshless hand link).
    """

    GRIPPER_CLOSED = 0.8

    GRIPPER_NAMES = _RobotiqGripperTF.GRIPPER_NAMES
    GRIPPER_TCP_NAME = _RobotiqGripperTF.GRIPPER_TCP_NAME

    LINK_NAMES = [
        # arm (franka panda)
        "panda_link1", "panda_link2", "panda_link3", "panda_link4",
        "panda_link5", "panda_link6", "panda_link7", "panda_link8",
        # gripper (robotiq 2f-85)
        "robotiq_arg2f_base_link",
        "left_outer_knuckle", "left_outer_finger",
        "left_inner_finger", "left_inner_knuckle",
        "right_outer_knuckle", "right_outer_finger",
        "right_inner_finger", "right_inner_knuckle",
    ]
    TCP_NAME = "panda_link8"
    MESH_PATHS = [
        # arm (franka panda)
        "third_party/urdf/franka_description/meshes/visual/link0.dae",
        "third_party/urdf/franka_description/meshes/visual/link1.dae",
        "third_party/urdf/franka_description/meshes/visual/link2.dae",
        "third_party/urdf/franka_description/meshes/visual/link3.dae",
        "third_party/urdf/franka_description/meshes/visual/link4.dae",
        "third_party/urdf/franka_description/meshes/visual/link5.dae",
        "third_party/urdf/franka_description/meshes/visual/link6.dae",
        "third_party/urdf/franka_description/meshes/visual/link7.dae",
        # gripper (robotiq 2f-85)
        "third_party/urdf/robotiq/robotiq_2f_85_gripper_visualization/meshes/visual/robotiq_arg2f_85_base_link.dae",
        "third_party/urdf/robotiq/robotiq_2f_85_gripper_visualization/meshes/visual/robotiq_arg2f_85_outer_knuckle.dae",
        "third_party/urdf/robotiq/robotiq_2f_85_gripper_visualization/meshes/visual/robotiq_arg2f_85_outer_finger.dae",
        "third_party/urdf/robotiq/robotiq_2f_85_gripper_visualization/meshes/visual/robotiq_arg2f_85_inner_finger.dae",
        "third_party/urdf/robotiq/robotiq_2f_85_gripper_visualization/meshes/visual/robotiq_arg2f_85_inner_knuckle.dae",
        "third_party/urdf/robotiq/robotiq_2f_85_gripper_visualization/meshes/visual/robotiq_arg2f_85_outer_knuckle.dae",
        "third_party/urdf/robotiq/robotiq_2f_85_gripper_visualization/meshes/visual/robotiq_arg2f_85_outer_finger.dae",
        "third_party/urdf/robotiq/robotiq_2f_85_gripper_visualization/meshes/visual/robotiq_arg2f_85_inner_finger.dae",
        "third_party/urdf/robotiq/robotiq_2f_85_gripper_visualization/meshes/visual/robotiq_arg2f_85_inner_knuckle.dae",
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
            gripper_closed_q=self.GRIPPER_CLOSED,
            gripper_mount_yaw_deg=gripper_mount_yaw_deg,
        )
        self.arm = FrankaTF()
        # Backward compat: old-style FrankaRobotiqTF(names, gripper_tcp_name)
        if gripper_names is None and arm_names is not None:
            gripper_names = arm_names
            gripper_tcp_name = arm_tcp_name
        gripper_names = gripper_names or self.GRIPPER_NAMES
        gripper_tcp_name = gripper_tcp_name or self.GRIPPER_TCP_NAME
        self.gripper = _RobotiqGripperTF(gripper_names, gripper_tcp_name)
        self.grasp_point_R_align = self._build_grasp_point_R_align(
            grasp_point_rotation_align
        )

    def _mount_T_for_tcp(self, q_arm: np.ndarray) -> np.ndarray:
        return self.arm.fkine_flange(q_arm)  # (1, 4, 4)

    def _mount_T_for_gripper_meshes(
        self, arm_tfs: np.ndarray, q_arm: np.ndarray
    ) -> np.ndarray:
        return arm_tfs[:, -1:][:, 0:1].reshape(1, 4, 4)  # last arm link -> (1, 4, 4)

    def fkine_all(self, q: np.ndarray) -> np.ndarray:
        """Arm mesh + gripper mesh (skip panda_hand), shape (1, n_arm+n_grip, 4, 4)."""
        full = super().fkine_all(q)  # (1, n_total, 4, 4)
        # full layout: [0..7 arm links], [8 hand], [9.. gripper]
        arm_mesh = full[:, : self.arm_joint_num + 1]   # (1, 8, 4, 4)
        grip_mesh = full[:, self.arm_joint_num + 2 :]  # (1, n_grip, 4, 4)
        return np.concatenate([arm_mesh, grip_mesh], axis=1)
