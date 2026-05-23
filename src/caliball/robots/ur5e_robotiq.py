"""UR5e + Robotiq 2F-85 composite."""

from __future__ import annotations

import numpy as np
from roboticstoolbox.robot.Robot import Robot

from caliball.robots._base import BaseTF
from caliball.robots._composite import ArmGripperCompositeTF
from caliball.robots._registry import register_robot
from caliball.robots._rtb import RoboticsToolBoxTF
from caliball.robots.ur5e import Ur5eTF


# ---------------------------------------------------------------------------
# URDF loader for Robotiq 2F-85 (inlined from robot/urdf/robotiq_85.py)
# ---------------------------------------------------------------------------

class _Robotiq85(Robot):
    def __init__(self):
        links, name, urdf_string, urdf_filepath = self.URDF_read(
            file_path="robotiq_2f_85_gripper_visualization/urdf/robotiq_arg2f_85_model.urdf",
            tld="./third_party/urdf/robotiq",
        )
        super().__init__(
            links,
            name=name,
            manufacturer="Robotiq",
            urdf_string=urdf_string,
            urdf_filepath=urdf_filepath,
        )


# ---------------------------------------------------------------------------
# Robotiq gripper sub-chain TF (shared with franka_robotiq)
# ---------------------------------------------------------------------------

class _RobotiqGripperTF(RoboticsToolBoxTF):
    """Robotiq 2F-85 gripper sub-chain. q: (1,) scalar [0, 0.8].

    fkine_flange: eef link origin pose (in flange frame).
    fkine_tcp: with fingertip offset TCP pose (in flange frame).
    """

    MIMIC_MULTIPLIERS = np.array([1.0, -1.0, 1.0, 1.0, -1.0, 1.0])
    TIP_OFFSET = np.array([0.0, -0.0139, 0.0445, 1.0])

    GRIPPER_NAMES = [
        "robotiq_arg2f_base_link",
        "left_outer_knuckle", "left_outer_finger",
        "left_inner_finger", "left_inner_knuckle",
        "right_outer_knuckle", "right_outer_finger",
        "right_inner_finger", "right_inner_knuckle",
    ]
    GRIPPER_TCP_NAME = "left_inner_finger"

    def __init__(self, name_list=None, tcp_name=None, **kwargs):
        super().__init__(name_list or self.GRIPPER_NAMES, tcp_name or self.GRIPPER_TCP_NAME)
        self._robot = _Robotiq85()
        self._init_urdf()

    @property
    def robot(self):
        return self._robot

    def _expand_q(self, q: np.ndarray) -> np.ndarray:
        """q: (1,) scalar -> (6,) expanded URDF joint angles."""
        return float(q[0]) * self.MIMIC_MULTIPLIERS

    def fkine_flange(self, q: np.ndarray) -> np.ndarray:
        """eef link pose (no fingertip offset), shape (1, 4, 4)."""
        q = np.asarray(q, dtype=np.float64)
        q6 = self._expand_q(q)
        eef_pose = self.robot.fkine(q6, end=self.tcp_name).A
        return eef_pose[np.newaxis]  # (1, 4, 4)

    def _fkine_tcp_raw(self, q: np.ndarray) -> np.ndarray:
        """With fingertip offset TCP pose, shape (1, 4, 4)."""
        q = np.asarray(q, dtype=np.float64)
        T_eef = self.fkine_flange(q)  # (1, 4, 4)
        T_tip = T_eef.copy()
        T_tip[0, :3, 3] = (T_eef[0] @ self.TIP_OFFSET)[:3]
        return T_tip  # (1, 4, 4)

    def fkine_all(self, q: np.ndarray) -> np.ndarray:
        """All link transforms (with mesh alignment), shape (1, n_links, 4, 4)."""
        q = np.asarray(q, dtype=np.float64)
        q6 = self._expand_q(q)
        all_tf = [self.robot.fkine(q6, end=name).A for name in self.name_list]
        result = np.array(all_tf)[np.newaxis]  # (1, n_links, 4, 4)
        for idx, name in enumerate(self.name_list):
            if name in self.mesh_adjust:
                result[0, idx] = result[0, idx] @ self.mesh_adjust[name]
        return result


# ---------------------------------------------------------------------------
# Composite: UR5e + Robotiq
# ---------------------------------------------------------------------------

@register_robot("ur5e_robotiq")
class Ur5eRobotiqTF(ArmGripperCompositeTF):
    """UR5e + Robotiq 2F-85. q: (7,) = 6 arm joints + 1 gripper [0, 0.8].

    fkine_flange: UR tool0 pose (no gripper offset).
    fkine_tcp: closed-gripper fingertip pose in world frame.
    """

    GRIPPER_CLOSED = 0.8
    MOUNT_LINK = "tool0"

    ARM_NAMES = Ur5eTF.ARM_NAMES
    ARM_TCP_NAME = Ur5eTF.ARM_TCP_NAME
    GRIPPER_NAMES = _RobotiqGripperTF.GRIPPER_NAMES
    GRIPPER_TCP_NAME = _RobotiqGripperTF.GRIPPER_TCP_NAME

    LINK_NAMES = [
        # arm (ur5e)
        "base_link", "shoulder_link", "upper_arm_link", "forearm_link",
        "wrist_1_link", "wrist_2_link", "wrist_3_link",
        # gripper (robotiq 2f-85)
        "robotiq_arg2f_base_link",
        "left_outer_knuckle", "left_outer_finger",
        "left_inner_finger", "left_inner_knuckle",
        "right_outer_knuckle", "right_outer_finger",
        "right_inner_finger", "right_inner_knuckle",
    ]
    TCP_NAME = "left_inner_finger"
    MESH_PATHS = [
        # arm (ur5e)
        "third_party/urdf/universal_robots/ur_description/meshes/ur5e/visual/base.dae",
        "third_party/urdf/universal_robots/ur_description/meshes/ur5e/visual/shoulder.dae",
        "third_party/urdf/universal_robots/ur_description/meshes/ur5e/visual/upperarm.dae",
        "third_party/urdf/universal_robots/ur_description/meshes/ur5e/visual/forearm.dae",
        "third_party/urdf/universal_robots/ur_description/meshes/ur5e/visual/wrist1.dae",
        "third_party/urdf/universal_robots/ur_description/meshes/ur5e/visual/wrist2.dae",
        "third_party/urdf/universal_robots/ur_description/meshes/ur5e/visual/wrist3.dae",
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
            arm_joint_num=6,
            gripper_closed_q=self.GRIPPER_CLOSED,
            gripper_mount_yaw_deg=gripper_mount_yaw_deg,
        )
        arm_names = arm_names or self.ARM_NAMES
        arm_tcp_name = arm_tcp_name or self.ARM_TCP_NAME
        gripper_names = gripper_names or self.GRIPPER_NAMES
        gripper_tcp_name = gripper_tcp_name or self.GRIPPER_TCP_NAME
        self.arm = Ur5eTF(arm_names, arm_tcp_name)
        self.gripper = _RobotiqGripperTF(gripper_names, gripper_tcp_name)
        self.grasp_point_R_align = self._build_grasp_point_R_align(
            grasp_point_rotation_align
        )

    def _mount_T_world(self, q_arm: np.ndarray) -> np.ndarray:
        """World-frame UR tool0 pose, shape (1, 4, 4)."""
        T = self.arm.robot.fkine(
            np.asarray(q_arm, dtype=np.float64), end=self.MOUNT_LINK
        ).A
        return T[np.newaxis]  # (1, 4, 4)

    def _mount_T_for_tcp(self, q_arm: np.ndarray) -> np.ndarray:
        return self._mount_T_world(q_arm)

    def _mount_T_for_gripper_meshes(
        self, _arm_tfs: np.ndarray, q_arm: np.ndarray
    ) -> np.ndarray:
        return self._mount_T_world(q_arm)
