import numpy as np

from src.caliball.robot.arm_gripper_composite import ArmGripperCompositeTF
from src.caliball.robot.gripper.robotiq import RobotiqTF
from src.caliball.robot.arm.ur5e import Ur5eTF


class Ur5eRobotiqTF(ArmGripperCompositeTF):
    """UR5e + Robotiq 2F-85。q: (7,) = 6 臂关节 + 1 夹爪开度 [0, 0.8]。

    fkine_eef: UR tool0 位姿（不含夹爪偏移）。
    fkine_gripper: 夹爪全闭时指尖在世界系下的位姿。
    """

    GRIPPER_CLOSED = 0.8
    MOUNT_LINK = "tool0"

    def __init__(
        self,
        arm_names,
        arm_eef_name,
        gripper_names,
        gripper_eef_name,
        *,
        gripper_mount_yaw_deg: float = 0.0,
        grasp_point_rotation_align=None,
    ):
        super().__init__(
            arm_joint_num=6,
            gripper_closed_q=self.GRIPPER_CLOSED,
            gripper_mount_yaw_deg=gripper_mount_yaw_deg,
        )
        self.arm = Ur5eTF(arm_names, arm_eef_name)
        self.gripper = RobotiqTF(gripper_names, gripper_eef_name)
        self.grasp_point_R_align = self._build_grasp_point_R_align(grasp_point_rotation_align)

    def _mount_T_world(self, q_arm: np.ndarray) -> np.ndarray:
        """世界系下 UR tool0 位姿，shape (1, 4, 4)。"""
        T = self.arm.robot.fkine(np.asarray(q_arm, dtype=np.float64), end=self.MOUNT_LINK).A
        return T[np.newaxis]  # (1, 4, 4)

    def _mount_T_for_tcp(self, q_arm: np.ndarray) -> np.ndarray:
        return self._mount_T_world(q_arm)

    def _mount_T_for_gripper_meshes(
        self, _arm_tfs: np.ndarray, q_arm: np.ndarray
    ) -> np.ndarray:
        return self._mount_T_world(q_arm)
