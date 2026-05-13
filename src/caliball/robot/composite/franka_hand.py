import numpy as np

from src.caliball.robot.arm_gripper_composite import ArmGripperCompositeTF
from src.caliball.robot.arm.fr3 import Fr3ArmTF
from src.caliball.robot.gripper.fr3_hand import Fr3PandaGripperTF


class FrankaPandaHandTF(ArmGripperCompositeTF):
    """FR3 臂（fr3.urdf）+ 原厂手爪（fr3_hand 子链）。

    q: (8,) = 7 臂关节 + 1 开合（复制到两指）。
    fkine_eef: fr3 法兰位姿。
    fkine_gripper: 法兰位姿 @ 全闭时 TCP。
    """

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
            arm_joint_num=7,
            gripper_closed_q=float(Fr3PandaGripperTF.FINGER_CLOSED),
            gripper_mount_yaw_deg=gripper_mount_yaw_deg,
        )
        self.arm = Fr3ArmTF(list(arm_names), arm_eef_name)
        self.gripper = Fr3PandaGripperTF(list(gripper_names), gripper_eef_name)
        self.grasp_point_R_align = self._build_grasp_point_R_align(grasp_point_rotation_align)

    def _mount_T_for_tcp(self, q_arm: np.ndarray) -> np.ndarray:
        return self.arm.fkine_eef(q_arm)  # (1, 4, 4)

    def _mount_T_for_gripper_meshes(
        self, arm_tfs: np.ndarray, q_arm: np.ndarray
    ) -> np.ndarray:
        return self.arm.fkine_eef(q_arm)  # (1, 4, 4)
