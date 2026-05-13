import numpy as np

from src.caliball.robot.arm_gripper_composite import ArmGripperCompositeTF
from src.caliball.robot.arm.franka import FrankaTF
from src.caliball.robot.gripper.robotiq import RobotiqTF


class FrankaRobotiqTF(ArmGripperCompositeTF):
    """Franka + Robotiq 2F-85。q: (8,) = 7 臂关节 + 1 夹爪开度 [0, 0.8]。

    fkine_eef: panda_link8 位姿。
    fkine_gripper: 夹爪全闭时指尖在世界系下的位姿。
    fkine_all: 臂 8 个 link + 夹爪 link（跳过无 mesh 的 hand link）。
    """

    GRIPPER_CLOSED = 0.8
    ARM_NAME_PREFIX_LEN = 8

    def __init__(
        self,
        arm_names,
        arm_eef_name,
        gripper_names,
        gripper_eef_name: str,
        *,
        gripper_mount_yaw_deg: float = 0.0,
        grasp_point_rotation_align=None,
    ):
        super().__init__(
            arm_joint_num=7,
            gripper_closed_q=self.GRIPPER_CLOSED,
            gripper_mount_yaw_deg=gripper_mount_yaw_deg,
        )
        self.arm = FrankaTF()
        self.gripper = RobotiqTF(gripper_names, gripper_eef_name)
        self.grasp_point_R_align = self._build_grasp_point_R_align(grasp_point_rotation_align)

    def _mount_T_for_tcp(self, q_arm: np.ndarray) -> np.ndarray:
        return self.arm.fkine_eef(q_arm)  # (1, 4, 4)

    def _mount_T_for_gripper_meshes(
        self, arm_tfs: np.ndarray, q_arm: np.ndarray
    ) -> np.ndarray:
        return arm_tfs[:, -1:][:, 0:1].reshape(1, 4, 4)  # last arm link -> (1, 4, 4)

    def fkine_all(self, q: np.ndarray) -> np.ndarray:
        """臂 mesh + 夹爪 mesh（跳过 panda_hand），shape (1, n_arm+n_grip, 4, 4)。"""
        full = super().fkine_all(q)  # (1, n_total, 4, 4)
        # full 布局: [0..7 臂 link], [8 hand], [9.. 夹爪]
        arm_mesh = full[:, : self.arm_joint_num + 1]  # (1, 8, 4, 4)
        grip_mesh = full[:, self.arm_joint_num + 2 :]   # (1, n_grip, 4, 4)
        return np.concatenate([arm_mesh, grip_mesh], axis=1)
