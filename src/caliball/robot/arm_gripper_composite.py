import numpy as np

from src.caliball.robot.base import BaseTF


class ArmGripperCompositeTF(BaseTF):
    """机械臂 TF + 末端子链 TF（夹爪、手爪等）复合基类。

    子类 ``__init__`` 须先调用::

        super().__init__(arm_joint_num=..., gripper_closed_q=..., gripper_mount_yaw_deg=...)

    再设置 ``self.arm``、``self.gripper``。

    子类须实现（单帧，q_arm: (arm_joints,)）：
      - _mount_T_for_tcp(q_arm)             -> (1, 4, 4)   供 fkine_gripper 使用
      - _mount_T_for_gripper_meshes(arm_tfs, q_arm) -> (1, 4, 4)   供 fkine_all 使用
        其中 arm_tfs: (1, n_arm_links, 4, 4)
    """

    def __init__(
        self,
        arm_joint_num: int,
        gripper_closed_q: float,
        gripper_mount_yaw_deg: float = 0.0,
    ):
        self.arm_joint_num = int(arm_joint_num)
        self.gripper_closed_q = float(gripper_closed_q)
        self.gripper_mount_yaw_deg = float(gripper_mount_yaw_deg)

    @staticmethod
    def _Rz_4x4(rad: float) -> np.ndarray:
        c, s = np.cos(rad), np.sin(rad)
        T = np.eye(4, dtype=np.float64)
        T[0, 0] = c
        T[0, 1] = -s
        T[1, 0] = s
        T[1, 1] = c
        return T

    def _gripper_mount_bias_T(self) -> np.ndarray:
        """(4,4) 绕 mount Z 的固定偏置；0° 为单位阵。"""
        return self._Rz_4x4(float(np.deg2rad(self.gripper_mount_yaw_deg)))

    def _mount_with_gripper_bias(self, T_mount: np.ndarray) -> np.ndarray:
        """T_mount: (1, 4, 4) → (1, 4, 4)，在末端与夹爪链之间插入绕 Z 的安装偏置。"""
        Tb = self._gripper_mount_bias_T()
        return np.asarray(T_mount, dtype=np.float64) @ Tb

    def fkine_eef(self, q: np.ndarray) -> np.ndarray:
        """臂末端 TCP（不含夹爪偏移），shape (1, 4, 4)。"""
        q = np.asarray(q, dtype=np.float64)
        return self.arm.fkine_eef(q[: self.arm_joint_num])

    def _fkine_gripper_raw(self, q: np.ndarray) -> np.ndarray:
        """全闭夹爪指尖位姿，shape (1, 4, 4)。"""
        q = np.asarray(q, dtype=np.float64)
        q_arm = q[: self.arm_joint_num]
        T_mount = self._mount_with_gripper_bias(self._mount_T_for_tcp(q_arm))  # (1, 4, 4)
        T_tcp = self.gripper.fkine_gripper(
            np.array([self.gripper_closed_q])
        )  # (1, 4, 4)
        return T_mount @ T_tcp  # (1, 4, 4)

    def fkine_all(self, q: np.ndarray) -> np.ndarray:
        """臂 + 夹爪所有 link，shape (1, n_arm+n_grip, 4, 4)。"""
        q = np.asarray(q, dtype=np.float64)
        q_arm = q[: self.arm_joint_num]
        q_grip = q[self.arm_joint_num : self.arm_joint_num + 1]  # (1,)
        arm_tfs = self.arm.fkine_all(q_arm)  # (1, n_arm_links, 4, 4)
        T_mount = self._mount_with_gripper_bias(
            self._mount_T_for_gripper_meshes(arm_tfs, q_arm)
        )  # (1, 4, 4)
        grip_loc = self.gripper.fkine_all(q_grip)  # (1, n_grip_links, 4, 4)
        grip_w = T_mount[:, np.newaxis] @ grip_loc  # (1, n_grip_links, 4, 4)
        return np.concatenate([arm_tfs, grip_w], axis=1)  # (1, n_total, 4, 4)

    def _mount_T_for_tcp(self, q_arm: np.ndarray) -> np.ndarray:
        """q_arm: (arm_joints,) -> (1, 4, 4)"""
        raise NotImplementedError

    def _mount_T_for_gripper_meshes(
        self, arm_tfs: np.ndarray, q_arm: np.ndarray
    ) -> np.ndarray:
        """arm_tfs: (1, n_arm_links, 4, 4), q_arm: (arm_joints,) -> (1, 4, 4)"""
        raise NotImplementedError
