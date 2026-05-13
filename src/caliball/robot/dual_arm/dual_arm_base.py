import numpy as np

from src.caliball.robot.base import BaseTF


class DualArmTF(BaseTF):
    """通用双臂合成基类。持有左右两个独立单臂 TF，各自处理自己的关节和 link。

    q: (n_left_joints + n_right_joints,)，前半左臂，后半右臂。
    fkine_eef:     (2, 4, 4)
    fkine_gripper: (2, 4, 4)
    fkine_all:     (2, n_links, 4, 4)  左右臂 link 数须相同
    """

    n_arms = 2

    def __init__(self, left: BaseTF, right: BaseTF, n_left_joints: int):
        self.left = left
        self.right = right
        self.n_left_joints = n_left_joints

    def fkine_eef(self, q: np.ndarray) -> np.ndarray:
        q = np.asarray(q, dtype=np.float64)
        ql, qr = q[: self.n_left_joints], q[self.n_left_joints :]
        return np.stack([self.left.fkine_eef(ql)[0], self.right.fkine_eef(qr)[0]])  # (2, 4, 4)

    def _fkine_gripper_raw(self, q: np.ndarray) -> np.ndarray:
        q = np.asarray(q, dtype=np.float64)
        ql, qr = q[: self.n_left_joints], q[self.n_left_joints :]
        return np.stack([self.left.fkine_gripper(ql)[0], self.right.fkine_gripper(qr)[0]])  # (2, 4, 4)

    def fkine_all(self, q: np.ndarray) -> np.ndarray:
        q = np.asarray(q, dtype=np.float64)
        ql, qr = q[: self.n_left_joints], q[self.n_left_joints :]
        tfl = self.left.fkine_all(ql)   # (1, n_links, 4, 4)
        tfr = self.right.fkine_all(qr)  # (1, n_links, 4, 4)
        return np.stack([tfl[0], tfr[0]])  # (2, n_links, 4, 4)

    def gripper_scalar(self, q: np.ndarray) -> float:
        """取左臂夹爪标量（左臂最后一个关节）。"""
        return float(np.asarray(q, dtype=np.float64)[self.n_left_joints - 1])

    def gripper_scalars(self, q: np.ndarray) -> np.ndarray:
        """返回 [左臂夹爪, 右臂夹爪]，shape (2,)。"""
        q = np.asarray(q, dtype=np.float64)
        ql = q[: self.n_left_joints]
        qr = q[self.n_left_joints :]
        return np.array([self.left.gripper_scalar(ql), self.right.gripper_scalar(qr)], dtype=np.float64)
