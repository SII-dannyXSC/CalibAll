from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


class BaseTF(ABC):
    """机器人 TF 基类。单帧输入 q: (n_joints,)，输出含 n_arms 维。

    - 单臂: n_arms=1，输出 (1, 4, 4)
    - 双臂: n_arms=2，顺序 [左, 右]，输出 (2, 4, 4)
    """

    n_arms: int = 1
    grasp_point_R_align: Optional[np.ndarray] = None  # (3,3) or None; set by subclass __init__

    @staticmethod
    def _build_grasp_point_R_align(val) -> Optional[np.ndarray]:
        """
        将 grasp_point_rotation_align 配置值转为 (3,3) 旋转矩阵。

        val 可以是：
          - None / 未指定 → 返回 None（即不对齐）
          - [rx, ry, rz]（度，euler_xyz）→ 转换为矩阵
          - [[r00,r01,r02], [r10,r11,r12], [r20,r21,r22]]（3×3 嵌套列表）→ 直接转换
        """
        if val is None:
            return None
        from scipy.spatial.transform import Rotation
        arr = np.array(val, dtype=np.float64)
        if arr.shape == (3,):
            return Rotation.from_euler("xyz", arr, degrees=True).as_matrix()
        if arr.shape == (3, 3):
            return arr
        raise ValueError(f"grasp_point_rotation_align 形状应为 (3,) 或 (3,3)，实际为 {arr.shape}")

    @abstractmethod
    def fkine_eef(self, q: np.ndarray) -> np.ndarray:
        """臂末端位姿（不含夹爪偏移）。

        参数:
            q: (n_joints,)
        返回:
            (n_arms, 4, 4)
        """

    @abstractmethod
    def _fkine_gripper_raw(self, q: np.ndarray) -> np.ndarray:
        """fkine_gripper 原始实现（不含旋转对齐）。子类实现此方法。

        参数:
            q: (n_joints,)
        返回:
            (n_arms, 4, 4)
        """

    def fkine_gripper(self, q: np.ndarray) -> np.ndarray:
        """夹爪闭合时末端位姿；无夹爪时与 fkine_eef 相同。
        自动应用 grasp_point_R_align（若已设置）。

        参数:
            q: (n_joints,)
        返回:
            (n_arms, 4, 4)
        """
        result = self._fkine_gripper_raw(q)
        if self.grasp_point_R_align is not None:
            result = result.copy()
            result[:, :3, :3] = result[:, :3, :3] @ self.grasp_point_R_align
        return result

    @abstractmethod
    def fkine_all(self, q: np.ndarray) -> np.ndarray:
        """所有 link 变换矩阵。

        参数:
            q: (n_joints,)
        返回:
            (n_arms, n_links, 4, 4)
        """

    def fkine(self, q: np.ndarray) -> np.ndarray:
        """fkine_gripper 的别名。"""
        return self.fkine_gripper(q)

    def gripper_scalar(self, q: np.ndarray) -> float:
        """从关节向量中提取夹爪开合标量（默认取末位关节）。"""
        return float(np.asarray(q, dtype=np.float64)[-1])

    def gripper_scalars(self, q: np.ndarray) -> np.ndarray:
        """每臂夹爪标量，shape (n_arms,)。"""
        return np.array([self.gripper_scalar(q)], dtype=np.float64)