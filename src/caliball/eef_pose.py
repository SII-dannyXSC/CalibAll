"""末端执行器位姿容器。内部以旋转矩阵存储，支持单/双/多臂。"""

from __future__ import annotations

from typing import Union

import numpy as np
from scipy.spatial.transform import Rotation


class EEFPose:
    """单时刻多臂末端位姿。

    内部存储（float64 numpy）：
        pos:     (n_arms, 3)    位置
        rot_mat: (n_arms, 3, 3) 旋转矩阵
        gripper: (n_arms,)      夹爪标量

    构造：
        EEFPose.from_hom(hom, gripper)
            hom: (n_arms, 4, 4) 或 (4, 4)

        EEFPose.from_vec(vec, rot_type, gripper)
            vec: (n_arms, D) 或 (D,)

    旋转表示（rot_type），与 utils_3d.py 约定一致：
        "euler_xyz"       – XYZ 欧拉角（弧度），6 维
        "euler_zyx"       – ZYX 欧拉角（弧度），6 维
        "axis_angle"      – 旋转向量（轴角），6 维
        "quaternion"      – scipy 约定 [qx, qy, qz, qw]，7 维
        "rotation_matrix" – 展平的 3×3 矩阵（含位置共 12 维）
    """

    def __init__(
        self,
        pos: np.ndarray,
        rot: np.ndarray,
        gripper: Union[np.ndarray, float, None] = None,
    ):
        """
        Args:
            pos:     (n_arms, 3) 或 (3,)
            rot:     (n_arms, 3, 3) 或 (3, 3)
            gripper: (n_arms,) 或 标量（广播到所有臂）或 None（置 0）
        """
        pos = np.asarray(pos, dtype=np.float64)
        rot = np.asarray(rot, dtype=np.float64)
        if pos.ndim == 1:
            pos = pos[np.newaxis]
        if rot.ndim == 2:
            rot = rot[np.newaxis]
        n = pos.shape[0]
        assert rot.shape[0] == n, f"pos n_arms={n} 与 rot n_arms={rot.shape[0]} 不匹配"
        self._pos = pos
        self._rot = rot
        if gripper is None:
            self._gripper = np.zeros(n, dtype=np.float64)
        else:
            g = np.asarray(gripper, dtype=np.float64).ravel()
            if g.shape[0] == 1 and n > 1:
                g = np.broadcast_to(g, (n,)).copy()
            assert g.shape[0] == n, f"gripper shape {g.shape} 与 n_arms={n} 不匹配"
            self._gripper = g

    # ------------------------------------------------------------------
    # 构造方法
    # ------------------------------------------------------------------

    @classmethod
    def from_hom(
        cls,
        hom: np.ndarray,
        gripper: Union[np.ndarray, float, None] = None,
    ) -> "EEFPose":
        """从齐次矩阵构造。hom: (n_arms, 4, 4) 或 (4, 4)。"""
        hom = np.asarray(hom, dtype=np.float64)
        if hom.ndim == 2:
            hom = hom[np.newaxis]
        return cls(pos=hom[:, :3, 3], rot=hom[:, :3, :3], gripper=gripper)

    @classmethod
    def from_vec(
        cls,
        vec: np.ndarray,
        rot_type: str = "euler_xyz",
        gripper: Union[np.ndarray, float, None] = None,
    ) -> "EEFPose":
        """从位姿向量构造。vec: (n_arms, D) 或 (D,)。"""
        vec = np.asarray(vec, dtype=np.float64)
        if vec.ndim == 1:
            vec = vec[np.newaxis]
        pos = vec[:, :3]
        if rot_type == "euler_xyz":
            rot = Rotation.from_euler("xyz", vec[:, 3:6]).as_matrix()
        elif rot_type == "euler_zyx":
            rot = Rotation.from_euler("zyx", vec[:, 3:6]).as_matrix()
        elif rot_type in ("axis_angle", "axis_angle_residual", "axis-angle", "axis-angle-residual"):
            rot = Rotation.from_rotvec(vec[:, 3:6]).as_matrix()
        elif rot_type == "quaternion":
            rot = Rotation.from_quat(vec[:, 3:7]).as_matrix()  # scipy xyzw
        elif rot_type == "rotation_matrix":
            rot = vec[:, 3:12].reshape(-1, 3, 3)
        else:
            raise ValueError(f"Unknown rot_type: {rot_type!r}")
        return cls(pos=pos, rot=rot, gripper=gripper)

    # ------------------------------------------------------------------
    # 属性
    # ------------------------------------------------------------------

    @property
    def n_arms(self) -> int:
        return self._pos.shape[0]

    @property
    def pos(self) -> np.ndarray:
        """(n_arms, 3)"""
        return self._pos

    @property
    def rot_mat(self) -> np.ndarray:
        """(n_arms, 3, 3)"""
        return self._rot

    @property
    def gripper(self) -> np.ndarray:
        """(n_arms,)"""
        return self._gripper

    # ------------------------------------------------------------------
    # 旋转表示转换
    # ------------------------------------------------------------------

    def euler_xyz(self) -> np.ndarray:
        """(n_arms, 3) XYZ 欧拉角（弧度）。"""
        return Rotation.from_matrix(self._rot).as_euler("xyz")

    def euler_zyx(self) -> np.ndarray:
        """(n_arms, 3) ZYX 欧拉角（弧度）。"""
        return Rotation.from_matrix(self._rot).as_euler("zyx")

    def axis_angle(self) -> np.ndarray:
        """(n_arms, 3) 旋转向量（轴角）。"""
        return Rotation.from_matrix(self._rot).as_rotvec()

    def quat_xyzw(self) -> np.ndarray:
        """(n_arms, 4) 四元数，scipy 约定 [qx, qy, qz, qw]。"""
        return Rotation.from_matrix(self._rot).as_quat()

    def quat_wxyz(self) -> np.ndarray:
        """(n_arms, 4) 四元数，[qw, qx, qy, qz]。"""
        q = self.quat_xyzw()
        return np.concatenate([q[:, 3:], q[:, :3]], axis=1)

    def hom(self) -> np.ndarray:
        """(n_arms, 4, 4) 齐次矩阵。"""
        n = self.n_arms
        T = np.eye(4, dtype=np.float64)[np.newaxis].repeat(n, axis=0)
        T[:, :3, :3] = self._rot
        T[:, :3, 3] = self._pos
        return T

    # ------------------------------------------------------------------
    # 向量化输出
    # ------------------------------------------------------------------

    def as_vec(self, rot_type: str = "euler_xyz", with_gripper: bool = True) -> np.ndarray:
        """(n_arms, D[+1]) 位姿向量，with_gripper=True 时末尾追加夹爪标量。

        D（不含 gripper）：euler*/axis_angle → 6；quaternion → 7；rotation_matrix → 12
        """
        pos = self._pos
        if rot_type == "euler_xyz":
            rot_part = self.euler_xyz()
        elif rot_type == "euler_zyx":
            rot_part = self.euler_zyx()
        elif rot_type in ("axis_angle", "axis_angle_residual", "axis-angle", "axis-angle-residual"):
            rot_part = self.axis_angle()
        elif rot_type == "quaternion":
            rot_part = self.quat_xyzw()
        elif rot_type == "rotation_matrix":
            rot_part = self._rot.reshape(self.n_arms, 9)
        else:
            raise ValueError(f"Unknown rot_type: {rot_type!r}")
        parts = [pos, rot_part]
        if with_gripper:
            parts.append(self._gripper[:, np.newaxis])
        return np.concatenate(parts, axis=1)

    # ------------------------------------------------------------------
    # 空间变换
    # ------------------------------------------------------------------

    def transform(self, T: np.ndarray) -> "EEFPose":
        """左乘变换矩阵（如外参），返回新坐标系下的 EEFPose。

        Args:
            T: (4, 4)
        """
        T = np.asarray(T, dtype=np.float64)
        R, t = T[:3, :3], T[:3, 3]
        new_pos = (R @ self._pos.T).T + t   # (n_arms, 3)
        new_rot = R[np.newaxis] @ self._rot  # (n_arms, 3, 3)
        return EEFPose(pos=new_pos, rot=new_rot, gripper=self._gripper.copy())

    # ------------------------------------------------------------------
    # 索引 & 表示
    # ------------------------------------------------------------------

    def __getitem__(self, idx: int) -> "EEFPose":
        """选取第 idx 臂，返回 n_arms=1 的 EEFPose。"""
        return EEFPose(
            pos=self._pos[idx: idx + 1],
            rot=self._rot[idx: idx + 1],
            gripper=self._gripper[idx: idx + 1],
        )

    def __repr__(self) -> str:
        return (
            f"EEFPose(n_arms={self.n_arms}, "
            f"pos={np.round(self._pos, 4).tolist()}, "
            f"gripper={np.round(self._gripper, 4).tolist()})"
        )
