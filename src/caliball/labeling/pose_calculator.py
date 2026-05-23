"""
3D 位姿计算、2D 投影、旋转表示转换、帧间差分。

PoseCalculator 封装 intrinsic / extrinsic 矩阵，提供：
  - 世界坐标系 -> 相机坐标系变换
  - 相机坐标系 -> 像素坐标系投影
  - 所有旋转表示（euler / quat / rotation_matrix）
  - 帧间 delta 计算
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.spatial.transform import Rotation

from caliball.utils.transforms import hom_mat_to_pose, pose_to_hom_mat


@dataclass
class PoseAllRepr:
    """相机坐标系位姿的所有旋转表示（内部计算用）。"""
    pos: list         # [x, y, z, g]
    rot_euler: list   # [x, y, z, rx, ry, rz, g]
    rot_quat: list    # [x, y, z, qw, qx, qy, qz, g]
    rot_mat: list     # [x, y, z, r00..r22, g]
    hom_cam: np.ndarray  # (4, 4)


@dataclass
class PoseDelta:
    """两帧位姿差分（内部计算用）。"""
    pos_delta: list
    rot_euler_delta: list
    rot_quat_delta: list
    rot_mat_delta: list


class PoseCalculator:
    """3D 位姿计算器，封装 intrinsic / extrinsic 矩阵。

    Args:
        intrinsic:  (3, 3) 相机内参矩阵
        extrinsic:  (4, 4) 世界 -> 相机外参矩阵
    """

    def __init__(self, intrinsic: np.ndarray, extrinsic: np.ndarray):
        self.intrinsic = np.asarray(intrinsic, dtype=np.float64)
        self.extrinsic = np.asarray(extrinsic, dtype=np.float64)

    # ------------------------------------------------------------------
    # 基础 3D / 2D 变换
    # ------------------------------------------------------------------

    def label_3d_mat(self, eef_pose, rotation_type="euler_xyz"):
        mat = pose_to_hom_mat(eef_pose, rotation_type)
        mat = np.dot(self.extrinsic, mat)
        pose = hom_mat_to_pose(mat, rotation_type)
        return pose

    def label_3d_point(self, eef_pose):
        eef_pose = np.array(eef_pose)[:3]
        eef_pose = np.append(eef_pose, 1)
        point_3d = np.dot(self.extrinsic, eef_pose)
        return point_3d[:3]

    def label_2d_point(self, eef_pose):
        point_3d = self.label_3d_point(eef_pose)
        point_2d = np.dot(self.intrinsic, point_3d)
        point_2d = point_2d / point_2d[2]
        return int(point_2d[0]), int(point_2d[1])


    # ------------------------------------------------------------------
    # EEF / Grip Point 位姿计算（所有旋转表示）
    # ------------------------------------------------------------------

    def label_pose_all_repr(self, hom_world, gripper_state):
        """
        将世界坐标系下的 4x4 齐次矩阵变换到相机坐标系，返回所有旋转表示。

        Args:
            hom_world:    (4, 4) 世界坐标系下的齐次矩阵
            gripper_state: 标量，夹爪状态

        Returns:
            PoseAllRepr
        """
        hom_cam = self.extrinsic @ np.array(hom_world)
        pos_cam = hom_cam[:3, 3]
        rot_cam = hom_cam[:3, :3]

        euler_cam = Rotation.from_matrix(rot_cam).as_euler("xyz")
        quat_xyzw = Rotation.from_matrix(rot_cam).as_quat()  # [qx, qy, qz, qw]
        quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
        mat_flat = rot_cam.flatten()
        g = float(gripper_state)

        return PoseAllRepr(
            pos=np.concatenate([pos_cam, [g]]).tolist(),
            rot_euler=np.concatenate([pos_cam, euler_cam, [g]]).tolist(),
            rot_quat=np.concatenate([pos_cam, quat_wxyz, [g]]).tolist(),
            rot_mat=np.concatenate([pos_cam, mat_flat, [g]]).tolist(),
            hom_cam=hom_cam,
        )

    def label_eef_all_repr(self, eef_pose_world, gripper_state,
                           rotation_type="euler_xyz"):
        """
        将数据集中的 EEF 位姿（世界坐标系）变换到相机坐标系，返回所有旋转表示。

        Args:
            eef_pose_world: 长度 6+ 的数组 [x, y, z, r1, r2, r3, ...]
            gripper_state:  标量
            rotation_type:  输入旋转的表示方式（euler_xyz / axis_angle / quaternion 等）。
                ``axis_angle``：r1,r2,r3 为旋转向量（与 OXE 轴角残差约定一致时用 ``axis_angle_residual`` 别名）。
        """
        eef_pose = np.array(eef_pose_world[:6])
        hom_world = pose_to_hom_mat(eef_pose, rotation_type)
        return self.label_pose_all_repr(hom_world, gripper_state)

    def label_grip_point_all_repr(self, joint_angles, tf_model, gripper_state=None, arm_idx=0):
        """
        通过 FK 计算 grip point（TCP）位姿，变换到相机坐标系，返回所有旋转表示。

        Args:
            joint_angles:  关节角
            tf_model:      BaseTF 子类
            gripper_state: 若为 None，则由 tf_model.gripper_scalars 推断
            arm_idx:       臂索引（0=左/单臂，1=右）
        """
        joints = np.asarray(joint_angles, dtype=np.float64)
        hom_world = np.asarray(tf_model.fkine_tcp(joints)[arm_idx], dtype=np.float64)  # (4,4)
        if gripper_state is None:
            gripper_state = float(tf_model.gripper_scalars(joints)[arm_idx])
        g = float(gripper_state)
        return self.label_pose_all_repr(hom_world, g)

    # ------------------------------------------------------------------
    # Delta（帧间差分）计算
    # ------------------------------------------------------------------

    def label_pose_delta(self, curr_repr: PoseAllRepr, prev_repr: Optional[PoseAllRepr]) -> PoseDelta:
        """
        计算两帧位姿之间的差分（delta）。
        pos_delta = curr_pos - prev_pos（相机坐标系）
        rot_delta = curr_rot @ inv(prev_rot)
        """
        if prev_repr is None:
            g = float(curr_repr.pos[-1])
            return PoseDelta(
                pos_delta=[0.0, 0.0, 0.0, g],
                rot_euler_delta=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, g],
                rot_quat_delta=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, g],
                rot_mat_delta=[0.0, 0.0, 0.0,
                               1.0, 0.0, 0.0,
                               0.0, 1.0, 0.0,
                               0.0, 0.0, 1.0, g],
            )

        curr_hom = curr_repr.hom_cam
        prev_hom = prev_repr.hom_cam
        g = float(curr_repr.pos[-1])

        pos_delta = curr_hom[:3, 3] - prev_hom[:3, 3]
        rot_delta = curr_hom[:3, :3] @ np.linalg.inv(prev_hom[:3, :3])

        euler_delta = Rotation.from_matrix(rot_delta).as_euler("xyz")
        quat_xyzw = Rotation.from_matrix(rot_delta).as_quat()
        quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
        mat_flat = rot_delta.flatten()

        return PoseDelta(
            pos_delta=np.concatenate([pos_delta, [g]]).tolist(),
            rot_euler_delta=np.concatenate([pos_delta, euler_delta, [g]]).tolist(),
            rot_quat_delta=np.concatenate([pos_delta, quat_wxyz, [g]]).tolist(),
            rot_mat_delta=np.concatenate([pos_delta, mat_flat, [g]]).tolist(),
        )

    # ------------------------------------------------------------------
    # 2D 投影 & UVD
    # ------------------------------------------------------------------

    def label_2d_and_uvd(self, hom_cam):
        """
        从相机坐标系的齐次矩阵投影到像素坐标，并获取深度 d（相机 Z 轴）。

        Args:
            hom_cam:   (4, 4) 相机坐标系下的齐次矩阵，或 (3,) 位置向量

        Returns:
            uv:  [u, v]       像素坐标（整数）
            uvd: [u, v, d]    像素坐标 + 深度（米）
        """
        if np.array(hom_cam).shape == (4, 4):
            pt_cam = np.array(hom_cam)[:3, 3]
        else:
            pt_cam = np.array(hom_cam)[:3]

        px = self.intrinsic @ pt_cam
        if abs(px[2]) < 1e-8:
            px[2] = 1e-8
        u = float(px[0] / px[2])
        v = float(px[1] / px[2])
        d = float(pt_cam[2])

        return [round(u), round(v)], [round(u), round(v), d]
