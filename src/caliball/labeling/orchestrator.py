"""
标注编排器：单帧标注、episode 标注。

LabelOrchestrator 组合 PoseCalculator + MaskRenderer，提供高层标注接口。
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from caliball.labeling.label_data import ArmLabel, FrameLabel, LabelData
from caliball.labeling.mask_renderer import MaskBboxResult, MaskRenderer
from caliball.labeling.pose_calculator import PoseCalculator


class LabelOrchestrator:
    """标注编排器。

    Args:
        pose_calculator:  PoseCalculator 实例
        mask_renderer:    MaskRenderer 实例
    """

    def __init__(
        self,
        pose_calculator: PoseCalculator,
        mask_renderer: MaskRenderer,
    ):
        self.pose_calculator = pose_calculator
        self.mask_renderer = mask_renderer

    # ------------------------------------------------------------------
    # 单帧标注
    # ------------------------------------------------------------------

    def label_frame(self, frame_idx, joint_angles,
                    tf_model, vertices_list, faces_list,
                    device="cuda", arm_mesh_num=None,
                    skip_mask=False, arm_names=None):
        """
        对单帧进行全量标注。

        Args:
            frame_idx:     帧索引
            joint_angles:  关节角
            tf_model:      BaseTF 子类
            vertices_list: mesh 顶点列表（CUDA）
            faces_list:    mesh 面列表（CUDA）
            device:        torch 设备
            arm_mesh_num:  臂 link mesh 数；None 时按 tf_model 推断
            skip_mask:     是否跳过 mask 渲染
            arm_names:     臂名列表，None 时自动生成

        Returns:
            FrameLabel
        """
        joints = np.array(joint_angles, dtype=np.float64)
        n_arms = tf_model.n_arms

        if arm_names is None:
            arm_names = (["left", "right"] if n_arms == 2 else
                         ["left"] if n_arms == 1 else
                         [f"arm{i}" for i in range(n_arms)])

        gripper_states = list(tf_model.gripper_scalars(joints))  # (n_arms,)

        # Mask 渲染（一次性，返回 per-arm 列表）
        if skip_mask:
            mask_results = [MaskBboxResult(
                mask_all=None, mask_arm=None, mask_gripper=None,
                bbox_all=None, bbox_arm=None, bbox_gripper=None,
            ) for _ in range(n_arms)]
        else:
            mask_results = self.mask_renderer.render_masks(
                joints, tf_model, vertices_list, faces_list,
                arm_mesh_num=arm_mesh_num, device=device,
            )

        # 逐臂计算 grip point
        arms_dict = {}
        for a, arm_name in enumerate(arm_names):
            grip_repr = self.pose_calculator.label_grip_point_all_repr(
                joints, tf_model, gripper_state=gripper_states[a], arm_idx=a
            )
            grip_uv, grip_uvd = self.pose_calculator.label_2d_and_uvd(grip_repr.hom_cam)
            mr = mask_results[a]
            arms_dict[arm_name] = ArmLabel(
                uv=grip_uv,
                xyz_euler_g=grip_repr.rot_euler,
                xyz_quat_g=grip_repr.rot_quat,
                xyz_mat_g=grip_repr.rot_mat,
                uvd=grip_uvd,
                mask_with_gripper=mr.mask_all,
                mask_without_gripper=mr.mask_arm,
                mask_gripper=mr.mask_gripper,
                bbox_with_gripper=mr.bbox_all,
                bbox_without_gripper=mr.bbox_arm,
                bbox_gripper=mr.bbox_gripper,
            )

        return FrameLabel(index=int(frame_idx), arms=arms_dict)

    # ------------------------------------------------------------------
    # 整个 episode 标注
    # ------------------------------------------------------------------

    def label_episode(self, joint_angles_list,
                      tf_model, vertices_list, faces_list,
                      device="cuda", arm_mesh_num=None,
                      skip_mask=False,
                      dataset_name: str = "", episode_id: str = "",
                      camera_name: str = "", arm_names=None) -> LabelData:
        """
        对整个 episode 的所有帧进行标注。

        Args:
            joint_angles_list: (T, n_joints) 关节角数组
            tf_model:          ArmGripperCompositeTF / ...
            vertices_list:     mesh 顶点列表（CUDA）
            faces_list:        mesh 面列表（CUDA）
            device:            torch 设备
            arm_mesh_num:      臂 mesh 数；None 时按 tf_model 推断
            skip_mask:         是否跳过 mask 渲染
            dataset_name:      数据集名
            episode_id:        episode 标识
            camera_name:       相机名
            arm_names:         臂名列表；None 时自动推断

        Returns:
            LabelData
        """
        n_arms = tf_model.n_arms
        if arm_names is None:
            arm_names = (["left", "right"] if n_arms == 2 else
                         ["left"] if n_arms == 1 else
                         [f"arm{i}" for i in range(n_arms)])

        label_data = LabelData(
            dataset_name=dataset_name,
            episode_id=str(episode_id),
            arm_names=arm_names,
        )

        for i, joints in enumerate(joint_angles_list):
            frame_label = self.label_frame(
                frame_idx=i,
                joint_angles=joints,
                tf_model=tf_model,
                vertices_list=vertices_list,
                faces_list=faces_list,
                device=device,
                arm_mesh_num=arm_mesh_num,
                skip_mask=skip_mask,
                arm_names=arm_names,
            )
            label_data.add_frame(camera_name, frame_label)

        return label_data
