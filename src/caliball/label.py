"""
机器人轨迹标注模块：EEF/Grip Point 位姿、2D 投影、Mask/BBox 渲染等。
支持 Franka、UR5e、ALOHA、ARX5，以及 Franka+Robotiq、UR5e+Robotiq、FR3+Panda 手等。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from src.caliball.utils.utils_3d import diff_rot_mat, hom_mat_to_pose, pose_to_hom_mat
from src.caliball.pipeline.label_data import ArmLabel, FrameLabel, LabelData
from src.caliball.eef_pose import EEFPose


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


@dataclass
class MaskBboxResult:
    """mask 渲染与 bbox 结果。"""
    mask_all: Optional[dict]
    mask_arm: Optional[dict]
    mask_gripper: Optional[dict]
    bbox_all: Optional[list]
    bbox_arm: Optional[list]
    bbox_gripper: Optional[list]
    depth_map: Optional[np.ndarray] = None


class Labeler:
    def __init__(self, config=None):
        self.config = config

    def label_3d_mat(self, eef_pose, extrinsic_matrix, rotation_type="euler_xyz"):
        mat = pose_to_hom_mat(eef_pose, rotation_type)
        mat = np.dot(extrinsic_matrix, mat)
        pose = hom_mat_to_pose(mat, rotation_type)
        return pose

    def label_3d_point(self, eef_pose, extrinsic_matrix):
        eef_pose = np.array(eef_pose)[:3]
        eef_pose = np.append(eef_pose, 1)
        extrinsic_matrix = np.array(extrinsic_matrix)
        point_3d = np.dot(extrinsic_matrix, eef_pose)
        return point_3d[:3]

    def label_2d_point(self, eef_pose, intrinsic_matrix, extrinsic_matrix):
        point_3d = self.label_3d_point(eef_pose, extrinsic_matrix)
        point_2d = np.dot(intrinsic_matrix, point_3d)
        point_2d = point_2d / point_2d[2]
        return int(point_2d[0]), int(point_2d[1])

    def label_orientation(self, eef_pose_list, extrinsic_matrix):
        eef_pose_list = np.array(eef_pose_list)[..., :3]

        eef_pose_cam_first = self.label_3d_point(eef_pose_list[0], extrinsic_matrix)
        eef_pose_cam_last = self.label_3d_point(eef_pose_list[-1], extrinsic_matrix)

        delta_eef_pose = eef_pose_cam_last - eef_pose_cam_first
        if_right = delta_eef_pose[0] > 0
        if_down = delta_eef_pose[1] > 0
        if_forward = delta_eef_pose[2] > 0

        return (if_right, if_down, if_forward)

    def segment_video(self, eef_pose_list, extrinsic_matrix):
        # 支持 list[EEFPose]：提取第 0 臂位置 + gripper
        if (eef_pose_list is not None
                and len(eef_pose_list) > 0
                and isinstance(eef_pose_list[0], EEFPose)):
            positions = np.stack([p.pos[0] for p in eef_pose_list])   # (T, 3)
            gripper = np.stack([p.gripper[0] for p in eef_pose_list]) # (T,)
            eef_pose_arr = np.concatenate([positions, gripper[:, np.newaxis]], axis=1)  # (T, 4)
        else:
            eef_pose_arr = np.array(eef_pose_list)
        gripper = None
        if eef_pose_arr.ndim == 2:
            gripper_index = getattr(self.config, "segment_gripper_index", None) if self.config else None
            if gripper_index is None and eef_pose_arr.shape[1] > 6:
                gripper_index = -1
            if gripper_index is not None:
                gripper = eef_pose_arr[:, gripper_index].astype(np.float64)

        eef_pose_list = eef_pose_arr[..., :3]
        eef_pose_cam_list = [self.label_3d_point(eef_pose, extrinsic_matrix) for eef_pose in eef_pose_list]
        eef_pose_cam_list = np.array(eef_pose_cam_list)
        num_frames = len(eef_pose_cam_list)
        if num_frames <= 2:
            return eef_pose_cam_list, [0, max(num_frames - 1, 0)]

        velocities = np.diff(eef_pose_cam_list, axis=0)
        speed = np.linalg.norm(velocities, axis=1)

        min_speed = float(getattr(self.config, "segment_min_speed", 1e-4)) if self.config else 1e-4
        angle_threshold_deg = float(getattr(self.config, "segment_angle_threshold_deg", 60.0)) if self.config else 60.0
        axis_flip_threshold = int(getattr(self.config, "segment_axis_flip_threshold", 2)) if self.config else 2
        min_segment_gap = int(getattr(self.config, "segment_min_gap", 10)) if self.config else 10
        gripper_threshold = float(getattr(self.config, "segment_gripper_threshold", 0.2)) if self.config else 0.2

        dirs = np.zeros_like(velocities, dtype=np.float64)
        valid = speed > min_speed
        dirs[valid] = velocities[valid] / speed[valid, None]

        gripper_open = None
        if gripper is not None and len(gripper) >= 2:
            gripper_open = gripper > gripper_threshold

        segment_points = [0]
        last_segment_idx = 0
        gripper_close_indices = []

        for i in range(1, len(velocities)):
            boundary_idx = i + 1

            gripper_flip = False
            if gripper_open is not None and i < len(gripper_open):
                if gripper_open[i] != gripper_open[i - 1]:
                    gripper_flip = True
                    if gripper_open[i - 1] and not gripper_open[i]:
                        gripper_close_indices.append(i)

            motion_change = False
            if valid[i - 1] and valid[i]:
                dot = float(np.clip(np.dot(dirs[i - 1], dirs[i]), -1.0, 1.0))
                angle_deg = np.degrees(np.arccos(dot))

                prev_sign = np.sign(velocities[i - 1])
                curr_sign = np.sign(velocities[i])
                axis_flip_count = int(np.sum((prev_sign != 0) & (curr_sign != 0) & (prev_sign != curr_sign)))

                motion_change = (angle_deg >= angle_threshold_deg) or (axis_flip_count >= axis_flip_threshold)

            is_sharp_change = motion_change or gripper_flip
            if is_sharp_change and (boundary_idx - last_segment_idx >= min_segment_gap):
                segment_points.append(boundary_idx)
                last_segment_idx = boundary_idx

        if segment_points[-1] != num_frames - 1:
            segment_points.append(num_frames - 1)

        return segment_points, gripper_close_indices

    def label_diff(self, eef_end, eef_start, rotation_type="euler_xyz", target_is_degree=True):
        pos_diff = None
        ori_diff = None
        if rotation_type == "euler_xyz":
            mat_0 = pose_to_hom_mat(eef_end, rotation_type)
            mat_1 = pose_to_hom_mat(eef_start, rotation_type)
            pos_diff = eef_end[:3] - eef_start[:3]
            ori_diff_mat = diff_rot_mat(mat_0[:3, :3], mat_1[:3, :3])
            ori_diff = Rotation.from_matrix(ori_diff_mat).as_euler("XYZ", degrees=target_is_degree)
        elif rotation_type in (
            "axis_angle",
            "axis_angle_residual",
            "axis-angle",
            "axis-angle-residual",
        ):
            mat_0 = pose_to_hom_mat(eef_end, rotation_type)
            mat_1 = pose_to_hom_mat(eef_start, rotation_type)
            pos_diff = eef_end[:3] - eef_start[:3]
            ori_diff_mat = diff_rot_mat(mat_0[:3, :3], mat_1[:3, :3])
            ori_diff = Rotation.from_matrix(ori_diff_mat).as_rotvec()
        elif rotation_type == "euler_zyx":
            pass
        elif rotation_type == "quaternion":
            pass
        elif rotation_type == "rotation_matrix":
            pass
        else:
            raise ValueError(f"Invalid rotation type: {rotation_type}")
        return pos_diff, ori_diff

    def label_diff_to_text(self, eef_end, eef_start, rotation_type="euler_xyz"):
        pos_diff, ori_diff = self.label_diff(eef_end, eef_start, rotation_type="euler_xyz")
        pos_diff = pos_diff * 100
        label_x_template = "move {} for {} cm"
        label_y_template = "move {} for {} cm"
        label_z_template = "move {} for {} cm"
        label_pitch_template = "pitch {} by {} degrees"
        label_yaw_template = "yaw {} by {} degrees"
        label_roll_template = "roll {} by {} degrees"

        label_x = ""
        label_y = ""
        label_z = ""
        label_pitch = ""
        label_yaw = ""
        label_roll = ""
        if pos_diff[0] > 0:
            label_x = label_x_template.format("right", abs(pos_diff[0]))
        else:
            label_x = label_x_template.format("left", abs(pos_diff[0]))
        if pos_diff[1] > 0:
            label_y = label_y_template.format("down", abs(pos_diff[1]))
        else:
            label_y = label_y_template.format("up", abs(pos_diff[1]))
        if pos_diff[2] > 0:
            label_z = label_z_template.format("forward", abs(pos_diff[2]))
        else:
            label_z = label_z_template.format("backward", abs(pos_diff[2]))
        if ori_diff[0] > 0:
            label_pitch = label_pitch_template.format("up", abs(ori_diff[0]))
        else:
            label_pitch = label_pitch_template.format("down", abs(ori_diff[0]))
        if ori_diff[1] > 0:
            label_yaw = label_yaw_template.format("right", abs(ori_diff[1]))
        else:
            label_yaw = label_yaw_template.format("left", abs(ori_diff[1]))
        if ori_diff[2] > 0:
            label_roll = label_roll_template.format("right", abs(ori_diff[2]))
        else:
            label_roll = label_roll_template.format("left", abs(ori_diff[2]))
        return f"{label_x}\n{label_y}\n{label_z}\n{label_pitch}\n{label_yaw}\n{label_roll}"

    # ------------------------------------------------------------------
    # EEF / Grip Point 位姿计算（所有旋转表示）
    # ------------------------------------------------------------------

    def label_pose_all_repr(self, hom_world, extrinsic, gripper_state):
        """
        将世界坐标系下的 4x4 齐次矩阵变换到相机坐标系，返回所有旋转表示。

        Args:
            hom_world:    (4, 4) 世界坐标系下的齐次矩阵
            extrinsic:    (4, 4) 世界→相机外参矩阵
            gripper_state: 标量，夹爪状态

        Returns:
            PoseAllRepr
        """
        hom_cam = np.array(extrinsic) @ np.array(hom_world)
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

    def label_eef_all_repr(self, eef_pose_world, extrinsic, gripper_state,
                           rotation_type="euler_xyz"):
        """
        将数据集中的 EEF 位姿（世界坐标系）变换到相机坐标系，返回所有旋转表示。

        Args:
            eef_pose_world: 长度 6+ 的数组 [x, y, z, r1, r2, r3, ...]
            extrinsic:      (4, 4) 外参矩阵
            gripper_state:  标量
            rotation_type:  输入旋转的表示方式（euler_xyz / axis_angle / quaternion 等）。
                ``axis_angle``：r1,r2,r3 为旋转向量（与 OXE 轴角残差约定一致时用 ``axis_angle_residual`` 别名）。
        """
        eef_pose = np.array(eef_pose_world[:6])
        hom_world = pose_to_hom_mat(eef_pose, rotation_type)
        return self.label_pose_all_repr(hom_world, extrinsic, gripper_state)

    def label_grip_point_all_repr(self, joint_angles, tf_model, extrinsic, gripper_state=None, arm_idx=0):
        """
        通过 FK 计算 grip point（TCP）位姿，变换到相机坐标系，返回所有旋转表示。

        Args:
            joint_angles:  关节角
            tf_model:      BaseTF 子类
            extrinsic:     (4, 4) 外参矩阵（世界→相机）
            gripper_state: 若为 None，则由 tf_model.gripper_scalars 推断
            arm_idx:       臂索引（0=左/单臂，1=右）
        """
        joints = np.asarray(joint_angles, dtype=np.float64)
        hom_world = np.asarray(tf_model.fkine_gripper(joints)[arm_idx], dtype=np.float64)  # (4,4)
        if gripper_state is None:
            gripper_state = float(tf_model.gripper_scalars(joints)[arm_idx])
        g = float(gripper_state)
        return self.label_pose_all_repr(hom_world, extrinsic, g)

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

    def label_2d_and_uvd(self, hom_cam, intrinsic):
        """
        从相机坐标系的齐次矩阵投影到像素坐标，并获取深度 d（相机 Z 轴）。

        Args:
            hom_cam:   (4, 4) 相机坐标系下的齐次矩阵，或 (3,) 位置向量
            intrinsic: (3, 3) 相机内参矩阵

        Returns:
            uv:  [u, v]       像素坐标（整数）
            uvd: [u, v, d]    像素坐标 + 深度（米）
        """
        if np.array(hom_cam).shape == (4, 4):
            pt_cam = np.array(hom_cam)[:3, 3]
        else:
            pt_cam = np.array(hom_cam)[:3]

        K = np.array(intrinsic)
        px = K @ pt_cam
        if abs(px[2]) < 1e-8:
            px[2] = 1e-8
        u = float(px[0] / px[2])
        v = float(px[1] / px[2])
        d = float(pt_cam[2])

        return [round(u), round(v)], [round(u), round(v), d]

    # ------------------------------------------------------------------
    # Mask & BBox 渲染（基于 NVDiffrast）
    # ------------------------------------------------------------------

    def label_mask_and_bbox(self, joint_angles, tf_model, renderer,
                            vertices_list, faces_list,
                            intrinsic, extrinsic,
                            arm_mesh_num=None, device="cuda"):
        """
        使用 NVDiffrast 渲染机器人 mask，计算 bbox。

        Args:
            joint_angles:    关节角（与 label_grip_point_all_repr 相同约定）
            tf_model:        ArmGripperCompositeTF / …
            renderer:        NVDiffrastRenderer 实例
            vertices_list:   list of (N_i, 3) torch.Tensor（CUDA float）
            faces_list:      list of (M_i, 3) torch.Tensor（CUDA int）
            intrinsic:       (3, 3) numpy 内参矩阵
            extrinsic:       (4, 4) numpy 外参矩阵（世界→相机）
            arm_mesh_num:    每臂前若干 mesh 计为臂体（余下计为夹爪）；None 时全部视为臂体
            device:          torch 设备

        Returns:
            list[MaskBboxResult]，长度等于 tf_model.n_arms，各元素对应一条臂
        """
        joints = np.asarray(joint_angles, dtype=np.float64)
        link_poses = tf_model.fkine_all(joints)  # (n_arms, n_links, 4, 4)
        link_poses_arr = np.asarray(link_poses).reshape(-1, 4, 4)  # (n_arms*n_links, 4, 4)
        link_poses_t = torch.tensor(link_poses_arr, dtype=torch.float32, device=device)

        extrinsic_t = torch.tensor(extrinsic, dtype=torch.float32, device=device)

        H, W = renderer.H, renderer.W
        final_depth = torch.full((H, W), 2.0, dtype=torch.float32, device=device)

        nlinks = len(vertices_list)
        n_arms = tf_model.n_arms
        n_links_per_arm = nlinks // n_arms

        if arm_mesh_num is None:
            arm_mesh_num = n_links_per_arm  # 全部视为臂，无夹爪分离

        arm_masks = [torch.zeros((H, W), dtype=torch.float32, device=device) for _ in range(n_arms)]
        grip_masks = [torch.zeros((H, W), dtype=torch.float32, device=device) for _ in range(n_arms)]

        white = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        K_t = torch.tensor(intrinsic, dtype=torch.float32, device=device)

        for link_idx in range(nlinks):
            a = link_idx // n_links_per_arm          # arm index
            local_idx = link_idx % n_links_per_arm   # index within this arm

            Tc_c2l = extrinsic_t @ link_poses_t[link_idx]
            verts = vertices_list[link_idx]
            faces = faces_list[link_idx]

            render_info = renderer.render_all(
                verts, faces, K=K_t, object_pose=Tc_c2l, mask_color=white
            )

            cur_depth = render_info["depth"]
            cur_mask = render_info["mask"]
            valid = (cur_depth > 0) & (cur_depth < final_depth)

            final_depth[valid] = cur_depth[valid]

            if local_idx < arm_mesh_num:
                arm_masks[a][valid] = cur_mask[valid]
            else:
                grip_masks[a][valid] = cur_mask[valid]

        final_depth[final_depth > 1] = 0.0
        depth_np = final_depth.cpu().numpy()

        results = []
        for a in range(n_arms):
            arm_np  = (arm_masks[a].cpu().numpy()  > 0.5).astype(np.uint8)
            grip_np = (grip_masks[a].cpu().numpy() > 0.5).astype(np.uint8)
            full_np = np.clip(arm_np + grip_np, 0, 1).astype(np.uint8)
            results.append(MaskBboxResult(
                mask_all=self._encode_mask_rle(full_np),
                mask_arm=self._encode_mask_rle(arm_np),
                mask_gripper=self._encode_mask_rle(grip_np),
                bbox_all=self._mask_to_bbox(full_np),
                bbox_arm=self._mask_to_bbox(arm_np),
                bbox_gripper=self._mask_to_bbox(grip_np),
                depth_map=depth_np if a == 0 else None,
            ))
        return results

    def _mask_to_bbox(self, mask):
        """返回 [x1, y1, x2, y2]，无前景时返回 None。"""
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            return None
        return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]

    def _encode_mask_rle(self, mask):
        """将二值 mask 编码为 COCO RLE 格式（JSON 可序列化）。"""
        try:
            from pycocotools import mask as coco_mask
            mask_f = np.asfortranarray(mask.astype(np.uint8))
            rle = coco_mask.encode(mask_f)
            if isinstance(rle["counts"], bytes):
                rle["counts"] = rle["counts"].decode("utf-8")
            return rle
        except ImportError:
            flat = mask.flatten().tolist()
            runs = []
            if flat:
                val, count = flat[0], 1
                for x in flat[1:]:
                    if x == val:
                        count += 1
                    else:
                        runs.append([int(val), count])
                        val, count = x, 1
                runs.append([int(val), count])
            return {"size": list(mask.shape), "counts": runs, "format": "simple_rle"}

    # ------------------------------------------------------------------
    # 单帧标注
    # ------------------------------------------------------------------

    def label_frame(self, frame_idx, joint_angles,
                    intrinsic, extrinsic,
                    tf_model, renderer, vertices_list, faces_list,
                    device="cuda", arm_mesh_num=None,
                    skip_mask=False, arm_names=None):
        """
        对单帧进行全量标注。

        Args:
            frame_idx:     帧索引
            joint_angles:  关节角
            intrinsic:     (3, 3) 内参矩阵
            extrinsic:     (4, 4) 外参矩阵
            tf_model:      BaseTF 子类
            renderer:      NVDiffrastRenderer 实例
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
            mask_results = self.label_mask_and_bbox(
                joints, tf_model, renderer, vertices_list, faces_list,
                intrinsic, extrinsic, arm_mesh_num=arm_mesh_num, device=device,
            )

        # 逐臂计算 grip point
        arms_dict = {}
        for a, arm_name in enumerate(arm_names):
            grip_repr = self.label_grip_point_all_repr(
                joints, tf_model, extrinsic, gripper_state=gripper_states[a], arm_idx=a
            )
            grip_uv, grip_uvd = self.label_2d_and_uvd(grip_repr.hom_cam, intrinsic)
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
                      intrinsic, extrinsic,
                      tf_model, renderer, vertices_list, faces_list,
                      device="cuda", arm_mesh_num=None,
                      skip_mask=False,
                      dataset_name: str = "", episode_id: str = "",
                      camera_name: str = "", arm_names=None) -> LabelData:
        """
        对整个 episode 的所有帧进行标注。

        Args:
            joint_angles_list: (T, n_joints) 关节角数组
            intrinsic:         (3, 3) 内参矩阵
            extrinsic:         (4, 4) 外参矩阵
            tf_model:          ArmGripperCompositeTF / …
            renderer:          NVDiffrastRenderer 实例
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
                intrinsic=intrinsic,
                extrinsic=extrinsic,
                tf_model=tf_model,
                renderer=renderer,
                vertices_list=vertices_list,
                faces_list=faces_list,
                device=device,
                arm_mesh_num=arm_mesh_num,
                skip_mask=skip_mask,
                arm_names=arm_names,
            )
            label_data.add_frame(camera_name, frame_label)

        return label_data

