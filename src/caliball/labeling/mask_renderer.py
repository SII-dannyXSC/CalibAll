"""
Mask 渲染与 BBox 计算（基于 NVDiffrast）。

MaskRenderer 封装 renderer / intrinsic / extrinsic，提供：
  - 机器人各 link mesh 的 mask 渲染
  - arm / gripper 区分
  - COCO RLE 编码
  - bbox 提取
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch


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


class MaskRenderer:
    """机器人 mask 渲染器，封装 renderer / intrinsic / extrinsic。

    Args:
        renderer:   NVDiffrastRenderer 实例（可为 None，此时 render_masks 返回空结果）
        intrinsic:  (3, 3) 相机内参矩阵
        extrinsic:  (4, 4) 世界 -> 相机外参矩阵
    """

    def __init__(self, renderer, intrinsic: np.ndarray, extrinsic: np.ndarray):
        self.renderer = renderer
        self.intrinsic = np.asarray(intrinsic, dtype=np.float64)
        self.extrinsic = np.asarray(extrinsic, dtype=np.float64)

    def render_masks(self, joint_angles, tf_model,
                     vertices_list, faces_list,
                     arm_mesh_num=None, device="cuda"):
        """
        使用 NVDiffrast 渲染机器人 mask，计算 bbox。

        Args:
            joint_angles:    关节角（与 label_grip_point_all_repr 相同约定）
            tf_model:        ArmGripperCompositeTF / ...
            vertices_list:   list of (N_i, 3) torch.Tensor（CUDA float）
            faces_list:      list of (M_i, 3) torch.Tensor（CUDA int）
            arm_mesh_num:    每臂前若干 mesh 计为臂体（余下计为夹爪）；None 时全部视为臂体
            device:          torch 设备

        Returns:
            list[MaskBboxResult]，长度等于 tf_model.n_arms，各元素对应一条臂
        """
        joints = np.asarray(joint_angles, dtype=np.float64)
        link_poses = tf_model.fkine_all(joints)  # (n_arms, n_links, 4, 4)
        link_poses_arr = np.asarray(link_poses).reshape(-1, 4, 4)  # (n_arms*n_links, 4, 4)
        link_poses_t = torch.tensor(link_poses_arr, dtype=torch.float32, device=device)

        extrinsic_t = torch.tensor(self.extrinsic, dtype=torch.float32, device=device)

        H, W = self.renderer.H, self.renderer.W
        final_depth = torch.full((H, W), 2.0, dtype=torch.float32, device=device)

        nlinks = len(vertices_list)
        n_arms = tf_model.n_arms
        n_links_per_arm = nlinks // n_arms

        if arm_mesh_num is None:
            arm_mesh_num = n_links_per_arm  # 全部视为臂，无夹爪分离

        arm_masks = [torch.zeros((H, W), dtype=torch.float32, device=device) for _ in range(n_arms)]
        grip_masks = [torch.zeros((H, W), dtype=torch.float32, device=device) for _ in range(n_arms)]

        white = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        K_t = torch.tensor(self.intrinsic, dtype=torch.float32, device=device)

        for link_idx in range(nlinks):
            a = link_idx // n_links_per_arm          # arm index
            local_idx = link_idx % n_links_per_arm   # index within this arm

            Tc_c2l = extrinsic_t @ link_poses_t[link_idx]
            verts = vertices_list[link_idx]
            faces = faces_list[link_idx]

            render_info = self.renderer.render_all(
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
