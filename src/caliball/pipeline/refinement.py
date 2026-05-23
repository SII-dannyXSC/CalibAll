"""Rendering-based extrinsic refinement.

Dependency-injection version: robot, mask extractor, and mesh paths
are constructed externally and passed in.
"""

import torch
import cv2
import numpy as np
import os
from PIL import Image

from caliball.utils.image import save_imgs, save_img
from caliball.algorithms.rendering_optimizer import RBSolver
from caliball.utils.image import add_mask


class Refinement:
    def __init__(self, robot, mask_extractor, mesh_paths, device="cuda"):
        self.robot_tf = robot
        self.sam3_extractor = mask_extractor
        self.mesh_paths = mesh_paths
        self.device = device

    def update_robot(self, robot, mesh_paths):
        """Switch the FK model and mesh paths (does not reload SAM3)."""
        self.robot_tf = robot
        self.mesh_paths = mesh_paths

    def _to_float_tensor(self, value):
        if isinstance(value, torch.Tensor):
            return value.detach().clone().to(device=self.device, dtype=torch.float32)
        return torch.as_tensor(value, dtype=torch.float32, device=self.device)

    def _prepare_link_poses(self, joint_angles):
        q = np.asarray(joint_angles).squeeze()  # (1, n_joints) -> (n_joints,)
        link_poses_list = self.robot_tf.fkine_all(q)
        # dual-arm: (2, n_links, 4, 4) -> merge to (1, n_left+n_right, 4, 4)
        if link_poses_list.shape[0] == 2 and link_poses_list.ndim == 4:
            link_poses_list = np.concatenate(
                [link_poses_list[0], link_poses_list[1]], axis=0
            )[np.newaxis]
        return torch.as_tensor(link_poses_list, dtype=torch.float32, device=self.device)

    def _prepare_mask_tensor(self, mask):
        if mask is None:
            raise ValueError("mask cannot be None")

        mask_tensor = self._to_float_tensor(mask)
        if mask_tensor.ndim == 2:
            mask_tensor = mask_tensor.unsqueeze(0)
        if mask_tensor.ndim == 4 and mask_tensor.shape[1] == 1:
            mask_tensor = mask_tensor[:, 0]
        if mask_tensor.ndim != 3:
            raise ValueError(f"mask must be (H, W) or (B, H, W), got shape={tuple(mask_tensor.shape)}")
        return (mask_tensor > 0).float()

    def render(self, video, joint_angles, intrinsic, extrinsic):
        H, W = video.shape[1:3]
        solver = RBSolver(self.mesh_paths, H, W, extrinsic, device=self.device)
        solver.to(self.device)

        link_poses_list = self._prepare_link_poses(joint_angles[:1])

        dps = {
            "global_step": 0,
            "mask": torch.zeros((H, W), device=self.device, dtype=torch.float32).unsqueeze(0),
            "link_poses": link_poses_list,
            "K": self._to_float_tensor(intrinsic).unsqueeze(0),
        }

        result, loss_dict = solver.forward(dps)

        cur_rgb = video[0][:, :, ::-1]
        render_pre = result["rendered_masks"][0].detach().cpu()
        overlay = cur_rgb.copy()
        mask_render = render_pre.numpy().astype(np.uint8)
        mask_render = (mask_render > 0).astype(np.uint8)
        overlay = add_mask(cur_rgb, mask_render)

        return result, loss_dict

    def refine(self, video, joint_angles, intrinsic, extrinsic, base_path, mask=None, max_steps=3000, mask_id=0, progress_callback=None, stop_check=None):
        """
        Supports single or multi-frame masks:
          - mask_id: int or list[int]
          - mask: np.ndarray (H,W) or list[np.ndarray] (one per mask_id)
        """
        H, W = video.shape[1:3]

        # Normalise to list format
        if isinstance(mask_id, int):
            mask_ids = [mask_id]
        else:
            mask_ids = list(mask_id)

        if mask is None:
            masks = [self.sam3_extractor.extract_masks(video[mid]) for mid in mask_ids]
        elif isinstance(mask, list):
            masks = mask
        else:
            masks = [mask]

        if len(masks) != len(mask_ids):
            raise ValueError(f"mask count ({len(masks)}) != mask_id count ({len(mask_ids)})")

        print(f"[refine] H={H}, W={W}, mask_ids={mask_ids}, n_masks={len(masks)}")
        extrinsic = self._to_float_tensor(extrinsic)
        print("[refine] creating RBSolver ...")
        solver = RBSolver(self.mesh_paths, H, W, extrinsic, device=self.device)
        solver.to(self.device)
        print("[refine] RBSolver ready")

        # Multi-frame link_poses: fkine_all per frame then concatenate (B, n_links, 4, 4)
        link_poses_parts = [self._prepare_link_poses(joint_angles[mid:mid+1]) for mid in mask_ids]
        link_poses_list = torch.cat(link_poses_parts, dim=0)
        print(f"[refine] link_poses ready, shape={link_poses_list.shape}")

        # Multi-frame mask: (B, H, W)
        mask_tensors = [self._prepare_mask_tensor(m) for m in masks]
        mask_tensor = torch.cat(mask_tensors, dim=0)  # (B, H, W)
        print(f"[refine] mask_tensor shape={mask_tensor.shape}, starting iteration")

        dps = {
            "global_step": 0,
            "mask": mask_tensor,
            "link_poses": link_poses_list,
            "K": self._to_float_tensor(intrinsic).unsqueeze(0),
        }

        pose_optimizer = torch.optim.Adam(
            solver.parameters(),
            lr=5e-4,
            weight_decay=1e-6,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            pose_optimizer, T_0=2000, T_mult=2, eta_min=1e-5,
        )

        # Best checkpoint tracking
        best_loss = float('inf')
        best_dof = solver.dof.detach().clone()

        os.makedirs(base_path, exist_ok=True)
        for k in range(max_steps):
            if k == 0:
                print("[refine] first forward ...")
            output, loss_dict = solver.forward(dps)
            if k == 0:
                print("[refine] first forward done")

            # Improved loss: MSE + IoU
            rendered = output["rendered_masks"]  # (B, H, W)
            gt = dps["mask"]  # (B, H, W)
            mse_loss = ((rendered - gt) ** 2).mean()
            intersection = (rendered * gt).sum(dim=(1, 2))
            union = rendered.sum(dim=(1, 2)) + gt.sum(dim=(1, 2)) - intersection
            iou_loss = (1 - intersection / (union + 1e-6)).mean()
            loss = mse_loss + 0.5 * iou_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(solver.parameters(), max_norm=1.0)
            pose_optimizer.step()
            pose_optimizer.zero_grad(set_to_none=True)
            scheduler.step(k)

            # Update best checkpoint
            loss_val = loss.item()
            if loss_val < best_loss:
                best_loss = loss_val
                best_dof = solver.dof.detach().clone()

            # Check for user-requested stop
            if stop_check is not None and stop_check():
                print(f"[refine] user stop requested (step={k}, best_loss={best_loss:.6f})")
                break

            if k % 100 == 0:
                lr_now = pose_optimizer.param_groups[0]["lr"]
                print(f"{k}  loss={loss_val:.6f}  mse={mse_loss.item():.6f}  iou_l={iou_loss.item():.6f}  lr={lr_now:.2e}  best={best_loss:.6f}")
                tsfm = output["tsfm"]
                loss_cpu = loss.detach().cpu()

                save_path = os.path.join(base_path, f"{k}")
                pred_mask_path = os.path.join(base_path, f"pred_mask")
                os.makedirs(save_path, exist_ok=True)
                os.makedirs(pred_mask_path, exist_ok=True)
                with open(os.path.join(save_path, 'loss.txt'), 'w') as f:
                    f.write(f"loss={loss_val:.6f} mse={mse_loss.item():.6f} iou={iou_loss.item():.6f} best={best_loss:.6f}")
                with open(os.path.join(save_path, 'tsfm.txt'), 'w') as f:
                    f.write(str(tsfm))
                with open(os.path.join(save_path, 'intrinsic.txt'), 'w') as f:
                    f.write(str(intrinsic))

                # Generate overlay for each mask frame
                overlays_rgb = []
                for bi, mid in enumerate(mask_ids):
                    cur_rgb = video[mid][:, :, ::-1]
                    if bi == 0:
                        img = Image.fromarray(video[mid])
                        img.save(os.path.join(base_path, f'origin.png'))

                    render_pre = output["rendered_masks"][bi].detach().cpu()
                    render_gt = mask_tensor.detach().cpu()[bi].float()

                    save_img(render_gt, path=os.path.join(save_path, f'mask_{bi}_gt.png'))
                    save_img(render_pre, path=os.path.join(save_path, f'mask_{bi}_pred.png'))
                    if bi == 0:
                        save_img(render_gt, path=os.path.join(base_path, f'mask_gt.png'))

                    color = [0, 255, 0]
                    mask_render = render_pre.numpy().astype(np.uint8)
                    mask_render = (mask_render > 0).astype(np.uint8)
                    overlay = add_mask(cur_rgb, mask_render, color=color, alpha=0.5)
                    cv2.imwrite(os.path.join(save_path, f'output_with_mask_{bi}.png'), overlay)
                    if bi == 0:
                        cv2.imwrite(os.path.join(pred_mask_path, f'{k}.png'), overlay)
                    overlays_rgb.append(overlay[:, :, ::-1])  # BGR -> RGB

                if progress_callback is not None:
                    progress_callback(step=k, max_steps=max_steps, loss=float(loss_val),
                                      overlay=overlays_rgb[0],
                                      overlays=overlays_rgb, mask_ids=mask_ids)

        # Restore best checkpoint
        with torch.no_grad():
            solver.dof.copy_(best_dof)
            solver.history_ops.zero_()
        print(f"[refine] restored best checkpoint (loss={best_loss:.6f})")

        # Final forward with best parameters
        with torch.no_grad():
            output, loss_dict = solver.forward(dps)
        return output, loss_dict
