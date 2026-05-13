import torch
import cv2
import numpy as np
import os
import time 
from PIL import Image

from src.caliball.utils.image import save_imgs, save_img
from src.caliball.pipeline.rendering_optimizer import RBSolver
from src.caliball.robot import build_robot
from src.caliball.config import build_robot_config
from src.caliball.utils.image import add_mask
from src.caliball.utils.sam3_extractor import Sam3Extractor

class Refinement:
    def __init__(self, config):
        self.config = config
        self.device = getattr(config, "device", "cuda")

        self.robot_config = build_robot_config(config)
        self.robot_tf = build_robot(config, self.robot_config)
        self.mesh_paths = self.robot_config.mesh_paths

        self.sam3_extractor = Sam3Extractor(bpe_path=config.bpe_path, ckpt_path=config.ckpt_path)

    def _to_float_tensor(self, value):
        if isinstance(value, torch.Tensor):
            return value.detach().clone().to(device=self.device, dtype=torch.float32)
        return torch.as_tensor(value, dtype=torch.float32, device=self.device)

    def _prepare_link_poses(self, joint_angles):
        link_poses_list = self.robot_tf.fkine_all(joint_angles)
        return torch.as_tensor(link_poses_list, dtype=torch.float32, device=self.device)

    def _prepare_mask_tensor(self, mask):
        if mask is None:
            raise ValueError("mask 不能为空")

        mask_tensor = self._to_float_tensor(mask)
        if mask_tensor.ndim == 2:
            mask_tensor = mask_tensor.unsqueeze(0)
        if mask_tensor.ndim == 4 and mask_tensor.shape[1] == 1:
            mask_tensor = mask_tensor[:, 0]
        if mask_tensor.ndim != 3:
            raise ValueError(f"mask 需要是 (H, W) 或 (B, H, W)，当前 shape={tuple(mask_tensor.shape)}")
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
        cv2.imwrite(f'output_with_mask{time.time()}.png', overlay)

        return result, loss_dict

    def refine(self, video, joint_angles, intrinsic, extrinsic, base_path, mask=None, max_steps=3000, mask_id=0):
        H, W = video.shape[1:3]
        extrinsic = self._to_float_tensor(extrinsic)
        solver = RBSolver(self.mesh_paths, H, W, extrinsic, device=self.device)
        solver.to(self.device)

        link_poses_list = self._prepare_link_poses(joint_angles[mask_id:mask_id + 1])

        if mask is None:    
            mask = self.sam3_extractor.extract_masks(video[mask_id])
        mask_tensor = self._prepare_mask_tensor(mask)

        dps = {
            "global_step": 0,
            "mask": mask_tensor,
            "link_poses": link_poses_list,
            "K": self._to_float_tensor(intrinsic).unsqueeze(0),
        }

        pose_optimizer =  torch.optim.Adam(
            solver.parameters(),
            lr=1e-4,
            weight_decay=1e-6,
        )

        os.makedirs(base_path, exist_ok=True)
        for k in range(max_steps):
            output, loss_dict = solver.forward(dps)
            loss = loss_dict["mask_loss"]
            loss.backward()
            pose_optimizer.step()
            pose_optimizer.zero_grad(set_to_none=True)
            if k % 100 == 0:
                print(k, loss)
                tsfm = output["tsfm"]
                loss = loss.detach().cpu()
                
                save_path = os.path.join(base_path,f"{k}")
                pred_mask_path = os.path.join(base_path,f"pred_mask")
                os.makedirs(save_path, exist_ok=True)
                os.makedirs(pred_mask_path, exist_ok=True)
                with open(os.path.join(save_path, 'loss.txt'),'w')as f:
                    f.write(str(loss))
                with open(os.path.join(save_path, 'tsfm.txt'),'w')as f:
                    f.write(str(tsfm))
                with open(os.path.join(save_path, 'intrinsic.txt'),'w') as f:
                    f.write(str(intrinsic))

                idx=0
                cur_rgb = video[mask_id][:, :, ::-1]
                img = Image.fromarray(video[mask_id])
                img.save(os.path.join(base_path, f'origin.png'))
                
                render_pre = output["rendered_masks"][idx].detach().cpu()
                render_gt = mask_tensor.detach().cpu()[0].float()
                
                save_img(render_gt,path = os.path.join(save_path, f'mask_{idx}_gt.png'))
                save_img(render_gt,path = os.path.join(base_path, f'mask_gt.png'))
                save_img(render_pre,path = os.path.join(save_path, f'mask_{idx}_pred.png'))

                # 生成彩色版本的 mask（红色叠加）
                # color = np.zeros_like(cur_rgb)
                color = [0, 255, 0]  # r
                mask_render = render_pre.numpy().astype(np.uint8)
                mask_render = (mask_render > 0).astype(np.uint8)
                overlay = add_mask(cur_rgb, mask_render, color=color, alpha=0.5)
                cv2.imwrite(os.path.join(save_path, f'output_with_mask_{idx}.png'), overlay)
                cv2.imwrite(os.path.join(pred_mask_path, f'{k}.png'), overlay)
        return output, loss_dict     