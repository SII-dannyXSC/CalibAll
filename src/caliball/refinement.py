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

        self.robot_tf = build_robot()
        self.robot_config = build_robot_config()
        self.mesh_paths = self.robot_config.mesh_paths

        self.sam3_extractor = Sam3Extractor(bpe_path=config.bpe_path, ckpt_path=config.ckpt_path)

    def render(self, video, joint_angles, intrinsic, extrinsic):
        H, W = video.shape[1:3]
        device = "cuda"
        solver = RBSolver(self.mesh_paths, H, W, extrinsic, device = device)
        solver.to(device)

        link_poses_list = self.robot_tf.fkine_all(joint_angles[:1])    # 1 N 4 4
        link_poses_list = torch.tensor(link_poses_list).to(device = device,dtype=torch.float32)

        dps = {
            "global_step": 0,
            "mask": torch.zeros((H, W)).unsqueeze(0).to(device=device),
            "link_poses": link_poses_list,
            "K": torch.tensor(intrinsic).unsqueeze(0).to(dtype=torch.float32, device=device)
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

    def refine(self, video, joint_angles, intrinsic, extrinsic, max_steps=3000):
        H, W = video.shape[1:3]
        device = "cuda"
        solver = RBSolver(self.mesh_paths, H, W, extrinsic, device = device)
        solver.to(device)

        link_poses_list = self.robot_tf.fkine_all(joint_angles[:1])    # 1 N 4 4
        link_poses_list = torch.tensor(link_poses_list).to(device = device,dtype=torch.float32)

        mask = self.sam3_extractor.extract_masks(video[0])

        dps = {
            "global_step": 0,
            "mask": mask.unsqueeze(0).to(device=device),
            "link_poses": link_poses_list,
            "K": torch.tensor(intrinsic).unsqueeze(0).to(dtype=torch.float32, device=device)
        }

        pose_optimizer =  torch.optim.Adam(
            solver.parameters(),
            lr=1e-4,
            weight_decay=1e-6,
        )

        base_path = f"./results/{time.time()}"
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
                os.makedirs(save_path, exist_ok=True)
                with open(os.path.join(save_path, 'loss.txt'),'w')as f:
                    f.write(str(loss))
                with open(os.path.join(save_path, 'tsfm.txt'),'w')as f:
                    f.write(str(tsfm))
                                
                idx=0
                cur_rgb = video[0][:, :, ::-1]
                render_pre = output["rendered_masks"][idx].detach().cpu()
                
                save_img(render_pre,path = os.path.join(save_path, f'mask_{idx}_pred.png'))

                # 生成彩色版本的 mask（红色叠加）
                color = np.zeros_like(cur_rgb)
                color[:, :, 2] = 255  # r
                mask_render = render_pre.numpy().astype(np.uint8)
                mask_render = (mask_render > 0).astype(np.uint8)
                overlay = add_mask(cur_rgb, mask_render, color=color, alpha=0.5)
                cv2.imwrite(os.path.join(save_path, f'output_with_mask_{idx}.png'), overlay)
        return output, loss_dict     
                        

        # H, W = video.shape[1:3]
        # solver = RBSolver(mesh_paths, H, W, extrinsic, device = "cuda")
        # solver = solver.to("cuda")
        # solver.eval()
        # solver.forward(video, joint_angles)
        # return solver.output