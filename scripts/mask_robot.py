from PIL import Image
import debugpy
import numpy as np
import os
import cv2
import time

# from src.caliball.dataset.lerobot_dataset import LeRobotDataset
from src.caliball.utils.intrinsic_estimator import build_intrinsic_estimator
from src.caliball.dataset.tfds_dataset import TfdsDataset
from src.caliball.robot import build_robot
from src.caliball.config import build_robot_config
from src.caliball.utils.nvdiffrast_renderer import NVDiffrastRenderer
from src.caliball.dataset.berkeley_ur5_dataset import BerkeleyUr5Dataset
from src.caliball.utils.image import add_mask
import torch
import trimesh
# debugpy.listen(("0.0.0.0", 10092))
# print("🔍 Waiting for VSCode attach on 0.0.0.0:10092 ...")
# debugpy.wait_for_client()

# dataset = DroidDataset("/cpfs02/user/xiesicheng.xsc/CalibAll/data",split="train[:10]")
# dataset = LeRobotDataset("/cpfs01/user/wenji.zj/dataspace/Data4QwenVLA/RoboMIND_lerobot_v2.1/benchmark1_1_compressed/franka_3rgb/put_the_red_apple_in_the_bowl")
device = "cuda"
# ROOT = "/inspire/hdd/global_user/xiesicheng-253108120120/project/dzj/CalibAll/dataset/tfds/"
# NAME = "toto"

# dataset = TfdsDataset(
#     root_dir=ROOT,
#     name=NAME,
#     split="train",       # 现在用 train 也不会在 init 爆内存（逐 episode 读）
#     max_episodes=3,
#     max_steps=200,       # 先小一点验证；跑通再放开/设为 None
#     action_key="world_vector"
# )

ROOT = "/inspire/hdd/global_user/xiesicheng-253108120120/project/dzj/CalibAll/dataset/tfds/"
NAME = "berkeley_autolab_ur5"

dataset = BerkeleyUr5Dataset(
    root_dir=ROOT,
    name=NAME,
    split="train",       # 现在用 train 也不会在 init 爆内存（逐 episode 读）
    max_episodes=3,
    max_steps=200,       # 先小一点验证；跑通再放开/设为 None
)

# Tc_c2b = torch.tensor([[ 0.8986,  0.4315,  0.0795, -0.3944],
#         [ 0.2157, -0.2767, -0.9364,  0.0679],
#         [-0.3821,  0.8586, -0.3417,  1.1462],
#         [ 0.0000,  0.0000,  0.0000,  1.0000]]).to(device=device)

Tc_c2b = torch.tensor([[ 0.6316,  0.7753,  0.0072,  0.3458],
        [ 0.4637, -0.3703, -0.8049,  0.1804],
        [-0.6213,  0.5117, -0.5934,  0.4648],
        [ 0.0000,  0.0000,  0.0000,  1.0000]]).to(device=device)

intrinsic_estimator = build_intrinsic_estimator()

config = type('', (), {})()  # create empty config object  
config.robot_type = "ur5e"
robot_config = build_robot_config(config)
tf = build_robot(config, robot_config)

vertices_list = []
faces_list = []
for link_idx, mesh_path in enumerate(robot_config.mesh_paths):
    mesh = trimesh.load(mesh_path, force = 'mesh')
    vertices = torch.from_numpy(mesh.vertices).float().to(device=device)
    faces = torch.from_numpy(mesh.faces).int().to(device=device)
    vertices_list.append(vertices)
    faces_list.append(faces)
nlinks = len(robot_config.mesh_paths)


# Initialize VideoWriter for each type of frame
def initialize_video_writer(output_dir, video_filename, frame_height, frame_width):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 format
    return cv2.VideoWriter(os.path.join(output_dir, video_filename), fourcc, 30, (frame_width, frame_height))


for data in dataset:
    video = data["video"]    # T H W C
    joint_angles = data["states"]  # T 6
    eef_pose = data['action']
    

    length = len(video)
    
    img_pil = Image.fromarray(video[0])
    _intrinsic, origin_width, origin_height = intrinsic_estimator.estimate(img_pil=img_pil)
    width, height = img_pil.size

    _intrinsic[0, :3] *= 1.0 * width / origin_width
    _intrinsic[1, :3] *= 1.0 * height / origin_height
    K = _intrinsic
    print(f"{K=}")
    
    vis_video = video.copy()
    H, W = vis_video[0].shape[:2]
    base_path = f"./visualized_videos/{time.time()}"
    os.makedirs(base_path, exist_ok=True)
    
    renderer = NVDiffrastRenderer([H, W], device=device)

    colors = np.array([
        [255, 0, 0],        # Red
        [0, 255, 0],        # Green
        [0, 0, 255],        # Blue
        [255, 165, 0],      # Orange
        [255, 20, 147],     # Deep Pink
        [138, 43, 226],     # Blue Violet
        [255, 105, 180],    # Hot Pink
        [75, 0, 130],       # Indigo
        [255, 99, 71],      # Tomato
        [128, 0, 0]         # Maroon
    ],dtype=np.float32)
    colors = colors / 255
    
    all_depths = []
    
    origin_path = os.path.join(base_path, "origin")
    yellow_path = os.path.join(base_path, "yellow")
    blue_path = os.path.join(base_path, "blue")
    color_path = os.path.join(base_path, "color")
    depth_path = os.path.join(base_path, "depth")
    os.makedirs(origin_path, exist_ok=True)
    os.makedirs(yellow_path, exist_ok=True)
    os.makedirs(blue_path, exist_ok=True)
    os.makedirs(color_path, exist_ok=True)
    os.makedirs(depth_path, exist_ok=True)


    # Initialize video writers for each type of image
    video_writer_depth = initialize_video_writer(base_path, "depth_video.mp4", H, W)
    video_writer_overlay1 = initialize_video_writer(base_path, "overlay1_video.mp4", H, W)
    video_writer_overlay2 = initialize_video_writer(base_path, "overlay2_video.mp4", H, W)
    video_writer_img = initialize_video_writer(base_path, "img_video.mp4", H, W)
    video_writer_blended = initialize_video_writer(base_path, "blended_video.mp4", H, W)

    for i in range(len(vis_video)):
        mg = video[i]
        states_list = joint_angles[i]
        link_poses_list = tf.fkine_all([states_list])    # 1 N 4 4
        link_poses_list = torch.tensor(link_poses_list).to(device=device, dtype=torch.float32)
        link_poses = link_poses_list[0]
        # --- 2. 初始化全局缓冲 ---
        # 全局颜色缓冲 (H, W, 3) - 设为黑色背景
        final_color = torch.zeros((H, W, 3), dtype=torch.float32, device=device)

        # 全局深度缓冲 (H, W) - 设为无穷大
        # 使用一个大数（例如 1e10）或 torch.finfo(torch.float32).max
        # Droid 数据集的深度通常在 0 到 1 之间 (z/w)，所以 2.0 已经足够大了
        final_depth = torch.full((H, W), 2.0, dtype=torch.float32, device=device)

        # 全局 Mask 缓冲 (H, W) - 设为 0
        final_mask = torch.zeros((H, W), dtype=torch.float32, device=device)

        # --- 3. 循环渲染和 Z-Buffer 合并 ---
        for link_idx in range(nlinks):
            # a. 计算当前 link 的相机坐标系下的位姿
            # 注意：link_poses[link_idx] 应该是 link 坐标系到 base 坐标系的变换 Tb_l
            # 所以 object_pose = Tc_b @ Tb_l = Tc_l
            Tc_c2l = Tc_c2b @ link_poses[link_idx]
            
            # b. 获取当前 link 的网格数据
            verts = vertices_list[link_idx]
            faces = faces_list[link_idx]
            
            # c. 渲染当前 link
            render_info = renderer.render_all(verts, faces, K=K, object_pose=Tc_c2l,mask_color=colors[link_idx])
            
            # render_info["color"] (H, W, 3), render_info["depth"] (H, W), render_info["mask"] (H, W)
            current_color = render_info["color"]
            current_depth = render_info["depth"]
            current_mask = render_info["mask"] # 用于标记前景区域 (深度 > 0)
            # current_mask[current_mask>0] = 1.0
            
            
            # 找到当前渲染物体中，深度大于 0 (即非背景) 的像素
            # 并满足当前深度 < 最终深度 的条件
            valid_mask = (current_depth > 0) & (current_depth < final_depth)
            
            # 确保 mask 维度兼容 (H, W, 3) 
            # torch.unsqueeze(valid_mask, dim=-1) -> (H, W, 1)
            # expand(H, W, 3) 广播到 (H, W, 3) 
            color_mask = valid_mask.unsqueeze(-1).expand_as(final_color) 

            # 1. 深度合并 (最小深度)
            # 只有更近的深度才更新
            final_depth[valid_mask] = current_depth[valid_mask]
            
            # 2. 颜色合并 (按深度大小合并)
            # 只有更近的颜色才更新
            final_color[color_mask] = current_color[color_mask]
            
            # 3. Mask 合并 (也按深度大小合并)
            # 只有更近的 Mask 值才更新
            # final_mask[valid_mask] = current_mask[valid_mask]
            final_mask[valid_mask] = current_mask[valid_mask]

        final_depth[final_depth > 1] = 0.0

        final_color = final_color.cpu().numpy()
        final_depth = final_depth.cpu().numpy()
        final_mask = final_mask.cpu().numpy()
        
        all_depths.append(final_depth)
        img = video[i]
        Image.fromarray(img).save(os.path.join(origin_path, f"{i:05d}.png"))
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        video_writer_img.write(img_bgr)
        
        color_vis = (final_color.clip(0, 1) * 255).astype(np.uint8)
        alpha = 0.5
        color_mask = np.any(color_vis > 0, axis=-1, keepdims=True)
        color_mask = np.broadcast_to(color_mask, img.shape)
        blended = img.copy()
        blended[color_mask] = (1 - alpha) * img[color_mask] + alpha * color_vis[color_mask]
        Image.fromarray(blended).save(os.path.join(color_path, f"{i:05d}.png"))
        blended_bgr = cv2.cvtColor(blended, cv2.COLOR_RGB2BGR)
        video_writer_blended.write(blended_bgr)
        
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        final_mask = final_mask.astype(int)
        overlay1 = add_mask(img, final_mask, color = [56,179,253], alpha= 0.7)
        cv2.imwrite(os.path.join(yellow_path, f"{i:05d}.png"), overlay1)
        video_writer_overlay1.write(overlay1)
        
        overlay2 = add_mask(img, final_mask, alpha= 0.7)
        cv2.imwrite(os.path.join(blue_path, f"{i:05d}.png"), overlay2)
        video_writer_overlay2.write(overlay2)
    cnt = 0
    all_depths = np.array(all_depths)
    foreground_depth = all_depths[all_depths > 0]
    d_min = foreground_depth.min()
    d_max = foreground_depth.max()
    print(d_min, d_max)
    for i in range(len(vis_video)):
        depth_np = all_depths[cnt]
        # depth_normalized = (depth_np - d_min) / (d_max - d_min)
        depth_normalized = (depth_np - d_min) / (d_max - d_min)
        depth_normalized = 0.2 + 0.8 * depth_normalized
        # 保持背景为 0
        depth_normalized[depth_np == 0] = 0
            
        depth_vis = (depth_normalized * 255).astype(np.uint8)
        Image.fromarray(depth_vis, mode='L').save(os.path.join(depth_path, f"{i:05d}.png"))
        depth_vis_bgr = cv2.cvtColor(depth_vis, cv2.COLOR_GRAY2BGR)
        video_writer_depth.write(depth_vis_bgr)
        cnt += 1
    
    break