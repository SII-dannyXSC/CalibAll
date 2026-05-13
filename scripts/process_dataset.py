"""
process_dataset.py

遍历 RoboMIND / franka_3rgb 数据集，对每个 episode 的每个相机
生成符合 AGENTS.md 存储格式的标注 JSON 文件。

用法示例：
    python scripts/process_dataset.py \
        --task_path /path/to/franka_3rgb/put_the_red_apple_in_the_bowl \
        --output_dir ./label_result \
        --dataset_name robomind/franka_3rgb \
        --camera_names camera_left camera_right camera_top \
        --robot_type franka \
        --device cuda \
        --max_episodes 10

若 --task_path 对应的目录为空或不存在，脚本会在 benchmark 根目录中自动寻找。
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import trimesh
from omegaconf import OmegaConf
from tqdm import tqdm

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.caliball.dataset.lerobot_dataset import LeRobotDataset
from src.caliball.config.camera_intrinsic_extrinsic import get_intrinsic_extrinsic
from src.caliball.robot import build_robot
from src.caliball.config import build_robot_config
from src.caliball.utils.nvdiffrast_renderer import NVDiffrastRenderer
from src.caliball.label import Labeler


def parse_args():
    parser = argparse.ArgumentParser(
        description="对 LeRobot 数据集进行多模态标注（EEF、mask、bbox、轨迹等）"
    )
    parser.add_argument(
        "--task_path", type=str, required=True,
        help="单个 task 数据集的本地路径，例如 .../franka_3rgb/put_the_red_apple_in_the_bowl"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="标注结果输出目录；每个 episode 存一个 JSON 文件"
    )
    parser.add_argument(
        "--dataset_name", type=str, default="robomind/franka_3rgb",
        help="数据集名称，用于查询相机内外参（camera_intrinsic_extrinsic.py）"
    )
    parser.add_argument(
        "--camera_names", nargs="+",
        default=["camera_left", "camera_right", "camera_top"],
        help="需要处理的相机列表"
    )
    parser.add_argument(
        "--robot_type", type=str, default="franka",
        choices=["franka", "ur5e", "aloha_cobot_magic", "arx5_robotwin"],
        help="机器人类型"
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="torch 设备（cuda / cpu）"
    )
    parser.add_argument(
        "--episode_start", type=int, default=0,
        help="起始 episode 索引（可用于断点续处理）"
    )
    parser.add_argument(
        "--max_episodes", type=int, default=None,
        help="最多处理的 episode 数量（默认全部）"
    )
    parser.add_argument(
        "--skip_mask", action="store_true",
        help="跳过 mask/bbox 渲染（调试时加速）"
    )
    parser.add_argument(
        "--arm_mesh_num", type=int, default=8,
        help="mesh_paths 中前 N 个属于机械臂（之后为 gripper）"
    )
    parser.add_argument(
        "--eef_rotation_type", type=str, default="euler_xyz",
        choices=[
            "euler_xyz",
            "euler_zyx",
            "quaternion",
            "axis_angle",
            "axis_angle_residual",
        ],
        help="数据集中 EEF 旋转的表示方式（axis_angle*=旋转向量 / 轴角残差）"
    )
    parser.add_argument(
        "--benchmark_roots", nargs="+", default=None,
        help="fallback 搜索的 benchmark 根目录列表"
    )
    return parser.parse_args()


_REQUIRED_LEROBOT_DIRS = {"data", "meta", "videos"}


def is_valid_lerobot_task(path):
    if not os.path.isdir(path):
        return False
    entries = set(os.listdir(path))
    return bool(_REQUIRED_LEROBOT_DIRS & entries)


def resolve_task_path(task_path, benchmark_roots):
    if is_valid_lerobot_task(task_path):
        return task_path, False

    task_name = os.path.basename(task_path.rstrip("/"))
    print(f"[WARN] 路径无效或为空: {task_path}")
    print(f"[INFO] 在其他 benchmark 目录中搜索任务: {task_name}")

    for root in benchmark_roots:
        candidate = os.path.join(root, task_name)
        if is_valid_lerobot_task(candidate):
            print(f"[INFO] 找到有效数据集路径: {candidate}")
            return candidate, True

    print(f"[ERROR] 在以下目录中均未找到有效数据集: {benchmark_roots}")
    return None, True


def load_meshes(robot_config, device):
    """加载机器人全部 link mesh，返回 vertices_list, faces_list（CUDA Tensor）。"""
    vertices_list = []
    faces_list = []
    for mesh_path in robot_config.mesh_paths:
        full_path = os.path.join(_ROOT, mesh_path) if not os.path.isabs(mesh_path) else mesh_path
        mesh = trimesh.load(full_path, force="mesh")
        vertices = torch.from_numpy(np.array(mesh.vertices)).float().to(device=device)
        faces = torch.from_numpy(np.array(mesh.faces)).int().to(device=device)
        vertices_list.append(vertices)
        faces_list.append(faces)
    return vertices_list, faces_list


def get_camera_key(cam_name):
    return f"observation.images.{cam_name}"


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    benchmark_roots = args.benchmark_roots
    if benchmark_roots is None:
        benchmark_roots = [
            os.path.join(_ROOT, "data/RoboMIND_lerobot_v2.1/benchmark1_0_compressed/franka_3rgb"),
            os.path.join(_ROOT, "data/RoboMIND_lerobot_v2.1/benchmark1_1_compressed/franka_3rgb"),
            os.path.join(_ROOT, "data/RoboMIND_lerobot_v2.1/benchmark1_2_compressed/franka_3rgb"),
        ]

    task_path, was_resolved = resolve_task_path(args.task_path, benchmark_roots)
    if task_path is None:
        print("[ERROR] 无法找到有效的数据集目录，退出。")
        sys.exit(1)
    if was_resolved and task_path != args.task_path:
        print(f"[INFO] 已切换到 fallback 路径: {task_path}")

    print(f"[INFO] 加载数据集: {task_path}")
    dataset = LeRobotDataset(task_path)
    task_name = os.path.basename(task_path.rstrip("/"))

    config = type("", (), {})()
    config.robot_type = args.robot_type
    robot_config = build_robot_config(config)
    tf = build_robot(config, robot_config)

    if not args.skip_mask:
        print("[INFO] 加载机器人 mesh ...")
        vertices_list, faces_list = load_meshes(robot_config, args.device)
    else:
        vertices_list, faces_list = [], []
        print("[INFO] --skip_mask 已设置，跳过 mesh 加载")

    labeler = Labeler(config=None)

    n_total = len(dataset)
    ep_end = n_total
    if args.max_episodes is not None:
        ep_end = min(ep_end, args.episode_start + args.max_episodes)
    print(f"[INFO] 将处理 episode [{args.episode_start}, {ep_end})，共 {ep_end - args.episode_start} 个")

    for ep_idx in tqdm(range(args.episode_start, ep_end), desc="Episodes"):
        out_path = os.path.join(args.output_dir, f"episode_{ep_idx:06d}.json")
        if os.path.exists(out_path):
            print(f"[SKIP] episode {ep_idx} 已存在，跳过")
            continue

        data = dataset[ep_idx]
        eef_poses = data.get("eef_pose_world")
        if eef_poses is None:
            eef_poses = data.get("actions")
        if eef_poses is None:
            eef_poses = data.get("action")
        joint_angles = data["states"]
        videos = data["videos"]

        if eef_poses is None or joint_angles is None:
            print(f"[WARN] episode {ep_idx} 缺少 eef_pose_world/actions 或 states，跳过")
            continue

        episode_result = {}

        for cam_name in args.camera_names:
            cam_key = get_camera_key(cam_name)
            if cam_key not in videos:
                continue
            try:
                _, extrinsic_seg = get_intrinsic_extrinsic(
                    args.dataset_name, task_name, cam_name, ep_idx
                )
                seg_points, _ = labeler.segment_video(eef_poses, extrinsic_seg)
                episode_result["video_seg"] = seg_points
            except AssertionError:
                episode_result["video_seg"] = []
            break

        if "video_seg" not in episode_result:
            episode_result["video_seg"] = []

        for cam_name in args.camera_names:
            cam_key = get_camera_key(cam_name)
            if cam_key not in videos:
                continue

            try:
                intrinsic, extrinsic = get_intrinsic_extrinsic(
                    args.dataset_name, task_name, cam_name, ep_idx
                )
            except AssertionError as e:
                print(f"[WARN] {e}，跳过相机 {cam_name}")
                continue

            cam_frames = videos[cam_key]
            cam_H, cam_W = cam_frames[0].shape[:2]
            if not args.skip_mask:
                renderer = NVDiffrastRenderer([cam_H, cam_W], device=args.device)
            else:
                renderer = None

            info_list = labeler.label_episode(
                eef_poses_world=eef_poses,
                joint_angles_list=joint_angles,
                intrinsic=intrinsic,
                extrinsic=extrinsic,
                tf_model=tf,
                renderer=renderer,
                vertices_list=vertices_list,
                faces_list=faces_list,
                eef_rotation_type=args.eef_rotation_type,
                device=args.device,
                arm_mesh_num=args.arm_mesh_num,
                skip_mask=args.skip_mask,
            )

            episode_result[cam_name] = {
                "intrinsic": intrinsic.tolist(),
                "extrinsic": extrinsic.tolist(),
                "info": info_list,
            }

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(episode_result, f, ensure_ascii=False, indent=2)

        print(f"[INFO] episode {ep_idx} → {out_path}")

    print("[DONE] 所有 episode 处理完毕。")


if __name__ == "__main__":
    main()
