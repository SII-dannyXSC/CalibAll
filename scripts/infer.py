"""
CalibAll 推理脚本 — 支持自动和手动标注两种模式。

自动模式:
    PYTHONPATH=. python scripts/infer.py \
        --dataset_path /path/to/lerobot/dataset \
        --robot_type franka \
        --episode 1

手动模式 (从 JSON 配置读取标注参数):
    PYTHONPATH=. python scripts/infer.py \
        --manual_config manual_label/config.json

公共可选参数:
    --camera          相机名 (自动模式下默认用第一个)
    --start / --end   帧范围
    --stride          跳帧采样间隔
    --state_key       关节角度的 key (默认 actions.joint_position)
    --max_steps       refinement 最大步数
    --output_dir      输出目录前缀
    --skip_coarse     跳过 coarse init, 直接从文件加载 extrinsic/intrinsic
    --init_extrinsic  初始 extrinsic 矩阵 (4x4 .npy 文件)
    --init_intrinsic  初始 intrinsic 矩阵 (3x3 .npy 文件)
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
from omegaconf import OmegaConf
from PIL import Image

from src.caliball.coarse_init import CoarseInit
from src.caliball.dataset.lerobot_dataset import LeRobotDataset
from src.caliball.refinement import Refinement


def parse_args():
    parser = argparse.ArgumentParser(description="CalibAll inference")

    # Mode selection
    parser.add_argument("--manual_config", type=str, default=None,
                        help="Path to manual annotation JSON config")

    # Dataset
    parser.add_argument("--dataset_path", type=str, default=None,
                        help="Path to LeRobot dataset")
    parser.add_argument("--robot_type", type=str, default="franka",
                        help="Robot type (franka, ur5e, aloha, arx5_robotwin, etc.)")
    parser.add_argument("--episode", type=int, default=0,
                        help="Episode index")
    parser.add_argument("--camera", type=str, default=None,
                        help="Camera name (default: first available)")
    parser.add_argument("--state_key", type=str, default="actions.joint_position",
                        help="Joint state key in dataset")

    # Frame range
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--stride", type=int, default=1,
                        help="Frame sampling stride")

    # Refinement
    parser.add_argument("--max_steps", type=int, default=3000)
    parser.add_argument("--skip_coarse", action="store_true",
                        help="Skip coarse init, use provided extrinsic/intrinsic")
    parser.add_argument("--init_extrinsic", type=str, default=None,
                        help="Path to initial extrinsic .npy (4x4)")
    parser.add_argument("--init_intrinsic", type=str, default=None,
                        help="Path to initial intrinsic .npy (3x3)")

    # Output
    parser.add_argument("--output_dir", type=str, default="results",
                        help="Output directory prefix")

    # Reference image for recognizer
    parser.add_argument("--ref_image", type=str, default="assets/test_img/source.png")
    parser.add_argument("--ref_point", type=int, nargs=2, default=[376, 131],
                        help="Reference point (u, v) on ref_image")

    return parser.parse_args()


def main():
    args = parse_args()

    # --- Load manual config if provided ---
    manual = None
    mask = None
    tracking_point = None
    mask_id = None

    if args.manual_config:
        with open(args.manual_config, "r") as f:
            manual = json.load(f)
        args.dataset_path = args.dataset_path or manual["task_path"]
        args.robot_type = manual.get("robot_type", args.robot_type)
        args.episode = manual.get("episode_idx", args.episode)
        args.camera = manual.get("camera_name", args.camera)
        args.start = manual.get("start_idx", args.start)
        args.end = manual.get("end_idx", args.end)
        args.state_key = manual.get("state_key", args.state_key)
        tracking_point = manual.get("tracking_point")
        stride = manual.get("strike", args.stride)
        args.stride = stride

        if "mask_save_path" in manual:
            mask = np.load(manual["mask_save_path"])
            mask_frame_idx = manual["mask_frame_idx"]

    if args.dataset_path is None:
        raise ValueError("Must provide --dataset_path or --manual_config")

    # --- Load model config ---
    model_config = OmegaConf.load("src/caliball/config/models.yaml")
    model_config.robot_type = args.robot_type

    # --- Load dataset ---
    dataset = LeRobotDataset(args.dataset_path, state_key=args.state_key)
    data = dataset[args.episode]

    # --- Extract video and joint angles ---
    videos = data["videos"]
    if args.camera:
        video = videos[args.camera]
    else:
        camera_key = list(videos.keys())[0]
        args.camera = camera_key
        video = videos[camera_key]
    print(f"Using camera: {args.camera}")

    joint_angles = data["states"]

    # Apply stride
    if args.stride > 1:
        video = video[::args.stride]
        joint_angles = joint_angles[::args.stride]

    # Apply frame range
    end = args.end if args.end is not None else len(video)
    video = video[args.start:end]
    joint_angles = joint_angles[args.start:end]
    print(f"Frames: {len(video)} (start={args.start}, end={end}, stride={args.stride})")

    # Compute mask_id relative to sliced range
    if mask is not None:
        mask_id = mask_frame_idx - args.start

    # --- Build save path ---
    if manual:
        dataset_name = manual.get("dataset_name", "unknown")
        task_name = manual.get("task_name", "unknown")
        save_path = os.path.join(args.output_dir, f"{dataset_name}.{task_name}.{args.camera}.{time.time()}")
    else:
        save_path = os.path.join(args.output_dir, f"{time.time()}")
    os.makedirs(save_path, exist_ok=True)
    print(f"Output: {save_path}")

    # --- Coarse Init ---
    if args.skip_coarse:
        if args.init_extrinsic and args.init_intrinsic:
            extrinsic = np.load(args.init_extrinsic)
            intrinsic = np.load(args.init_intrinsic)
        else:
            raise ValueError("--skip_coarse requires --init_extrinsic and --init_intrinsic")
    else:
        img_pil = Image.open(args.ref_image).convert("RGB")
        p = tuple(args.ref_point)

        coarse_init_pipe = CoarseInit(config=model_config)
        coarse_init_pipe.to("cuda")
        coarse_init_pipe._init_recognizer(img_pil, p)
        coarse_init_pipe._init_intrinsic()

        extrinsic, intrinsic = coarse_init_pipe.get_extrinsic(
            video=video,
            joint_angles=joint_angles,
            tracking_point=tracking_point,
            img_idx=0,
            save_path=save_path,
        )

    print(f"extrinsic=\n{extrinsic}")
    print(f"intrinsic=\n{intrinsic}")

    # --- Refinement ---
    refinement_pipe = Refinement(config=model_config)

    refine_kwargs = dict(
        video=video,
        joint_angles=joint_angles,
        intrinsic=intrinsic,
        extrinsic=extrinsic,
        base_path=save_path,
        max_steps=args.max_steps,
    )
    if mask is not None:
        refine_kwargs["mask"] = mask
        refine_kwargs["mask_id"] = mask_id

    result, loss_dict = refinement_pipe.refine(**refine_kwargs)
    print("Refinement done.")


if __name__ == "__main__":
    main()
