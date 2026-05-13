"""
用 Hydra ``compose`` 加载任务 YAML，按 ``tf``、``robot``、``dataset`` 初始化，导出 **按 episode 的标注**
（pickle ``LabelData`` + JSON frames 两种格式）。

示例：
    PYTHONPATH=. python scripts/label_from_config.py \\
        --config src/caliball/config/berkeley_autolab_ur5.yaml \\
        --output_dir ./label_out/berkeley_autolab_ur5
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.caliball.config import (
    compose_job_config_from_path,
    instantiate_dataset,
    instantiate_tf,
)
from src.caliball.config.camera_intrinsic_extrinsic import get_intrinsic_extrinsic
from src.caliball.label import Labeler
from src.caliball.utils.mesh_loader import load_meshes
from src.caliball.pipeline.label_data import LabelData
from src.caliball.utils.nvdiffrast_renderer import NVDiffrastRenderer


def parse_args():
    p = argparse.ArgumentParser(description="Hydra 任务配置：导出 episode JSON 标注")
    p.add_argument(
        "--config",
        type=str,
        required=True,
        help="任务 YAML（如 src/caliball/config/berkeley_autolab_ur5.yaml）",
    )
    p.add_argument("--output_dir", type=str, default=None, help="覆盖配置中的 label.output_dir")
    p.add_argument(
        "--calib_dataset_name",
        type=str,
        default=None,
        help="camera_intrinsic_extrinsic 中的数据集键，默认 label.calib_dataset_name 或 dataset_name",
    )
    p.add_argument(
        "--camera_names",
        nargs="+",
        default=None,
        help="逻辑相机名，默认 label.camera_names",
    )
    p.add_argument("--episode_start", type=int, default=None)
    p.add_argument("--max_episodes", type=int, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--skip_mask", action="store_true", help="跳过 mesh 与 mask 渲染")
    p.add_argument("--arm_mesh_num", type=int, default=None)
    p.add_argument(
        "--base_path",
        type=str,
        default=None,
        help="覆盖 YAML 中的 base_path，并刷新 dataset.repo_id（与 --dataset_name 联用做批量任务）",
    )
    p.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="覆盖 YAML 中的 dataset_name（LeRobot 任务子目录名）",
    )
    return p.parse_args()


def _label_cfg(cfg: DictConfig) -> Dict[str, Any]:
    lc = cfg.get("label") or OmegaConf.create({})
    return OmegaConf.to_container(lc, resolve=True) or {}


def main():
    args = parse_args()
    cfg = compose_job_config_from_path(args.config, project_root=_PROJECT_ROOT)

    if args.base_path is not None:
        cfg.base_path = os.path.expanduser(args.base_path)
    if args.dataset_name is not None:
        cfg.dataset_name = args.dataset_name
    if args.base_path is not None or args.dataset_name is not None:
        bp = cfg.get("base_path")
        dn = cfg.get("dataset_name")
        if bp is None or str(bp).strip() == "":
            raise SystemExit("批量覆盖需要 YAML 或 --base_path 提供 base_path")
        if dn is None or str(dn).strip() == "":
            raise SystemExit("批量覆盖需要 YAML 或 --dataset_name 提供 dataset_name")
        bp_path = Path(str(bp)).expanduser()
        if not bp_path.is_absolute():
            bp_path = (_PROJECT_ROOT / bp_path).resolve()
        else:
            bp_path = bp_path.resolve()
        cfg.dataset.repo_id = str(bp_path / str(dn))

    label_over = _label_cfg(cfg)

    if "tf" not in cfg or cfg.tf is None or not cfg.tf.get("_target_"):
        raise SystemExit("配置中缺少 tf 段或 _target_")
    if "robot" not in cfg or cfg.robot is None:
        raise SystemExit("配置中缺少合并后的 robot")
    if "dataset" not in cfg or cfg.dataset is None:
        raise SystemExit("配置中缺少 dataset 段")

    output_dir = args.output_dir or label_over.get("output_dir")
    if not output_dir:
        raise SystemExit("请指定输出目录：YAML 中 label.output_dir 或命令行 --output_dir")
    output_dir = os.path.expanduser(str(output_dir))
    os.makedirs(output_dir, exist_ok=True)
    label_data_dir = os.path.join(output_dir, "label_data")
    os.makedirs(label_data_dir, exist_ok=True)

    calib_name = cfg.get("calib_dataset_name") or cfg.get("dataset_name")
    if not calib_name:
        raise SystemExit("无法得到 calib 数据集名：请设 label.calib_dataset_name 或 dataset_name")

    camera_names = args.camera_names or label_over.get("camera_names") or ["image"]
    episode_start = args.episode_start if args.episode_start is not None else int(label_over.get("episode_start", 0))
    max_episodes = args.max_episodes if args.max_episodes is not None else label_over.get("max_episodes")
    device = args.device or label_over.get("device", "cuda")
    skip_mask = args.skip_mask or bool(label_over.get("skip_mask", False))

    # arm_mesh_num 优先级：CLI > YAML label.arm_mesh_num > robot.arm.mesh_paths 自动推断
    if args.arm_mesh_num is not None:
        arm_mesh_num = args.arm_mesh_num
    elif label_over.get("arm_mesh_num") is not None:
        arm_mesh_num = int(label_over["arm_mesh_num"])
    else:
        arm_cfg = cfg.robot.get("arm") if hasattr(cfg.robot, "get") else getattr(cfg.robot, "arm", None)
        if arm_cfg is not None and getattr(arm_cfg, "mesh_paths", None) is not None:
            arm_mesh_num = len(list(arm_cfg.mesh_paths))
        else:
            arm_mesh_num = None  # label_mask_and_bbox 内部会按 n_links_per_arm 处理

    print(f"[INFO] tf: {cfg.tf._target_}")
    tf = instantiate_tf(cfg)
    robot_config = cfg.robot

    n_arms = tf.n_arms
    arm_names = ["left", "right"] if n_arms == 2 else ["left"] if n_arms == 1 else [f"arm{i}" for i in range(n_arms)]
    print(f"[INFO] n_arms={n_arms}, arm_names={arm_names}")

    print(f"[INFO] dataset: {cfg.dataset._target_}")
    dataset = instantiate_dataset(cfg)
    
    if not skip_mask:
        print("[INFO] 加载 robot mesh_paths（cfg.robot）...")
        vertices_list, faces_list = load_meshes(robot_config, device, project_root=_PROJECT_ROOT)
    else:
        vertices_list, faces_list = [], []
        print("[INFO] 跳过 mesh 加载（skip_mask）")

    labeler = Labeler(config=None)
    task_name = str(cfg.get("dataset_name", "task"))

    n_total = len(dataset)
    ep_end = n_total
    if max_episodes is not None:
        ep_end = min(ep_end, episode_start + int(max_episodes))
    print(f"[INFO] calib_dataset_name={calib_name!r} cameras={camera_names}")
    print(f"[INFO] 导出格式: episode JSON")
    print(f"[INFO] 处理 episode [{episode_start}, {ep_end})，共 {ep_end - episode_start} 个")

    for ep_idx in tqdm(range(episode_start, ep_end), desc="Episodes"):
        out_path = os.path.join(output_dir, f"episode_{ep_idx:06d}.json")
        if os.path.exists(out_path):
            print(f"[SKIP] episode {ep_idx} 已存在")
            continue

        data = dataset[ep_idx]
        joint_angles = data.get("states")
        videos = data.get("videos") or {}

        if joint_angles is None:
            print(f"[WARN] episode {ep_idx} 缺少 states（关节角），无法标注，跳过")
            continue

        episode_result: dict = {}

        label_data = LabelData(
            dataset_name=task_name,
            episode_id=str(ep_idx),
            arm_names=arm_names,
        )
        pkl_path = os.path.join(label_data_dir, f"episode_{ep_idx:06d}.pkl")

        for cam_name in camera_names:
            cam_key = cam_name
            if cam_key not in videos:
                print(f"[WARN] episode {ep_idx} 无视频键 {cam_key}，跳过该相机")
                continue
            try:
                intrinsic, extrinsic = get_intrinsic_extrinsic(calib_name, task_name, cam_name, ep_idx)
            except (AssertionError, ValueError) as e:
                print(f"[WARN] {e}，跳过相机 {cam_name}")
                continue

            cam_frames = videos[cam_key]
            cam_H, cam_W = cam_frames[0].shape[:2]
            renderer = None if skip_mask else NVDiffrastRenderer([cam_H, cam_W], device=device)

            cam_label = labeler.label_episode(
                joint_angles_list=joint_angles,
                intrinsic=intrinsic,
                extrinsic=extrinsic,
                tf_model=tf,
                renderer=renderer,
                vertices_list=vertices_list,
                faces_list=faces_list,
                device=device,
                arm_mesh_num=arm_mesh_num,
                skip_mask=skip_mask,
                dataset_name=task_name,
                episode_id=str(ep_idx),
                camera_name=cam_name,
                arm_names=arm_names,
            )

            for frame in cam_label.cameras.get(cam_name, []):
                label_data.add_frame(cam_name, frame)

            episode_result[cam_name] = {
                "intrinsic": intrinsic.tolist(),
                "extrinsic": extrinsic.tolist(),
                "frames": [asdict(f) for f in cam_label.cameras.get(cam_name, [])],
            }

        label_data.save(pkl_path)
        print(f"[INFO] episode {ep_idx} LabelData → {pkl_path}")

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(episode_result, f, ensure_ascii=False, indent=2)
        print(f"[INFO] episode {ep_idx} → {out_path}")

    print("[DONE]")


if __name__ == "__main__":
    main()
