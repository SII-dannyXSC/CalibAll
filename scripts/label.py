#!/usr/bin/env python3
"""
label.py

通过 YAML config 运行标注流水线，支持两种输出格式：
  - json (默认): 导出 episode JSON + pickle LabelData
  - lerobot: 创建 LeRobot 2.1 格式数据集 (parquet + meta + video symlinks)

用法：
    # JSON 格式（默认）
    PYTHONPATH=. python scripts/label.py \\
        --config src/caliball/config/berkeley_autolab_ur5.yaml \\
        --output_dir ./label_out/berkeley_autolab_ur5

    # LeRobot 格式
    PYTHONPATH=. python scripts/label.py \\
        --config src/caliball/config/berkeley_autolab_ur5.yaml \\
        --output_dir ./label_out/berkeley_autolab_ur5_lerobot \\
        --format lerobot

    # 断点续标
    PYTHONPATH=. python scripts/label.py ... --resume
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
import sys
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
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
from src.caliball.dataset.lerobot_writer import ALIGN_COLS, LeRobotDatasetWriter
from src.caliball.label import Labeler
from src.caliball.pipeline.label_data import LabelData
from src.caliball.utils.mesh_loader import load_meshes
from src.caliball.utils.nvdiffrast_renderer import NVDiffrastRenderer


def parse_args():
    p = argparse.ArgumentParser(description="标注流水线：从 YAML config 生成标注")
    p.add_argument("--config",       required=True,  help="任务 YAML")
    p.add_argument("--output_dir",   default=None,   help="输出目录")
    p.add_argument("--format",       default="json",
                   choices=["json", "lerobot"],
                   help="输出格式: json (默认) 或 lerobot")
    p.add_argument("--base_path",    default=None,   help="覆盖 YAML base_path")
    p.add_argument("--dataset_name", default=None,   help="覆盖 YAML dataset_name")
    p.add_argument("--camera_names", nargs="+",      default=None)
    p.add_argument("--episode_start",type=int, default=None)
    p.add_argument("--max_episodes", type=int, default=None)
    p.add_argument("--device",       default=None)
    p.add_argument("--skip_mask",    action="store_true")
    p.add_argument("--arm_mesh_num", type=int, default=None)
    p.add_argument("--resume",       action="store_true", help="跳过已存在的输出")
    p.add_argument("--include_original", action="store_true",
                   help="(lerobot 模式) 复制原始 state/action 列")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = compose_job_config_from_path(args.config, project_root=_PROJECT_ROOT)

    # Override base_path / dataset_name
    if args.base_path:
        cfg.base_path = os.path.expanduser(args.base_path)
    if args.dataset_name:
        cfg.dataset_name = args.dataset_name
    if args.base_path or args.dataset_name:
        bp = Path(str(cfg.get("base_path", ""))).expanduser()
        dn = str(cfg.get("dataset_name", ""))
        if not str(bp) or not dn:
            raise SystemExit("--base_path 和 --dataset_name 需要同时提供")
        if not bp.is_absolute():
            bp = (_PROJECT_ROOT / bp).resolve()
        cfg.dataset.repo_id = str(bp / dn)

    label_over = OmegaConf.to_container(
        cfg.get("label") or OmegaConf.create({}), resolve=True
    ) or {}

    output_dir = args.output_dir or label_over.get("output_dir")
    if not output_dir:
        raise SystemExit("请指定输出目录：YAML label.output_dir 或 --output_dir")
    output_dir = Path(os.path.expanduser(str(output_dir)))
    output_dir.mkdir(parents=True, exist_ok=True)

    calib_name   = str(cfg.get("calib_dataset_name") or cfg.get("dataset_name") or "")
    task_name    = str(cfg.get("dataset_name", "task"))
    camera_names = args.camera_names or label_over.get("camera_names") or ["image"]
    ep_start     = args.episode_start if args.episode_start is not None else int(label_over.get("episode_start", 0))
    max_ep       = args.max_episodes  if args.max_episodes  is not None else label_over.get("max_episodes")
    device       = args.device        or label_over.get("device", "cuda")
    skip_mask    = args.skip_mask     or bool(label_over.get("skip_mask", False))

    if args.arm_mesh_num is not None:
        arm_mesh_num = args.arm_mesh_num
    elif label_over.get("arm_mesh_num") is not None:
        arm_mesh_num = int(label_over["arm_mesh_num"])
    else:
        arm_cfg = (cfg.robot.get("arm") if hasattr(cfg.robot, "get")
                   else getattr(cfg.robot, "arm", None))
        arm_mesh_num = (len(list(arm_cfg.mesh_paths))
                        if arm_cfg and getattr(arm_cfg, "mesh_paths", None) else None)

    # Init robot TF
    print(f"[INFO] tf: {cfg.tf._target_}")
    tf_model = instantiate_tf(cfg)
    n_arms   = tf_model.n_arms
    arm_names = (["left", "right"] if n_arms == 2
                 else ["left"]     if n_arms == 1
                 else [f"arm{i}"  for i in range(n_arms)])
    print(f"[INFO] n_arms={n_arms}, arm_names={arm_names}")

    # Init dataset
    print(f"[INFO] dataset: {cfg.dataset._target_}")
    dataset = instantiate_dataset(cfg)

    # Load meshes
    if not skip_mask:
        print("[INFO] 加载 robot mesh_paths...")
        vertices_list, faces_list = load_meshes(cfg.robot, device, project_root=_PROJECT_ROOT)
    else:
        vertices_list, faces_list = [], []

    labeler = Labeler(config=None)

    # Init writer (lerobot mode)
    writer = None
    if args.format == "lerobot":
        orig_root = Path(dataset.lerobot_ds.root)
        chunks_size = dataset.lerobot_ds.meta.chunks_size
        sample_pq_path = orig_root / dataset.lerobot_ds.meta.get_data_file_path(0)
        orig_schema = pq.read_schema(sample_pq_path)
        align_cols_present = [c for c in ALIGN_COLS if c in orig_schema.names]
        orig_copy_cols = (
            [c for c in orig_schema.names if not c.startswith("annotation")]
            if args.include_original else align_cols_present
        )
        writer = LeRobotDatasetWriter(
            output_dir=output_dir,
            orig_root=orig_root,
            chunks_size=chunks_size,
            include_original=args.include_original,
        )
    else:
        label_data_dir = output_dir / "label_data"
        label_data_dir.mkdir(parents=True, exist_ok=True)

    n_total = len(dataset)
    ep_end  = min(n_total, ep_start + int(max_ep)) if max_ep else n_total
    print(f"[INFO] format={args.format}, 处理 episode [{ep_start}, {ep_end})，共 {ep_end - ep_start} 个")

    for ep_idx in tqdm(range(ep_start, ep_end), desc="Episodes"):
        # Resume check
        if args.resume:
            if args.format == "lerobot" and writer.episode_exists(ep_idx):
                writer._processed_eps.append(ep_idx)
                continue
            elif args.format == "json":
                json_path = output_dir / f"episode_{ep_idx:06d}.json"
                if json_path.exists():
                    continue

        data         = dataset[ep_idx]
        joint_angles = data.get("states")
        videos_data  = data.get("videos") or {}

        if joint_angles is None:
            print(f"[WARN] episode {ep_idx} 缺少 states，跳过")
            continue

        label_data = LabelData(
            dataset_name=task_name,
            episode_id=str(ep_idx),
            arm_names=arm_names,
        )
        episode_json: dict = {}

        for cam_name in camera_names:
            if cam_name not in videos_data:
                print(f"[WARN] episode {ep_idx} 无视频键 {cam_name}，跳过")
                continue
            try:
                intrinsic, extrinsic = get_intrinsic_extrinsic(
                    calib_name, task_name, cam_name, ep_idx
                )
            except (AssertionError, ValueError) as e:
                print(f"[WARN] {e}，跳过 {cam_name}")
                continue

            cam_H, cam_W = videos_data[cam_name][0].shape[:2]
            renderer = (None if skip_mask
                        else NVDiffrastRenderer([cam_H, cam_W], device=device))

            cam_label = labeler.label_episode(
                joint_angles_list=joint_angles,
                intrinsic=intrinsic,
                extrinsic=extrinsic,
                tf_model=tf_model,
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

            if args.format == "json":
                episode_json[cam_name] = {
                    "intrinsic": intrinsic.tolist(),
                    "extrinsic": extrinsic.tolist(),
                    "frames": [asdict(f) for f in cam_label.cameras.get(cam_name, [])],
                }

        # Write output
        if args.format == "lerobot":
            align_table = pq.read_table(
                orig_root / dataset.lerobot_ds.meta.get_data_file_path(ep_idx),
                columns=orig_copy_cols,
            )
            writer.write_episode(ep_idx, align_table, label_data.to_columns())
        else:
            label_data.save(output_dir / "label_data" / f"episode_{ep_idx:06d}.pkl")
            with open(output_dir / f"episode_{ep_idx:06d}.json", "w", encoding="utf-8") as f:
                json.dump(episode_json, f, ensure_ascii=False, indent=2)

    # Finalize
    if args.format == "lerobot" and writer and writer.processed_episodes:
        calib: dict = {}
        for cam_name in camera_names:
            try:
                K, T = get_intrinsic_extrinsic(calib_name, task_name, cam_name, 0)
                calib[cam_name] = {"intrinsic": K.tolist(), "extrinsic": T.tolist()}
            except Exception:
                pass
        writer.write_calibration(calib)
        writer.finalize(align_cols_present)

    print(f"\n[DONE] format={args.format}, 输出: {output_dir}")


if __name__ == "__main__":
    main()
