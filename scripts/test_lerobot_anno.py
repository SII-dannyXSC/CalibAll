#!/usr/bin/env python3
"""
test_lerobot_anno.py

用官方 LeRobotDataset API 验证 create_lerobot_with_anno.py 输出的标注数据集可用性。

用法：
    python scripts/test_lerobot_anno.py --dataset_dir /tmp/ucsd_kitchen_anno
"""
import argparse
import sys
from pathlib import Path

import torch

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_dir", required=True, help="create_lerobot_with_anno.py 的输出目录")
    p.add_argument("--episodes", type=int, default=3, help="抽查的 episode 数量")
    return p.parse_args()


def main():
    args = parse_args()
    dataset_dir = Path(args.dataset_dir).resolve()

    print(f"[INFO] 加载数据集: {dataset_dir}")
    ds = LeRobotDataset(
        repo_id=str(dataset_dir),
        root=str(dataset_dir),
    )
    print(f"  total_episodes : {ds.meta.total_episodes}")
    print(f"  total_frames   : {ds.meta.total_frames}")
    print(f"  chunks_size    : {ds.meta.chunks_size}")

    # 列出所有 feature 键
    all_keys = list(ds.features.keys())
    anno_keys = [k for k in all_keys if k.startswith("annotation.")]
    align_keys = [k for k in all_keys if not k.startswith("annotation.")]
    print(f"\n  对齐列 ({len(align_keys)}): {align_keys}")
    print(f"  标注列 ({len(anno_keys)}):")
    for k in anno_keys:
        feat = ds.features[k]
        print(f"    {k}: {feat}")

    # 抽查前 N 个 episode 的第一帧
    n_ep = min(args.episodes, ds.meta.total_episodes)
    print(f"\n[INFO] 抽查前 {n_ep} 个 episode 的第一帧 ...")

    ep_to_frame = {}
    for i in range(len(ds)):
        ep = int(ds[i]["episode_index"])
        if ep not in ep_to_frame:
            ep_to_frame[ep] = i
        if len(ep_to_frame) >= n_ep:
            break

    for ep_idx, frame_idx in sorted(ep_to_frame.items()):
        sample = ds[frame_idx]
        print(f"\n  episode {ep_idx:03d}  (frame {frame_idx}):")
        for k in anno_keys[:4]:          # 只打印前 4 个标注列
            v = sample.get(k)
            if isinstance(v, torch.Tensor):
                print(f"    {k}: shape={list(v.shape)}  {v[:4].tolist()} ...")
            else:
                txt = str(v)
                print(f"    {k}: {txt[:80]}")

    print("\n[PASS] 数据集可正常用 LeRobotDataset 读取")


if __name__ == "__main__":
    main()