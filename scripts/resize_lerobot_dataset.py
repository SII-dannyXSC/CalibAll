#!/usr/bin/env python3
"""
将 LeRobot v2.1 本地任务目录复制为新数据集：所有 ``videos/**/episode_*.mp4`` 缩放为固定分辨率（默认 224x224），
并更新 ``meta/info.json`` 中视频特征的 shape / video_info。

可选：若 parquet 中含 ``calib.intrinsic``、``calib.arms.*.2d_grip_point``、``calib.bbox_*``，按缩放比例同步数值。

依赖：系统 PATH 中有 ``ffmpeg``（libx264）。

示例：
    PYTHONPATH=. python scripts/resize_lerobot_dataset.py \\
        --src /path/to/task \\
        --dst /path/to/task_224
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.caliball.lerobot_resize_dataset import resize_lerobot_dataset


def _ffmpeg_ok() -> bool:
    try:
        r = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return r.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def main() -> None:
    p = argparse.ArgumentParser(description="LeRobot v2.1：视频缩放并写出新目录")
    p.add_argument("--src", type=str, required=True, help="源数据集根目录（含 meta/data/videos）")
    p.add_argument("--dst", type=str, required=True, help="输出根目录（将创建）")
    p.add_argument("--height", type=int, default=224, help="目标高度，默认 224")
    p.add_argument("--width", type=int, default=224, help="目标宽度，默认 224")
    p.add_argument(
        "--no-scale-parquet-calib",
        action="store_true",
        help="不根据分辨率缩放 parquet 中的 calib 内参/2d/bbox",
    )
    p.add_argument(
        "--no-data",
        action="store_true",
        help="不复制 data/（仅写 meta + 视频）",
    )
    p.add_argument("--crf", type=int, default=23, help="libx264 CRF，默认 23")
    p.add_argument(
        "--overwrite-dst",
        action="store_true",
        help="若目标根目录已存在则先删除再写入",
    )
    args = p.parse_args()

    if not _ffmpeg_ok():
        raise SystemExit("未找到可用的 ffmpeg，请先安装并加入 PATH")

    src = Path(args.src).expanduser().resolve()
    dst = Path(args.dst).expanduser().resolve()
    if not src.is_dir():
        raise SystemExit(f"源目录不存在: {src}")
    if dst.resolve() == src.resolve():
        raise SystemExit("dst 不能与 src 相同")
    if dst.exists():
        if args.overwrite_dst:
            shutil.rmtree(dst)
        else:
            raise SystemExit(f"目标已存在: {dst}（加 --overwrite-dst 强制覆盖）")

    resize_lerobot_dataset(
        src,
        dst,
        out_h=args.height,
        out_w=args.width,
        scale_parquet_calib=not args.no_scale_parquet_calib,
        copy_data=not args.no_data,
        ffmpeg_crf=args.crf,
    )
    print(f"[DONE] 已写入 {dst}")


if __name__ == "__main__":
    main()
