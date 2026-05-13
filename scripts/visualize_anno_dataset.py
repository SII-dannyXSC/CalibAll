#!/usr/bin/env python3
"""
visualize_anno_dataset.py

从 create_lerobot_with_anno.py 生成的标注 lerobot 数据集中读取 annotation 列，
叠加到原始视频帧上并输出 MP4。

用法：
    python scripts/visualize_anno_dataset.py \
        --dataset_dir /tmp/ucsd_kitchen_anno \
        --output_dir  /tmp/ucsd_kitchen_anno_vis

    # 只处理指定 episode
    python scripts/visualize_anno_dataset.py \
        --dataset_dir /tmp/ucsd_kitchen_anno \
        --output_dir  /tmp/ucsd_kitchen_anno_vis \
        --episodes 0 1 2

    # 关闭 mask（加快速度）
    python scripts/visualize_anno_dataset.py ... --no_mask
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import pyarrow.parquet as pq
from tqdm import tqdm

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ── 复用 visualize_labels.py 的绘图工具 ──────────────────────────────────────
from scripts.visualize_labels import (
    FfmpegVideoReader,
    FfmpegVideoWriter,
    _ARM_PALETTE,
    decode_mask,
    draw_axes,
    draw_bbox,
    draw_point,
    overlay_mask,
)

# ── annotation 列前缀 ─────────────────────────────────────────────────────────
_ANN_PREFIX = "annotation."


def _get_ffmpeg_exe() -> str:
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return "ffmpeg"


_FFMPEG = _get_ffmpeg_exe()


# ── 解析标注列名：annotation.<cam>.<arm>.<field> ──────────────────────────────

def parse_ann_cols(col_names: List[str]):
    """返回 {cam: {arm: set(fields)}} 结构。"""
    struct: Dict[str, Dict[str, set]] = {}
    for c in col_names:
        if not c.startswith(_ANN_PREFIX):
            continue
        rest = c[len(_ANN_PREFIX):]
        parts = rest.split(".")
        if len(parts) < 3:
            continue
        # field 是最后一段，arm 是倒数第二段，cam 是中间所有段
        field = parts[-1]
        arm   = parts[-2]
        cam   = ".".join(parts[:-2])
        struct.setdefault(cam, {}).setdefault(arm, set()).add(field)
    return struct


def _col(cam: str, arm: str, field: str) -> str:
    return f"{_ANN_PREFIX}{cam}.{arm}.{field}"


# ── 读取 calibration（可选，用于画坐标轴） ────────────────────────────────────

def load_calibration(dataset_dir: Path) -> Dict[str, dict]:
    p = dataset_dir / "meta" / "calibration.json"
    if p.exists():
        return json.loads(p.read_text())
    return {}


# ── 找视频文件 ────────────────────────────────────────────────────────────────

def find_video(dataset_dir: Path, cam: str, ep_idx: int, chunks_size: int) -> Optional[Path]:
    chunk = ep_idx // chunks_size
    # lerobot 2.1 布局: videos/chunk-xxx/<cam>/episode_xxxxxx.mp4
    # 也兼容: videos/<cam>/chunk-xxx/episode_xxxxxx.mp4
    ep_str = f"episode_{ep_idx:06d}.mp4"
    candidates = [
        dataset_dir / "videos" / f"chunk-{chunk:03d}" / cam / ep_str,
        dataset_dir / "videos" / cam / f"chunk-{chunk:03d}" / ep_str,
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


# ── 渲染一帧 ─────────────────────────────────────────────────────────────────

def render_frame(
    bgr: np.ndarray,
    row: dict,
    cam: str,
    arm_names: List[str],
    show_mask: bool,
    show_bbox: bool,
    show_point: bool,
    mask_alpha: float,
    K: Optional[np.ndarray],
    axes_scale: float,
) -> np.ndarray:
    H, W = bgr.shape[:2]
    out = bgr.copy()

    for arm_i, arm in enumerate(arm_names):
        pal = _ARM_PALETTE[min(arm_i, len(_ARM_PALETTE) - 1)]
        tag = arm[0].upper()

        def get(field):
            v = row.get(_col(cam, arm, field))
            if v is None:
                return None
            if hasattr(v, "tolist"):
                v = v.tolist()
            return v

        uv           = get("uv")
        bbox_all     = get("bbox_with_gripper")
        bbox_arm     = get("bbox_without_gripper")
        bbox_grip    = get("bbox_gripper")
        rle_arm_str  = get("mask_without_gripper")
        rle_grip_str = get("mask_gripper")
        xyz_mat_g    = get("xyz_mat_g")
        xyz_euler_g  = get("xyz_euler_g")

        # decode mask RLE strings (JSON-encoded by create_lerobot_with_anno.py)
        def decode_rle_str(s):
            if not s:
                return None
            try:
                return decode_mask(json.loads(s), (H, W))
            except Exception:
                return None

        if show_mask:
            out = overlay_mask(out, decode_rle_str(rle_arm_str),  pal["mask_arm"],  alpha=mask_alpha)
            out = overlay_mask(out, decode_rle_str(rle_grip_str), pal["mask_grip"], alpha=mask_alpha * 0.6)

        if show_bbox:
            out = draw_bbox(out, bbox_all,  pal["bbox_all"],  f"{tag}:robot")
            out = draw_bbox(out, bbox_arm,  pal["bbox_arm"],  f"{tag}:arm")
            out = draw_bbox(out, bbox_grip, pal["bbox_grip"], f"{tag}:grip")

        if show_point and uv:
            out = draw_point(out, uv, pal["grip"], tag)

        if K is not None and xyz_mat_g and len(xyz_mat_g) >= 12:
            draw_axes(out, K, xyz_mat_g[:3], xyz_mat_g[3:12], scale=axes_scale)

        # # gripper state label
        # if xyz_euler_g and len(xyz_euler_g) >= 7:
        #     cv2.putText(out, f"grip:{float(xyz_euler_g[-1]):.2f}",
        #                 (10, 20 + arm_i * 18), cv2.FONT_HERSHEY_SIMPLEX,
        #                 0.45, (200, 200, 200), 1, cv2.LINE_AA)

    return out


# ── 处理单个 episode ──────────────────────────────────────────────────────────

def process_episode(
    dataset_dir: Path,
    ep_idx: int,
    chunks_size: int,
    ann_struct: Dict[str, Dict[str, set]],
    calib: Dict[str, dict],
    output_dir: Path,
    fps: int,
    show_mask: bool,
    show_bbox: bool,
    show_point: bool,
    mask_alpha: float,
    axes_scale: float,
):
    chunk = ep_idx // chunks_size
    pq_path = dataset_dir / "data" / f"chunk-{chunk:03d}" / f"episode_{ep_idx:06d}.parquet"
    if not pq_path.exists():
        print(f"[WARN] parquet 不存在: {pq_path}")
        return

    table = pq.read_table(pq_path)
    n_rows = len(table)
    # 转为列字典（避免逐行 slice 开销）
    col_data: Dict[str, list] = {
        name: table.column(name).to_pylist()
        for name in table.schema.names
    }

    ep_out = output_dir / f"episode_{ep_idx:06d}"
    ep_out.mkdir(parents=True, exist_ok=True)

    for cam, arm_dict in ann_struct.items():
        arm_names = sorted(arm_dict.keys())
        K = None
        if cam in calib:
            K = np.array(calib[cam]["intrinsic"], dtype=np.float64)
            if K.shape == (9,):
                K = K.reshape(3, 3)

        vid_path = find_video(dataset_dir, cam, ep_idx, chunks_size)
        if vid_path is None:
            print(f"[WARN] 找不到视频: cam={cam} ep={ep_idx}")
            continue

        reader = FfmpegVideoReader(str(vid_path))
        out_mp4 = ep_out / f"ep{ep_idx:06d}_{cam}_anno.mp4"
        writer: Optional[FfmpegVideoWriter] = None

        for t in range(n_rows):
            ok, bgr = reader.read()
            if not ok:
                break
            H, W = bgr.shape[:2]
            if writer is None:
                writer = FfmpegVideoWriter(str(out_mp4), fps, W, H)

            row = {name: col_data[name][t] for name in col_data}
            rendered = render_frame(
                bgr, row, cam, arm_names,
                show_mask, show_bbox, show_point, mask_alpha, K, axes_scale,
            )
            writer.write(rendered)

        reader.release()
        if writer:
            writer.release()
            print(f"  → {out_mp4.relative_to(output_dir)}")


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="将标注 lerobot 数据集可视化为 MP4")
    p.add_argument("--dataset_dir", required=True, help="create_lerobot_with_anno.py 输出目录")
    p.add_argument("--output_dir",  required=True, help="MP4 输出目录")
    p.add_argument("--episodes",    nargs="*", type=int, default=None,
                   help="指定 episode 编号（默认全部）")
    p.add_argument("--fps",         type=int,   default=15)
    p.add_argument("--no_mask",     action="store_true")
    p.add_argument("--no_bbox",     action="store_true")
    p.add_argument("--no_point",    action="store_true")
    p.add_argument("--alpha",       type=float, default=0.45)
    p.add_argument("--axes_scale",  type=float, default=0.10)
    return p.parse_args()


def main():
    args = parse_args()
    dataset_dir = Path(args.dataset_dir).resolve()
    output_dir  = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # 读 meta
    info = json.loads((dataset_dir / "meta" / "info.json").read_text())
    chunks_size    = info.get("chunks_size", 1000)
    total_episodes = info.get("total_episodes", 0)

    # 解析 annotation 列结构
    ann_col_names = [k for k in info.get("features", {}) if k.startswith(_ANN_PREFIX)]
    ann_struct = parse_ann_cols(ann_col_names)
    if not ann_struct:
        print("[ERROR] 未找到 annotation 列，请确认数据集由 create_lerobot_with_anno.py 生成")
        sys.exit(1)

    cams = sorted(ann_struct.keys())
    print(f"[INFO] 相机: {cams}")
    print(f"[INFO] 标注 arm: { {c: sorted(ann_struct[c].keys()) for c in cams} }")

    calib = load_calibration(dataset_dir)

    ep_list = args.episodes if args.episodes is not None else list(range(total_episodes))
    print(f"[INFO] 处理 {len(ep_list)} 个 episode → {output_dir}")

    for ep_idx in tqdm(ep_list, desc="Episodes"):
        process_episode(
            dataset_dir, ep_idx, chunks_size, ann_struct, calib,
            output_dir,
            fps=args.fps,
            show_mask=not args.no_mask,
            show_bbox=not args.no_bbox,
            show_point=not args.no_point,
            mask_alpha=args.alpha,
            axes_scale=args.axes_scale,
        )

    print(f"\n[DONE] 输出: {output_dir}")


if __name__ == "__main__":
    main()
