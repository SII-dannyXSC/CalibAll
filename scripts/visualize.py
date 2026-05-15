#!/usr/bin/env python3
"""
visualize_labels.py

将标注结果叠加到原始视频帧上并输出可视化视频/图片。

支持两种输入格式：
  - json (默认): 从 label.py --format json 生成的 JSON 文件读取
  - lerobot: 从 label.py --format lerobot 生成的 parquet 数据集读取

用法 (JSON 模式)：
    python scripts/visualize_labels.py \\
        --json_path label_out/berkeley/episode_000000.json \\
        --task_path /path/to/lerobot_dataset \\
        --output_dir label_out/vis

用法 (LeRobot 模式)：
    python scripts/visualize_labels.py \\
        --input_format lerobot \\
        --dataset_dir /tmp/ucsd_kitchen_anno \\
        --output_dir /tmp/vis
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.caliball.utils.video_io import FfmpegVideoReader, FfmpegVideoWriter
from src.caliball.utils.visualization import (
    ArmRenderData,
    BboxItem,
    MaskItem,
    decode_mask,
    render_frame,
    render_frame_split,
)

# ─────────────────────────────── 调色板（脚本层定义） ──────────────────────────

ARM_COLORS = [
    dict(mask_arm=(255, 255, 0), mask_grip=(0, 180, 255),
         bbox_all=(0, 255, 0), bbox_arm=(255, 200, 0), bbox_grip=(0, 128, 255),
         grip=(255, 0, 0)),
    dict(mask_arm=(0, 255, 180), mask_grip=(200, 100, 255),
         bbox_all=(0, 180, 80), bbox_arm=(100, 200, 80), bbox_grip=(180, 80, 255),
         grip=(0, 80, 255)),
]


def _get_palette(arm_i: int) -> dict:
    return ARM_COLORS[min(arm_i, len(ARM_COLORS) - 1)]


# ═══════════════════════════════ 格式转换 ═══════════════════════════════════════


def json_frame_to_arms(frame: dict, mask_shape: tuple, mask_alpha: float = 0.45) -> List[ArmRenderData]:
    """将 JSON 标注帧转换为 ArmRenderData 列表。"""
    result = []
    arm_i = 0
    for arm_name, arm in frame.get("arms", {}).items():
        if arm.get("is_placeholder", False):
            continue
        pal = _get_palette(arm_i)
        tag = arm_name[0].upper()
        xyz_euler_g = arm.get("xyz_euler_g")
        xyz_mat_g = arm.get("xyz_mat_g")

        masks = []
        m = decode_mask(arm.get("mask_without_gripper"), mask_shape)
        if m is not None:
            masks.append(MaskItem(m, pal["mask_arm"], mask_alpha))
        m = decode_mask(arm.get("mask_gripper"), mask_shape)
        if m is not None:
            masks.append(MaskItem(m, pal["mask_grip"], mask_alpha * 0.6))

        bboxes = []
        for key, color, label in [
            ("bbox_with_gripper",    pal["bbox_all"],  f"{tag}:robot"),
            ("bbox_without_gripper", pal["bbox_arm"],  f"{tag}:arm"),
            ("bbox_gripper",         pal["bbox_grip"], f"{tag}:grip"),
        ]:
            b = arm.get(key)
            if b is not None:
                bboxes.append(BboxItem(b, color, label))

        result.append(ArmRenderData(
            name=arm_name,
            uv=arm.get("uv"),
            uv_color=pal["grip"],
            masks=masks,
            bboxes=bboxes,
            xyz_cam=xyz_mat_g[:3] if xyz_mat_g else None,
            rot_flat=xyz_mat_g[3:12] if xyz_mat_g else None,
            gripper_val=float(xyz_euler_g[-1]) if xyz_euler_g else None,
        ))
        arm_i += 1
    return result


def lerobot_row_to_arms(row: dict, cam: str, arm_names: List[str],
                        mask_shape: tuple, prefix: str = "annotation",
                        mask_alpha: float = 0.45) -> List[ArmRenderData]:
    """将 LeRobot parquet 行转换为 ArmRenderData 列表。"""
    result = []
    for arm_i, arm in enumerate(arm_names):
        pal = _get_palette(arm_i)
        tag = arm[0].upper()

        def get(field):
            v = row.get(f"{prefix}.{cam}.{arm}.{field}")
            if v is None:
                return None
            return v.tolist() if hasattr(v, "tolist") else v

        xyz_mat_g = get("xyz_mat_g")

        def decode_rle_str(s):
            if not s:
                return None
            try:
                return decode_mask(json.loads(s), mask_shape)
            except Exception:
                return None

        masks = []
        m = decode_rle_str(get("mask_without_gripper"))
        if m is not None:
            masks.append(MaskItem(m, pal["mask_arm"], mask_alpha))
        m = decode_rle_str(get("mask_gripper"))
        if m is not None:
            masks.append(MaskItem(m, pal["mask_grip"], mask_alpha * 0.6))

        bboxes = []
        for key, color, label in [
            ("bbox_with_gripper",    pal["bbox_all"],  f"{tag}:robot"),
            ("bbox_without_gripper", pal["bbox_arm"],  f"{tag}:arm"),
            ("bbox_gripper",         pal["bbox_grip"], f"{tag}:grip"),
        ]:
            b = get(key)
            if b is not None:
                bboxes.append(BboxItem(b, color, label))

        result.append(ArmRenderData(
            name=arm,
            uv=get("uv"),
            uv_color=pal["grip"],
            masks=masks,
            bboxes=bboxes,
            xyz_cam=xyz_mat_g[:3] if xyz_mat_g and len(xyz_mat_g) >= 12 else None,
            rot_flat=xyz_mat_g[3:12] if xyz_mat_g and len(xyz_mat_g) >= 12 else None,
        ))
    return result


# ═══════════════════════════════ JSON MODE ══════════════════════════════════════


def _camera_keys_from_json(label_data: dict):
    return [k for k, v in label_data.items()
            if k != "video_seg" and isinstance(v, dict) and "frames" in v]


def _visualize_json_episode(json_path, task_path, output_dir,
                            camera_names, fps, show_mask, show_bbox,
                            show_point, mask_alpha, show_axes, axes_scale,
                            first_frame_only):
    with open(json_path, "r", encoding="utf-8") as f:
        label_data = json.load(f)

    if not camera_names:
        camera_names = _camera_keys_from_json(label_data)
        if not camera_names:
            print("[ERROR] 未指定 --cameras，且无法从 JSON 推断")
            return

    ep_idx = int(os.path.splitext(os.path.basename(json_path))[0].split("_")[-1])
    ep_tag = f"ep{ep_idx:06d}"
    os.makedirs(output_dir, exist_ok=True)

    rk = dict(show_mask=show_mask, show_bbox=show_bbox, show_point=show_point,
              show_axes=show_axes, axes_scale=axes_scale)

    for cam_name in camera_names:
        if cam_name not in label_data:
            continue
        cam_data = label_data[cam_name]
        K_mat = cam_data.get("intrinsic") if isinstance(cam_data, dict) else None
        K = np.array(K_mat, dtype=np.float64) if K_mat else None
        vid_pattern = os.path.join(task_path, "videos", "chunk-*", cam_name,
                                   f"episode_{ep_idx:06d}.mp4")
        matches = sorted(glob.glob(vid_pattern))
        if not matches:
            continue
        info_list = cam_data.get("frames", []) if isinstance(cam_data, dict) else cam_data
        tag = f"{ep_tag}_{cam_name}"

        try:
            reader = FfmpegVideoReader(matches[0])
        except Exception:
            continue
        H, W = reader.height, reader.width

        if first_frame_only:
            ok, bgr = reader.read()
            reader.release()
            if not ok:
                continue
            arms = json_frame_to_arms(info_list[0] if info_list else {}, (H, W), mask_alpha)
            full, mask_vis, bbox_vis, pts_vis = render_frame_split(bgr, arms, K=K, **rk)
            for suffix, img in [("full", full), ("mask", mask_vis),
                                ("bbox", bbox_vis), ("points", pts_vis)]:
                cv2.imwrite(os.path.join(output_dir, f"{tag}_{suffix}.jpg"), img)
        else:
            T = len(info_list) if reader.n_frames < 0 else min(reader.n_frames, len(info_list))
            writers = {k: FfmpegVideoWriter(os.path.join(output_dir, f"{tag}_{k}.mp4"), fps, W, H)
                       for k in ("full", "mask", "bbox", "points")}
            for i in range(T):
                ok, bgr = reader.read()
                if not ok:
                    break
                arms = json_frame_to_arms(info_list[i], (H, W), mask_alpha)
                full, mask_vis, bbox_vis, pts_vis = render_frame_split(bgr, arms, K=K, **rk)
                writers["full"].write(full)
                writers["mask"].write(mask_vis)
                writers["bbox"].write(bbox_vis)
                writers["points"].write(pts_vis)
            reader.release()
            for w in writers.values():
                w.release()
        print(f"[OK] ep {ep_idx} / {cam_name} → {output_dir}/")


# ═══════════════════════════════ LEROBOT MODE ══════════════════════════════════

_ANN_PREFIX = "annotation."


def _parse_ann_cols(col_names):
    struct: Dict[str, Dict[str, set]] = {}
    for c in col_names:
        if not c.startswith(_ANN_PREFIX):
            continue
        parts = c[len(_ANN_PREFIX):].split(".")
        if len(parts) < 3:
            continue
        struct.setdefault(".".join(parts[:-2]), {}).setdefault(parts[-2], set()).add(parts[-1])
    return struct


def _find_video(dataset_dir, cam, ep_idx, chunks_size):
    chunk = ep_idx // chunks_size
    ep_str = f"episode_{ep_idx:06d}.mp4"
    for p in [dataset_dir / "videos" / f"chunk-{chunk:03d}" / cam / ep_str,
              dataset_dir / "videos" / cam / f"chunk-{chunk:03d}" / ep_str]:
        if p.exists():
            return p
    return None


def _visualize_lerobot(dataset_dir, output_dir, episodes, fps,
                       show_mask, show_bbox, show_point, mask_alpha, axes_scale):
    import pyarrow.parquet as pq
    from tqdm import tqdm

    dataset_dir = Path(dataset_dir).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    info = json.loads((dataset_dir / "meta" / "info.json").read_text())
    chunks_size = info.get("chunks_size", 1000)
    total_episodes = info.get("total_episodes", 0)

    ann_struct = _parse_ann_cols([k for k in info.get("features", {}) if k.startswith(_ANN_PREFIX)])
    if not ann_struct:
        print("[ERROR] 未找到 annotation 列")
        return

    calib_path = dataset_dir / "meta" / "calibration.json"
    calib = json.loads(calib_path.read_text()) if calib_path.exists() else {}

    ep_list = episodes if episodes is not None else list(range(total_episodes))

    for ep_idx in tqdm(ep_list, desc="Episodes"):
        chunk = ep_idx // chunks_size
        pq_path = dataset_dir / "data" / f"chunk-{chunk:03d}" / f"episode_{ep_idx:06d}.parquet"
        if not pq_path.exists():
            continue

        table = pq.read_table(pq_path)
        n_rows = len(table)
        col_data = {name: table.column(name).to_pylist() for name in table.schema.names}
        ep_out = output_dir / f"episode_{ep_idx:06d}"
        ep_out.mkdir(parents=True, exist_ok=True)

        for cam, arm_dict in ann_struct.items():
            arm_names = sorted(arm_dict.keys())
            K = None
            if cam in calib:
                K = np.array(calib[cam]["intrinsic"], dtype=np.float64)
                if K.shape == (9,):
                    K = K.reshape(3, 3)

            vid_path = _find_video(dataset_dir, cam, ep_idx, chunks_size)
            if vid_path is None:
                continue

            reader = FfmpegVideoReader(str(vid_path))
            out_mp4 = ep_out / f"ep{ep_idx:06d}_{cam}_anno.mp4"
            writer = None

            for t in range(n_rows):
                ok, bgr = reader.read()
                if not ok:
                    break
                H, W = bgr.shape[:2]
                if writer is None:
                    writer = FfmpegVideoWriter(str(out_mp4), fps, W, H)
                row = {name: col_data[name][t] for name in col_data}
                arms = lerobot_row_to_arms(row, cam, arm_names, (H, W), mask_alpha=mask_alpha)
                rendered = render_frame(bgr, arms, K=K, show_mask=show_mask,
                                        show_bbox=show_bbox, show_point=show_point,
                                        axes_scale=axes_scale)
                writer.write(rendered)

            reader.release()
            if writer:
                writer.release()
                print(f"  → {out_mp4.relative_to(output_dir)}")


# ═══════════════════════════════ CLI ═══════════════════════════════════════════


def parse_args():
    p = argparse.ArgumentParser(description="标注可视化（支持 JSON 和 LeRobot 格式输入）")
    p.add_argument("--input_format", default="json", choices=["json", "lerobot"])
    p.add_argument("--json_path", default=None)
    p.add_argument("--json_dir",  default=None)
    p.add_argument("--task_path", default=None)
    p.add_argument("--cameras",   nargs="*", default=None)
    p.add_argument("--first_frame_only", action="store_true")
    p.add_argument("--episode",   type=int, default=None)
    p.add_argument("--dataset_dir", default=None)
    p.add_argument("--episodes",    nargs="*", type=int, default=None)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--fps",        type=int, default=15)
    p.add_argument("--no_mask",    action="store_true")
    p.add_argument("--no_bbox",    action="store_true")
    p.add_argument("--no_point",   action="store_true")
    p.add_argument("--no_axes",    action="store_true")
    p.add_argument("--alpha",      type=float, default=0.45)
    p.add_argument("--axes_scale", type=float, default=0.05)
    return p.parse_args()


def main():
    args = parse_args()

    if args.input_format == "lerobot":
        if not args.dataset_dir:
            print("[ERROR] lerobot 模式需要 --dataset_dir"); sys.exit(1)
        _visualize_lerobot(
            args.dataset_dir, args.output_dir, args.episodes,
            args.fps, not args.no_mask, not args.no_bbox,
            not args.no_point, args.alpha, args.axes_scale,
        )
    else:
        if not args.task_path:
            print("[ERROR] json 模式需要 --task_path"); sys.exit(1)
        cams = args.cameras
        if cams is not None and len(cams) == 0:
            cams = None
        kwargs = dict(
            task_path=args.task_path, camera_names=cams, fps=args.fps,
            show_mask=not args.no_mask, show_bbox=not args.no_bbox,
            show_point=not args.no_point, mask_alpha=args.alpha,
            show_axes=not args.no_axes, axes_scale=args.axes_scale,
            first_frame_only=args.first_frame_only,
        )
        if args.json_path:
            ep_name = os.path.splitext(os.path.basename(args.json_path))[0]
            _visualize_json_episode(json_path=args.json_path,
                                    output_dir=os.path.join(args.output_dir, ep_name), **kwargs)
        elif args.json_dir:
            json_files = sorted(f for f in os.listdir(args.json_dir)
                                if f.startswith("episode_") and f.endswith(".json"))
            if args.episode is not None:
                target = f"episode_{args.episode:06d}.json"
                json_files = [f for f in json_files if f == target]
            for fname in json_files:
                ep_name = os.path.splitext(fname)[0]
                _visualize_json_episode(
                    json_path=os.path.join(args.json_dir, fname),
                    output_dir=os.path.join(args.output_dir, ep_name), **kwargs)
        else:
            print("[ERROR] json 模式需要 --json_path 或 --json_dir"); sys.exit(1)

    print("[DONE]")


if __name__ == "__main__":
    main()
