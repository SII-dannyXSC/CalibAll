#!/usr/bin/env python3
"""
create_lerobot_with_anno.py

通过 YAML config 运行标注流水线，创建新的 lerobot 格式数据集：
  - data/chunk-*/episode_*.parquet  alignment 列 + annotation 列
  - meta/                           从原始复制 + info.json 更新 features
  - videos/                         symlink 到原始视频文件

默认只含 annotation 列；加 --include_original 可将原始 state/action 等列一并复制。

用法：
    python scripts/create_lerobot_with_anno.py \\
        --config src/caliball/config/ucsd_kitchen.yaml \\
        --output_dir /tmp/ucsd_kitchen_anno

    # 包含原始数据集的 state/action 列
    python scripts/create_lerobot_with_anno.py \\
        --config src/caliball/config/ucsd_kitchen.yaml \\
        --output_dir /tmp/ucsd_kitchen_anno \\
        --include_original

    # 断点续标
    python scripts/create_lerobot_with_anno.py ... --resume
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
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
from src.caliball.label import Labeler
from src.caliball.pipeline.label_data import LabelData
from src.caliball.utils.mesh_loader import load_meshes
from src.caliball.utils.nvdiffrast_renderer import NVDiffrastRenderer

# ──────────────────────────────────────────────────────────────────────────────
# Annotation column definitions
# ──────────────────────────────────────────────────────────────────────────────

ARRAY_FIELDS: List[tuple] = [
    ("uv",                    2),
    ("uvd",                   3),
    ("xyz_euler_g",           7),
    ("xyz_quat_g",            8),
    ("xyz_mat_g",            13),
    ("bbox_with_gripper",     4),
    ("bbox_without_gripper",  4),
    ("bbox_gripper",          4),
]
MASK_FIELDS: List[str] = [
    "mask_with_gripper",
    "mask_without_gripper",
    "mask_gripper",
]

# Alignment columns carried over from original parquet for row-level join
ALIGN_COLS = ["episode_index", "frame_index", "index", "timestamp", "task_index"]

# lerobot load_info schema for alignment columns
_ALIGN_INFO_SCHEMA = {
    "episode_index": {"dtype": "int64",   "shape": [1], "names": None},
    "frame_index":   {"dtype": "int64",   "shape": [1], "names": None},
    "index":         {"dtype": "int64",   "shape": [1], "names": None},
    "timestamp":     {"dtype": "float32", "shape": [1], "names": None},
    "task_index":    {"dtype": "int64",   "shape": [1], "names": None},
}


def _col(cam: str, arm: str, field: str) -> str:
    return f"annotation.{cam}.{arm}.{field}"


def _rle_to_str(rle: Optional[dict]) -> str:
    if rle is None:
        return ""
    counts = rle.get("counts", "")
    if isinstance(counts, (bytes, bytearray)):
        counts = counts.decode("ascii")
    return json.dumps({"size": rle["size"], "counts": counts}, ensure_ascii=True)


def _bbox_or_zero(bbox) -> list:
    return list(bbox) if bbox is not None else [0, 0, 0, 0]


# ──────────────────────────────────────────────────────────────────────────────
# LabelData → annotation columns
# ──────────────────────────────────────────────────────────────────────────────

def build_annotation_columns(label_data: LabelData) -> Dict[str, list]:
    cols: Dict[str, list] = {}
    for cam, frames in label_data.cameras.items():
        T = len(frames)
        if T == 0:
            continue
        for arm_name in label_data.arm_names:
            for field, _ in ARRAY_FIELDS:
                cols[_col(cam, arm_name, field)] = [None] * T
            for field in MASK_FIELDS:
                cols[_col(cam, arm_name, field)] = [""] * T

        for t, frame in enumerate(frames):
            for arm_name, arm_label in frame.arms.items():
                if arm_label.is_placeholder:
                    continue
                cols[_col(cam, arm_name, "uv")][t]                   = list(map(float, arm_label.uv))
                cols[_col(cam, arm_name, "uvd")][t]                  = list(map(float, arm_label.uvd))
                cols[_col(cam, arm_name, "xyz_euler_g")][t]          = list(map(float, arm_label.xyz_euler_g))
                cols[_col(cam, arm_name, "xyz_quat_g")][t]           = list(map(float, arm_label.xyz_quat_g))
                cols[_col(cam, arm_name, "xyz_mat_g")][t]            = list(map(float, arm_label.xyz_mat_g))
                cols[_col(cam, arm_name, "bbox_with_gripper")][t]    = _bbox_or_zero(arm_label.bbox_with_gripper)
                cols[_col(cam, arm_name, "bbox_without_gripper")][t] = _bbox_or_zero(arm_label.bbox_without_gripper)
                cols[_col(cam, arm_name, "bbox_gripper")][t]         = _bbox_or_zero(arm_label.bbox_gripper)
                cols[_col(cam, arm_name, "mask_with_gripper")][t]    = _rle_to_str(arm_label.mask_with_gripper)
                cols[_col(cam, arm_name, "mask_without_gripper")][t] = _rle_to_str(arm_label.mask_without_gripper)
                cols[_col(cam, arm_name, "mask_gripper")][t]         = _rle_to_str(arm_label.mask_gripper)

    # fill remaining None with zeros
    for col, vals in cols.items():
        fill = None
        for v in vals:
            if v is not None:
                fill = [0.0] * len(v) if isinstance(v, list) else ""
                break
        cols[col] = [fill if v is None else v for v in vals] if fill is not None else vals
    return cols


# ──────────────────────────────────────────────────────────────────────────────
# Parquet write
# ──────────────────────────────────────────────────────────────────────────────

def write_episode_parquet(
    output_dir: Path,
    ep_idx: int,
    align_table: pa.Table,
    ann_cols: Dict[str, list],
    chunks_size: int,
) -> None:
    chunk = ep_idx // chunks_size
    out_path = output_dir / "data" / f"chunk-{chunk:03d}" / f"episode_{ep_idx:06d}.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    table = align_table  # already filtered to ALIGN_COLS
    n_rows = len(table)

    for col_name, values in ann_cols.items():
        if len(values) != n_rows:
            print(f"[WARN] {col_name} length {len(values)} != {n_rows}, skip")
            continue
        sample = values[0] if values else None
        if isinstance(sample, list):
            flat = [x for row in values for x in row]
            arr = pa.FixedSizeListArray.from_arrays(
                pa.array(flat, type=pa.float32()), len(sample)
            )
        else:
            arr = pa.array(values, type=pa.large_utf8())
        table = table.append_column(col_name, arr)

    # HF metadata for parquet schema (datasets 4.x format)
    hf_feats: dict = {}
    for col in align_table.schema.names:
        dt = align_table.schema.field(col).type
        hf_feats[col] = {"dtype": str(dt), "_type": "Value"}
    for col_name, values in ann_cols.items():
        s = values[0] if values else None
        if isinstance(s, list):
            hf_feats[col_name] = {
                "_type": "Sequence",
                "feature": {"dtype": "float32", "_type": "Value"},
                "length": len(s),
            }
        else:
            hf_feats[col_name] = {"dtype": "string", "_type": "Value"}

    hf_meta = json.dumps({"info": {"features": hf_feats}}, ensure_ascii=False).encode()
    table = table.replace_schema_metadata({b"huggingface": hf_meta})

    tmp = out_path.with_suffix(".tmp.parquet")
    pq.write_table(table, tmp, compression="zstd")
    tmp.replace(out_path)


# ──────────────────────────────────────────────────────────────────────────────
# Per-episode stats
# ──────────────────────────────────────────────────────────────────────────────

def compute_episode_stats(ann_cols: Dict[str, list]) -> dict:
    # Annotation stats are not required by lerobot for data loading, and
    # lerobot's aggregate_stats enforces shape (3,1,1) for any feature key
    # containing "image", which conflicts with annotation array shapes.
    # Return empty dict to keep episodes_stats.jsonl valid but stat-free.
    return {}


# ──────────────────────────────────────────────────────────────────────────────
# Meta
# ──────────────────────────────────────────────────────────────────────────────

def write_meta(
    output_dir: Path,
    orig_root: Path,
    ann_cols: Dict[str, list],           # sample ann_cols from one episode (for schema)
    align_cols_present: List[str],
    episode_stats_list: List[dict],
    n_episodes: Optional[int] = None,
    include_original: bool = False,
) -> None:
    meta_out = output_dir / "meta"
    meta_out.mkdir(exist_ok=True)

    # Copy episodes.jsonl and tasks.jsonl verbatim
    for fname in ("episodes.jsonl", "tasks.jsonl"):
        src = orig_root / "meta" / fname
        if src.exists():
            shutil.copy2(src, meta_out / fname)

    # info.json: alignment features (lerobot load_info format) + annotation features
    orig_info = json.loads((orig_root / "meta" / "info.json").read_text())
    new_feats: dict = {}
    if include_original:
        # Keep all original features
        new_feats.update(orig_info.get("features", {}))
    else:
        # Only keep video/image type features
        for col, feat in orig_info.get("features", {}).items():
            if feat.get("dtype") in ("video", "image"):
                new_feats[col] = feat
    for col in align_cols_present:
        if col in _ALIGN_INFO_SCHEMA:
            new_feats[col] = _ALIGN_INFO_SCHEMA[col]
    for col, vals in ann_cols.items():
        s = vals[0] if vals else None
        if isinstance(s, list):
            new_feats[col] = {"dtype": "float32", "shape": [len(s)], "names": None}
        else:
            new_feats[col] = {"dtype": "string",  "shape": [],       "names": None}

    # total_frames: stats are always empty, so read directly from orig info.
    total_frames = orig_info.get("total_frames", 0)
    total_eps = n_episodes if n_episodes is not None else len(episode_stats_list)

    new_info = {**orig_info, "features": new_feats,
                "total_episodes": total_eps,
                "total_frames": total_frames}
    (meta_out / "info.json").write_text(json.dumps(new_info, indent=4, ensure_ascii=False))

    # episodes_stats.jsonl
    with open(meta_out / "episodes_stats.jsonl", "w") as f:
        for s in episode_stats_list:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")


# ──────────────────────────────────────────────────────────────────────────────
# Video symlinks
# ──────────────────────────────────────────────────────────────────────────────

def symlink_videos(output_dir: Path, orig_root: Path, ep_indices: List[int]) -> None:
    orig_vid = orig_root / "videos"
    if not orig_vid.is_dir():
        return
    out_vid = output_dir / "videos"
    for cam_chunk_dir in orig_vid.glob("chunk-*/*"):
        out_cam = out_vid / cam_chunk_dir.relative_to(orig_vid)
        out_cam.mkdir(parents=True, exist_ok=True)
        for ep_idx in ep_indices:
            src = cam_chunk_dir / f"episode_{ep_idx:06d}.mp4"
            if src.exists():
                link = out_cam / src.name
                if not link.exists():
                    link.symlink_to(src.resolve())


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="标注流水线 → 新 lerobot 数据集（仅 annotation）")
    p.add_argument("--config",       required=True,  help="任务 YAML")
    p.add_argument("--output_dir",   required=True,  help="新数据集输出目录")
    p.add_argument("--base_path",    default=None,   help="覆盖 YAML base_path")
    p.add_argument("--dataset_name", default=None,   help="覆盖 YAML dataset_name")
    p.add_argument("--episode_start",type=int, default=None)
    p.add_argument("--max_episodes", type=int, default=None)
    p.add_argument("--device",       default=None)
    p.add_argument("--skip_mask",    action="store_true")
    p.add_argument("--arm_mesh_num", type=int, default=None)
    p.add_argument("--resume",       action="store_true", help="跳过已存在的 parquet")
    p.add_argument("--include_original", action="store_true",
                   help="将原始数据集的 state/action 等列也复制到新数据集（默认只含 annotation）")
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

    calib_name   = str(cfg.get("calib_dataset_name") or cfg.get("dataset_name") or "")
    task_name    = str(cfg.get("dataset_name", "task"))
    camera_names = label_over.get("camera_names") or ["image"]
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

    output_dir = Path(os.path.expanduser(args.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)

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
    dataset  = instantiate_dataset(cfg)
    orig_root = Path(dataset.lerobot_ds.root)
    chunks_size = dataset.lerobot_ds.meta.chunks_size

    # Detect which ALIGN_COLS exist in original
    sample_pq_path = orig_root / dataset.lerobot_ds.meta.get_data_file_path(0)
    orig_schema = pq.read_schema(sample_pq_path)
    align_cols_present = [c for c in ALIGN_COLS if c in orig_schema.names]

    # If --include_original, copy all non-video columns from original parquet
    if args.include_original:
        orig_copy_cols = [c for c in orig_schema.names if not c.startswith("annotation")]
    else:
        orig_copy_cols = align_cols_present

    # Load meshes
    if not skip_mask:
        print("[INFO] 加载 robot mesh_paths...")
        vertices_list, faces_list = load_meshes(cfg.robot, device, project_root=_PROJECT_ROOT)
    else:
        vertices_list, faces_list = [], []

    labeler = Labeler(config=None)

    n_total = len(dataset)
    ep_end  = min(n_total, ep_start + int(max_ep)) if max_ep else n_total
    print(f"[INFO] 处理 episode [{ep_start}, {ep_end})，共 {ep_end - ep_start} 个")

    episode_stats_list: List[dict] = []
    processed_eps: List[int] = []
    sample_ann_cols: Optional[Dict[str, list]] = None   # schema reference

    for ep_idx in tqdm(range(ep_start, ep_end), desc="Episodes"):
        chunk   = ep_idx // chunks_size
        out_pq  = output_dir / "data" / f"chunk-{chunk:03d}" / f"episode_{ep_idx:06d}.parquet"

        if args.resume and out_pq.exists():
            processed_eps.append(ep_idx)
            continue

        # Load episode data from original dataset
        data         = dataset[ep_idx]
        joint_angles = data.get("states")
        videos_data  = data.get("videos") or {}

        if joint_angles is None:
            print(f"[WARN] episode {ep_idx} 缺少 states，跳过")
            continue

        # Read columns from original parquet
        orig_pq = orig_root / dataset.lerobot_ds.meta.get_data_file_path(ep_idx)
        align_table = pq.read_table(orig_pq, columns=orig_copy_cols)

        # Run labeling
        label_data = LabelData(
            dataset_name=task_name,
            episode_id=str(ep_idx),
            arm_names=arm_names,
        )

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

        # Build annotation cols and write parquet
        ann_cols = build_annotation_columns(label_data)
        write_episode_parquet(output_dir, ep_idx, align_table, ann_cols, chunks_size)

        # Track for meta/stats
        if sample_ann_cols is None:
            sample_ann_cols = ann_cols
        ep_stats = compute_episode_stats(ann_cols)
        episode_stats_list.append({"episode_index": ep_idx, "stats": ep_stats})
        processed_eps.append(ep_idx)

    if not processed_eps:
        print("[WARN] 没有处理任何 episode，退出")
        return

    if sample_ann_cols is None:
        # resume mode: infer schema from first written parquet
        first_pq = output_dir / "data" / f"chunk-{processed_eps[0] // chunks_size:03d}" / \
                   f"episode_{processed_eps[0]:06d}.parquet"
        schema = pq.read_schema(first_pq)
        sample_ann_cols = {}
        for c in schema.names:
            if not c.startswith("annotation"):
                continue
            field = schema.field(c)
            if pa.types.is_fixed_size_list(field.type):
                size = field.type.list_size
                sample_ann_cols[c] = [[0.0] * size]
            else:
                sample_ann_cols[c] = [""]

    # Write meta
    write_meta(output_dir, orig_root, sample_ann_cols, align_cols_present,
               episode_stats_list, n_episodes=len(processed_eps),
               include_original=args.include_original)

    # Write calibration.json
    calib: dict = {}
    for cam_name in camera_names:
        try:
            K, T = get_intrinsic_extrinsic(calib_name, task_name, cam_name, 0)
            calib[cam_name] = {"intrinsic": K.tolist(), "extrinsic": T.tolist()}
        except Exception:
            pass
    if calib:
        (output_dir / "meta" / "calibration.json").write_text(
            json.dumps(calib, indent=2, ensure_ascii=False)
        )

    # Symlink videos
    symlink_videos(output_dir, orig_root, processed_eps)

    print(f"\n[DONE] 输出: {output_dir}")
    print(f"  episodes : {len(processed_eps)}")
    print(f"  ann cols : {len(sample_ann_cols)}")
    print(f"  videos → : {orig_root}/videos/")


if __name__ == "__main__":
    main()
