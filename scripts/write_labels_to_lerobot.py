#!/usr/bin/env python3
"""
write_labels_to_lerobot.py

从 label_out/visual_whole 读取 LabelData (pkl)，将标注写回 LeRobot 数据集（in-place）：
  - data/chunk-*/episode_*.parquet  追加 annotation.* 列
  - meta/calibration.json           写入相机 K / T（幂等）
  - meta/info.json                  追加 features 声明（幂等）

列命名规则：
  annotation.{cam_full}.{arm}.{field}
  e.g.  annotation.observation.images.camera_top.left.uv       (T, 2)
        annotation.observation.images.camera_top.left.mask_with_gripper  (T,) str

is_placeholder=True 的臂不写入。
幂等：若目标 parquet 中已有 annotation.* 列则跳过该 episode。

用法：
    # 批量（处理 label_out/visual_whole 下所有已完成任务）
    python scripts/write_labels_to_lerobot.py

    # 单任务调试
    python scripts/write_labels_to_lerobot.py \\
        --label_dir  label_out/visual_whole/ur_1rgb/benchmark1_0_compressed/bread_in_basket_1 \\
        --dataset_dir data/caliball_data/RoboMIND_lerobot_v2.1/benchmark1_0_compressed/ur_1rgb/bread_in_basket_1

    # dry-run
    python scripts/write_labels_to_lerobot.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.caliball.pipeline.label_data import ArmLabel, LabelData

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

LABEL_ROOT = _PROJECT_ROOT / "label_out" / "visual_whole"

# Scalar / array annotation fields from ArmLabel (in order)
ARRAY_FIELDS: List[tuple[str, int]] = [
    ("uv",            2),
    ("uvd",           3),
    ("xyz_euler_g",  7),
    ("xyz_quat_g",   8),
    ("xyz_mat_g",   13),
    ("bbox_with_gripper",      4),
    ("bbox_without_gripper",   4),
    ("bbox_gripper",           4),
]
MASK_FIELDS: List[str] = [
    "mask_with_gripper",
    "mask_without_gripper",
    "mask_gripper",
]

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _col(cam: str, arm: str, field: str) -> str:
    return f"annotation.{cam}.{arm}.{field}"


def _rle_to_str(rle: Optional[dict]) -> str:
    """COCO RLE dict → JSON string。counts 兼容 bytes/str。"""
    if rle is None:
        return ""
    counts = rle.get("counts", "")
    if isinstance(counts, (bytes, bytearray)):
        counts = counts.decode("ascii")
    return json.dumps({"size": rle["size"], "counts": counts}, ensure_ascii=True)


def _bbox_or_zero(bbox: Optional[list]) -> list:
    return list(bbox) if bbox is not None else [0, 0, 0, 0]


# ──────────────────────────────────────────────────────────────────────────────
# Build annotation columns from LabelData
# ──────────────────────────────────────────────────────────────────────────────

def build_annotation_columns(
    label_data: LabelData,
) -> Dict[str, Any]:
    """
    返回 {col_name: list_of_T_values}，供追加到 parquet。
    array 字段每行是 Python list；mask 字段每行是 str。
    """
    cols: Dict[str, list] = {}

    for cam, frames in label_data.cameras.items():
        T = len(frames)
        if T == 0:
            continue

        # 初始化所有列
        for arm_name in label_data.arm_names:
            for field, dim in ARRAY_FIELDS:
                cols[_col(cam, arm_name, field)] = [None] * T
            for field in MASK_FIELDS:
                cols[_col(cam, arm_name, field)] = [""] * T

        for t, frame in enumerate(frames):
            for arm_name, arm_label in frame.arms.items():
                if arm_label.is_placeholder:
                    continue
                # ── array fields ──────────────────────────────────────────
                cols[_col(cam, arm_name, "uv")][t]           = list(map(float, arm_label.uv))
                cols[_col(cam, arm_name, "uvd")][t]          = list(map(float, arm_label.uvd))
                cols[_col(cam, arm_name, "xyz_euler_g")][t]  = list(map(float, arm_label.xyz_euler_g))
                cols[_col(cam, arm_name, "xyz_quat_g")][t]   = list(map(float, arm_label.xyz_quat_g))
                cols[_col(cam, arm_name, "xyz_mat_g")][t]    = list(map(float, arm_label.xyz_mat_g))
                cols[_col(cam, arm_name, "bbox_with_gripper")][t]    = _bbox_or_zero(arm_label.bbox_with_gripper)
                cols[_col(cam, arm_name, "bbox_without_gripper")][t] = _bbox_or_zero(arm_label.bbox_without_gripper)
                cols[_col(cam, arm_name, "bbox_gripper")][t]         = _bbox_or_zero(arm_label.bbox_gripper)
                # ── mask fields ───────────────────────────────────────────
                cols[_col(cam, arm_name, "mask_with_gripper")][t]    = _rle_to_str(arm_label.mask_with_gripper)
                cols[_col(cam, arm_name, "mask_without_gripper")][t] = _rle_to_str(arm_label.mask_without_gripper)
                cols[_col(cam, arm_name, "mask_gripper")][t]         = _rle_to_str(arm_label.mask_gripper)

    # Fill remaining None with zero lists (for placeholder arms that were skipped)
    for col, vals in cols.items():
        if not vals:
            continue
        # Infer fill value from first non-None
        fill: Any = None
        for v in vals:
            if v is not None:
                fill = [0.0] * len(v) if isinstance(v, list) else ""
                break
        if fill is None:
            fill = ""
        cols[col] = [fill if v is None else v for v in vals]

    return cols


# ──────────────────────────────────────────────────────────────────────────────
# Feature schema helpers
# ──────────────────────────────────────────────────────────────────────────────

def _feature_entry_parquet(dim: int) -> dict:
    """parquet HF schema metadata 用的格式（datasets 4.x generate_from_dict 兼容）。"""
    return {
        "_type": "Sequence",
        "feature": {"dtype": "float32", "_type": "Value"},
        "length": dim,
    }


def _feature_entry_info(dim: int) -> dict:
    """info.json 用的格式（lerobot load_info 期望 shape/dtype/names）。"""
    return {"dtype": "float32", "shape": [dim], "names": None}


def build_parquet_feature_declarations(label_data: LabelData) -> dict:
    """生成写入 parquet HF schema metadata 的 features 字典（datasets 4.x 格式）。"""
    features: dict = {}
    for cam in label_data.camera_names:
        for arm_name in label_data.arm_names:
            for field, dim in ARRAY_FIELDS:
                features[_col(cam, arm_name, field)] = _feature_entry_parquet(dim)
            for field in MASK_FIELDS:
                features[_col(cam, arm_name, field)] = {
                    "dtype": "string", "_type": "Value"
                }
    return features


def build_info_feature_declarations(label_data: LabelData) -> dict:
    """生成写入 info.json features 的声明字典（lerobot load_info 格式）。"""
    features: dict = {}
    for cam in label_data.camera_names:
        for arm_name in label_data.arm_names:
            for field, dim in ARRAY_FIELDS:
                features[_col(cam, arm_name, field)] = _feature_entry_info(dim)
            for field in MASK_FIELDS:
                features[_col(cam, arm_name, field)] = {
                    "dtype": "string", "shape": [], "names": None
                }
    return features


def build_feature_declarations(label_data: LabelData) -> dict:
    """向后兼容别名，返回 parquet 格式声明。"""
    return build_parquet_feature_declarations(label_data)


# ──────────────────────────────────────────────────────────────────────────────
# Parquet update (atomic)
# ──────────────────────────────────────────────────────────────────────────────

def _parquet_path(dataset_dir: Path, episode_idx: int, chunks_size: int = 1000) -> Path:
    chunk = episode_idx // chunks_size
    return dataset_dir / "data" / f"chunk-{chunk:03d}" / f"episode_{episode_idx:06d}.parquet"


def _already_annotated(table: pa.Table) -> bool:
    return any(c.startswith("annotation.") for c in table.schema.names)


def _update_hf_metadata(existing_meta: bytes, new_features: dict) -> bytes:
    """更新 parquet schema 里 huggingface 元数据的 features 字段。"""
    try:
        meta = json.loads(existing_meta.decode("utf-8"))
        info = meta.get("info", {})
        info.setdefault("features", {}).update(new_features)
        meta["info"] = info
        return json.dumps(meta, ensure_ascii=False).encode("utf-8")
    except Exception:
        return existing_meta


def write_annotation_to_parquet(
    dataset_dir: Path,
    episode_idx: int,
    ann_cols: Dict[str, Any],
    new_features: dict,
) -> bool:
    """
    追加标注列到指定 parquet。
    返回 True=写入成功，False=已存在跳过。
    """
    parquet_path = _parquet_path(dataset_dir, episode_idx)
    if not parquet_path.exists():
        raise FileNotFoundError(f"parquet 不存在: {parquet_path}")

    table = pq.read_table(parquet_path)
    if _already_annotated(table):
        return False   # 已有标注，跳过

    n_rows = len(table)

    # 构建新列的 pyarrow arrays
    new_arrays: list[tuple[str, pa.Array]] = []
    for col_name, values in ann_cols.items():
        if len(values) != n_rows:
            logging.warning(f"列 {col_name} 长度 {len(values)} ≠ 行数 {n_rows}，跳过")
            continue
        sample = values[0] if values else None
        if isinstance(sample, list):
            # fixed-size list of float32
            flat = [x for row in values for x in (row if row else [0.0] * len(sample))]
            arr = pa.FixedSizeListArray.from_arrays(
                pa.array(flat, type=pa.float32()), len(sample)
            )
        else:
            # string
            arr = pa.array(values, type=pa.large_utf8())
        new_arrays.append((col_name, arr))

    # 追加到 table
    for col_name, arr in new_arrays:
        table = table.append_column(col_name, arr)

    # 更新 schema metadata（huggingface 字段）
    existing_meta = table.schema.metadata or {}
    hf_bytes = existing_meta.get(b"huggingface", b"{}")
    updated_hf = _update_hf_metadata(hf_bytes, new_features)
    new_meta = {**existing_meta, b"huggingface": updated_hf}
    table = table.replace_schema_metadata(new_meta)

    # 原子写（先写临时文件，再 rename）
    tmp = parquet_path.with_suffix(".tmp.parquet")
    pq.write_table(table, tmp, compression="zstd")
    tmp.replace(parquet_path)
    return True


# ──────────────────────────────────────────────────────────────────────────────
# meta/calibration.json & meta/info.json
# ──────────────────────────────────────────────────────────────────────────────

def write_calibration_json(
    dataset_dir: Path,
    episode_json: dict,
) -> None:
    """将 episode JSON 里的 intrinsic/extrinsic 写入 meta/calibration.json（幂等）。"""
    calib_path = dataset_dir / "meta" / "calibration.json"
    existing: dict = {}
    if calib_path.exists():
        with open(calib_path) as f:
            existing = json.load(f)

    changed = False
    for cam_name, cam_data in episode_json.items():
        if cam_name not in existing:
            existing[cam_name] = {
                "intrinsic": cam_data.get("intrinsic"),
                "extrinsic": cam_data.get("extrinsic"),
            }
            changed = True

    if changed:
        with open(calib_path, "w") as f:
            json.dump(existing, f, indent=2, ensure_ascii=False)


def update_info_json(dataset_dir: Path, new_features: dict) -> None:
    """在 meta/info.json 的 features 里追加新声明（幂等）。"""
    info_path = dataset_dir / "meta" / "info.json"
    if not info_path.exists():
        return
    with open(info_path) as f:
        info = json.load(f)
    feats = info.setdefault("features", {})
    changed = False
    for k, v in new_features.items():
        if k not in feats:
            feats[k] = v
            changed = True
    if changed:
        with open(info_path, "w") as f:
            json.dump(info, f, indent=4, ensure_ascii=False)


# ──────────────────────────────────────────────────────────────────────────────
# Per-task processing
# ──────────────────────────────────────────────────────────────────────────────

def process_task(label_dir: Path, dataset_dir: Path, dry_run: bool = False) -> str:
    """
    处理单个 task 目录。
    返回状态字符串：OK / SKIP / FAIL:<reason>
    """
    label_data_dir = label_dir / "label_data"
    if not label_data_dir.exists():
        return "SKIP:no_label_data_dir"

    pkl_files = sorted(label_data_dir.glob("episode_*.pkl"))
    if not pkl_files:
        return "SKIP:no_pkl"

    json_dir = label_dir  # episode_*.json 与 label_data/ 同级

    written = 0
    skipped = 0
    for pkl_path in pkl_files:
        ep_stem = pkl_path.stem        # episode_000000
        ep_idx = int(ep_stem.split("_")[1])

        # 加载 pkl
        try:
            label_data: LabelData = LabelData.load(pkl_path)
        except Exception as e:
            return f"FAIL:load_pkl:{e}"

        # 加载对应 episode JSON（含 intrinsic/extrinsic）
        json_path = json_dir / f"{ep_stem}.json"
        episode_json: dict = {}
        if json_path.exists():
            with open(json_path) as f:
                episode_json = json.load(f)

        ann_cols        = build_annotation_columns(label_data)
        new_feats       = build_parquet_feature_declarations(label_data)  # for parquet HF metadata
        new_info_feats  = build_info_feature_declarations(label_data)     # for info.json

        if dry_run:
            logging.info(f"  [DRY] {label_dir.name}/{ep_stem}: "
                         f"{len(ann_cols)} 列, episode→{dataset_dir}")
            written += 1
            continue

        # 写 parquet
        try:
            did_write = write_annotation_to_parquet(
                dataset_dir, ep_idx, ann_cols, new_feats
            )
        except FileNotFoundError as e:
            return f"FAIL:parquet_missing:{e}"
        except Exception as e:
            return f"FAIL:parquet_write:{e}"

        if did_write:
            written += 1
        else:
            skipped += 1

        # 写 calibration + info（任意 episode 写一次即可）
        if episode_json:
            try:
                write_calibration_json(dataset_dir, episode_json)
            except Exception as e:
                logging.warning(f"写 calibration.json 失败: {e}")
        try:
            update_info_json(dataset_dir, new_info_feats)
        except Exception as e:
            logging.warning(f"更新 info.json 失败: {e}")

    return f"OK:written={written},skipped={skipped}"


# ──────────────────────────────────────────────────────────────────────────────
# Task enumeration（复用 label_visual_whole 的路径逻辑）
# ──────────────────────────────────────────────────────────────────────────────

def iter_labeled_tasks():
    """
    扫描 label_out/visual_whole，yield (label_dir, dataset_dir) 对。
    依赖与 label_visual_whole.py 相同的目录结构。
    """
    data_base  = _PROJECT_ROOT / "data" / "caliball_data"
    label_root = LABEL_ROOT

    # ── OXE flat ──────────────────────────────────────────────────────────────
    OXE = {
        "berkeley_autolab_ur5":    data_base / "berkeley_autolab_ur5",
        "kaist_nonprehensile":     data_base / "kaist_nonprehensile_converted_externally_to_rlds",
        "nyu_franka":              data_base / "nyu_franka_play_dataset_converted_externally_to_rlds",
        "toto":                    data_base / "toto",
        "ucsd_kitchen":            data_base / "ucsd_kitchen_dataset_converted_externally_to_rlds",
    }
    for group, ds_path in OXE.items():
        label_dir = label_root / group / ds_path.name
        if label_dir.exists():
            yield label_dir, ds_path

    robomind_root = data_base / "RoboMIND_lerobot_v2.1"

    for robot_type, config_yaml in [
        ("ur_1rgb",     "robomind_ur5e_1rgb"),
        ("agilex_3rgb", "robomind_aloha"),
        ("franka_3rgb", "robomind_franka"),
    ]:
        for bench_dir in sorted(robomind_root.glob("*/{}".format(robot_type))):
            if not bench_dir.parent.name.startswith("benchmark"):
                continue
            bench_name = bench_dir.parent.name
            label_bench = label_root / robot_type / bench_name
            if not label_bench.exists():
                continue
            for task_label_dir in sorted(label_bench.iterdir()):
                if not task_label_dir.is_dir():
                    continue
                ds_task = bench_dir / task_label_dir.name
                yield task_label_dir, ds_task

    # ── rdt_aloha ──────────────────────────────────────────────────────────────
    rdt_base = data_base / "rdt_aloha_lerobot2.1"
    rdt_label = label_root / "rdt_aloha"
    if rdt_label.exists():
        for task_label_dir in sorted(rdt_label.iterdir()):
            if task_label_dir.is_dir():
                yield task_label_dir, rdt_base / task_label_dir.name


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--label_dir",   type=str, default=None,
                   help="单任务：标注输出目录（label_out/visual_whole/.../task）")
    p.add_argument("--dataset_dir", type=str, default=None,
                   help="单任务：对应 LeRobot 数据集根目录")
    p.add_argument("--dry-run", action="store_true",
                   help="不实际写入，只打印计划")
    return p.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    if args.label_dir and args.dataset_dir:
        # 单任务模式
        pairs = [(Path(args.label_dir), Path(args.dataset_dir))]
    else:
        # 批量模式
        pairs = list(iter_labeled_tasks())
        logging.info(f"共找到 {len(pairs)} 个标注任务")

    ok = skip = fail = 0
    for label_dir, dataset_dir in pairs:
        status = process_task(label_dir, dataset_dir, dry_run=args.dry_run)
        tag = str(label_dir.resolve().relative_to(LABEL_ROOT)) if LABEL_ROOT in label_dir.resolve().parents or label_dir.resolve() == LABEL_ROOT else str(label_dir)
        if status.startswith("OK"):
            ok += 1
            logging.info(f"[OK  ] {tag}  {status}")
        elif status.startswith("SKIP"):
            skip += 1
            logging.debug(f"[SKIP] {tag}  {status}")
        else:
            fail += 1
            logging.warning(f"[FAIL] {tag}  {status}")

    logging.info(f"完成: ok={ok} skip={skip} fail={fail}")


if __name__ == "__main__":
    main()
