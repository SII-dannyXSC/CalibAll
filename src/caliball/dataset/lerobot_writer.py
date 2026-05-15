"""
LeRobot 数据集写入器。

将任意列数据写入 LeRobot 2.1 格式：
  - data/chunk-NNN/episode_NNNNNN.parquet
  - meta/ (info.json, episodes.jsonl, tasks.jsonl, episodes_stats.jsonl)
  - videos/ (symlink)
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional

import pyarrow as pa
import pyarrow.parquet as pq


ALIGN_COLS = ["episode_index", "frame_index", "index", "timestamp", "task_index"]

_ALIGN_INFO_SCHEMA = {
    "episode_index": {"dtype": "int64",   "shape": [1], "names": None},
    "frame_index":   {"dtype": "int64",   "shape": [1], "names": None},
    "index":         {"dtype": "int64",   "shape": [1], "names": None},
    "timestamp":     {"dtype": "float32", "shape": [1], "names": None},
    "task_index":    {"dtype": "int64",   "shape": [1], "names": None},
}


class LeRobotDatasetWriter:
    """LeRobot 2.1 格式数据集写入器。

    Parameters
    ----------
    output_dir : 输出根目录
    orig_root : 原始数据集根目录（用于复制 meta 和 symlink videos）
    chunks_size : parquet chunk 大小（episode 分片依据）
    include_original : 是否复制原始数据集的非视频列
    """

    def __init__(
        self,
        output_dir: str | Path,
        orig_root: str | Path,
        chunks_size: int,
        include_original: bool = False,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.orig_root = Path(orig_root)
        self.chunks_size = chunks_size
        self.include_original = include_original
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._episode_stats: List[dict] = []
        self._processed_eps: List[int] = []
        self._sample_columns: Optional[Dict[str, list]] = None

    @property
    def processed_episodes(self) -> List[int]:
        return list(self._processed_eps)

    def episode_exists(self, ep_idx: int) -> bool:
        chunk = ep_idx // self.chunks_size
        path = self.output_dir / "data" / f"chunk-{chunk:03d}" / f"episode_{ep_idx:06d}.parquet"
        return path.exists()

    def write_episode(
        self,
        ep_idx: int,
        align_table: pa.Table,
        columns: Dict[str, list],
    ) -> None:
        chunk = ep_idx // self.chunks_size
        out_path = self.output_dir / "data" / f"chunk-{chunk:03d}" / f"episode_{ep_idx:06d}.parquet"
        out_path.parent.mkdir(parents=True, exist_ok=True)

        table = align_table
        n_rows = len(table)

        for col_name, values in columns.items():
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

        hf_feats: dict = {}
        for col in align_table.schema.names:
            dt = align_table.schema.field(col).type
            hf_feats[col] = {"dtype": str(dt), "_type": "Value"}
        for col_name, values in columns.items():
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

        if self._sample_columns is None:
            self._sample_columns = columns
        self._episode_stats.append({"episode_index": ep_idx, "stats": {}})
        self._processed_eps.append(ep_idx)

    def write_meta(
        self,
        align_cols_present: Optional[List[str]] = None,
    ) -> None:
        if not self._processed_eps:
            return

        if align_cols_present is None:
            align_cols_present = ALIGN_COLS

        meta_out = self.output_dir / "meta"
        meta_out.mkdir(exist_ok=True)

        for fname in ("episodes.jsonl", "tasks.jsonl"):
            src = self.orig_root / "meta" / fname
            if src.exists():
                shutil.copy2(src, meta_out / fname)

        orig_info = json.loads((self.orig_root / "meta" / "info.json").read_text())
        new_feats: dict = {}
        if self.include_original:
            new_feats.update(orig_info.get("features", {}))
        else:
            for col, feat in orig_info.get("features", {}).items():
                if feat.get("dtype") in ("video", "image"):
                    new_feats[col] = feat
        for col in align_cols_present:
            if col in _ALIGN_INFO_SCHEMA:
                new_feats[col] = _ALIGN_INFO_SCHEMA[col]

        sample = self._sample_columns
        if sample is None:
            first_ep = self._processed_eps[0]
            chunk = first_ep // self.chunks_size
            first_pq = (self.output_dir / "data" / f"chunk-{chunk:03d}" /
                        f"episode_{first_ep:06d}.parquet")
            schema = pq.read_schema(first_pq)
            sample = {}
            for c in schema.names:
                if c in ALIGN_COLS:
                    continue
                fld = schema.field(c)
                if pa.types.is_fixed_size_list(fld.type):
                    sample[c] = [[0.0] * fld.type.list_size]
                else:
                    sample[c] = [""]

        for col, vals in sample.items():
            s = vals[0] if vals else None
            if isinstance(s, list):
                new_feats[col] = {"dtype": "float32", "shape": [len(s)], "names": None}
            else:
                new_feats[col] = {"dtype": "string", "shape": [], "names": None}

        new_info = {
            **orig_info,
            "features": new_feats,
            "total_episodes": len(self._processed_eps),
            "total_frames": orig_info.get("total_frames", 0),
        }
        (meta_out / "info.json").write_text(
            json.dumps(new_info, indent=4, ensure_ascii=False)
        )

        with open(meta_out / "episodes_stats.jsonl", "w") as f:
            for s in self._episode_stats:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")

    def symlink_videos(self, ep_indices: Optional[List[int]] = None) -> None:
        if ep_indices is None:
            ep_indices = self._processed_eps
        orig_vid = self.orig_root / "videos"
        if not orig_vid.is_dir():
            return
        out_vid = self.output_dir / "videos"
        for cam_chunk_dir in orig_vid.glob("chunk-*/*"):
            out_cam = out_vid / cam_chunk_dir.relative_to(orig_vid)
            out_cam.mkdir(parents=True, exist_ok=True)
            for ep_idx in ep_indices:
                src = cam_chunk_dir / f"episode_{ep_idx:06d}.mp4"
                if src.exists():
                    link = out_cam / src.name
                    if not link.exists():
                        link.symlink_to(src.resolve())

    def write_calibration(self, calib: dict) -> None:
        if not calib:
            return
        meta_out = self.output_dir / "meta"
        meta_out.mkdir(exist_ok=True)
        (meta_out / "calibration.json").write_text(
            json.dumps(calib, indent=2, ensure_ascii=False)
        )

    def finalize(self, align_cols_present: Optional[List[str]] = None) -> None:
        self.write_meta(align_cols_present)
        self.symlink_videos()
