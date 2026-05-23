"""
LeRobot 2.1 数据集读取器
========================

直接读取本地 parquet + meta + video 文件，无 lerobot 库依赖。

数据集目录结构::

    dataset_root/
    ├── meta/
    │   ├── info.json
    │   └── episodes.jsonl
    ├── data/
    │   └── chunk-000/
    │       └── episode_000000.parquet
    └── videos/
        └── chunk-000/
            └── <video_key>/
                └── episode_000000.mp4
"""

from __future__ import annotations

import json
import re
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, List, Union

from caliball.dataset.state_processors import StateProcessor
from caliball.utils.video_io import decode_video_frames


class LeRobotDataset:
    """LeRobot 2.1 数据集读取器。

    Args:
        repo_id: 数据集本地目录路径
        episodes: 要加载的 episode 索引列表，None = 全部
        state_keys: 要读取的 parquet 列名
        state_processor: StateProcessor 实例，处理 columns → joint angles
        decode_video_keys: 要解码的视频 key 列表，None = 全部
        video_backend: 视频解码后端
    """

    def __init__(
        self,
        repo_id: str,
        episodes: Optional[List[int]] = None,
        state_keys: Union[str, List[str]] = "observation.state",
        state_processor: Optional[StateProcessor] = None,
        decode_video_keys: Optional[List[str]] = None,
        video_backend: str = "pyav",
    ) -> None:
        self.repo_id = repo_id
        if isinstance(state_keys, str):
            state_keys = [state_keys]
        self.state_keys = list(state_keys)
        self.state_processor = state_processor or StateProcessor()
        self.decode_video_keys = decode_video_keys
        self.video_backend = video_backend

        self._dataset_path = Path(repo_id)
        if not self._dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {self._dataset_path}")

        # --- 读取 meta/info.json ---
        info = self._load_json(self._dataset_path / "meta" / "info.json")
        self.fps: float = info.get("fps", 30)
        self.robot_type: str = info.get("robot_type", "unknown")
        self._chunks_size: int = info.get("chunks_size", 1000)
        self._data_path_pattern: str = info.get(
            "data_path",
            "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        )
        self._video_path_pattern: str = info.get("video_path", "")

        # 从 features 提取视频/相机 key
        features = info.get("features", {})
        self.video_keys: set[str] = set()
        self.camera_keys: list[str] = []
        for k, v in features.items():
            if isinstance(v, dict) and v.get("dtype") == "video":
                self.video_keys.add(k)
                self.camera_keys.append(k)

        # --- 读取 episodes ---
        episodes_info = self._load_episodes(episodes)
        self._episodes_info = episodes_info
        self.total_episodes = len(episodes_info)

        # episode 帧范围索引
        self._ep_from: list[int] = []
        self._ep_to: list[int] = []
        running = 0
        for ep in episodes_info:
            self._ep_from.append(running)
            running += ep["length"]
            self._ep_to.append(running)

        # parquet 缓存
        self._cached_ep_index: Optional[int] = None
        self._cached_df: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # 公共接口
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self.total_episodes

    def __getitem__(self, idx: int) -> dict:
        """返回 {"videos": {cam: (T,H,W,3)}, "states": (T, D)}。"""
        start, end = self._ep_from[idx], self._ep_to[idx]
        if start >= end:
            return dict(videos={}, states=None)

        batch = self._read_parquet_range(idx, start, end)
        ep_idx = int(batch["episode_index"].reshape(-1)[0])
        timestamps = batch["timestamp"].flatten().tolist()

        # 视频解码
        videos = {}
        tolerance = 1.0 / self.fps
        for vid_key in (self.decode_video_keys or self.video_keys):
            if vid_key not in self.video_keys:
                continue
            chunk = ep_idx // self._chunks_size
            video_path = self._dataset_path / self._video_path_pattern.format(
                episode_chunk=chunk, video_key=vid_key, episode_index=ep_idx,
            )
            videos[vid_key] = decode_video_frames(video_path, timestamps, tolerance, self.video_backend)

        # columns → state_processor
        columns = {}
        for sk in self.state_keys:
            if sk in batch:
                columns[sk] = np.asarray(batch[sk], dtype=np.float32)
        states = self.state_processor(columns) if columns else None

        return dict(videos=videos, states=states)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _load_episodes(self, episodes_filter: Optional[List[int]] = None) -> list:
        """读取 meta/episodes.jsonl，可选按 episode_index 过滤。"""
        path = self._dataset_path / "meta" / "episodes.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"episodes.jsonl not found: {path}")

        # 如果未指定 filter，扫描实际存在的 parquet 文件
        if episodes_filter is None:
            data_dir = self._dataset_path / "data"
            if data_dir.is_dir():
                available = set()
                for p in data_dir.rglob("episode_*.parquet"):
                    m = re.match(r"episode_(\d+)\.parquet$", p.name)
                    if m:
                        available.add(int(m.group(1)))
                if available:
                    episodes_filter = sorted(available)

        filter_set = set(episodes_filter) if episodes_filter else None
        episodes = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                ep = json.loads(line)
                if filter_set is not None and ep["episode_index"] not in filter_set:
                    continue
                episodes.append(ep)
        return episodes

    def _read_parquet_range(self, ep_local_idx: int, start: int, end: int) -> dict:
        """读取指定 episode 的 parquet 帧范围，返回 {列名: ndarray}。"""
        ep_info = self._episodes_info[ep_local_idx]
        ep_index = ep_info["episode_index"]

        # 带缓存的 parquet 加载
        if self._cached_ep_index != ep_index or self._cached_df is None:
            chunk = ep_index // self._chunks_size
            path = self._dataset_path / self._data_path_pattern.format(
                episode_chunk=chunk, episode_index=ep_index,
            )
            if not path.exists():
                raise FileNotFoundError(f"Parquet not found: {path}")
            self._cached_df = pd.read_parquet(path)
            self._cached_ep_index = ep_index

        local_start = start - self._ep_from[ep_local_idx]
        local_end = end - self._ep_from[ep_local_idx]
        sub = self._cached_df.iloc[local_start:local_end]

        result: dict = {}
        for col in sub.columns:
            vals = sub[col].values
            try:
                result[col] = np.stack(vals)
            except (ValueError, TypeError):
                result[col] = list(vals)
        return result

    @staticmethod
    def _load_json(path: Path) -> dict:
        if not path.exists():
            raise FileNotFoundError(f"Not found: {path}")
        with open(path) as f:
            return json.load(f)
