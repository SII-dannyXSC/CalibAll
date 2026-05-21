"""
LeRobot 2.1 数据集读取器（无 lerobot 库依赖版本）
=================================================

参考 starVLA/LeRobotSingleDataset 的设计，直接读取本地 parquet + meta 文件，
去掉对 lerobot 库的依赖。

数据集目录结构（lerobot v2.1）::

    dataset_root/
    ├── meta/
    │   ├── info.json          # fps, robot_type, features, data_path, video_path ...
    │   ├── episodes.jsonl     # 每行一个 episode {episode_index, length, tasks}
    │   └── tasks.jsonl        # 任务列表
    ├── data/
    │   └── chunk-000/
    │       ├── episode_000000.parquet
    │       └── ...
    └── videos/
        └── chunk-000/
            └── <video_key>/
                ├── episode_000000.mp4
                └── ...
"""

import bisect
import json
import re
import warnings
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import Dataset
from typing import Optional, List


# ---------------------------------------------------------------------------
# 视频解码（独立实现，不依赖 lerobot）
# ---------------------------------------------------------------------------

_VIDEO_DECODE_AVAILABLE = False
try:
    import torchvision
    _VIDEO_DECODE_AVAILABLE = True
except ImportError:
    pass

_DECODE_VIDEO_WARNED = False


def _decode_video_frames(
    video_path: str,
    timestamps: list,
    tolerance_s: float = 0.04,
    backend: str = "pyav",
) -> torch.Tensor:
    """用 torchvision VideoReader 解码指定时间戳的视频帧。

    独立于 lerobot 库，参考 starVLA 和 lerobot 的实现。
    自动将 frame['pts']（可能为 fractions.Fraction）转为 float，
    避免 torch.cdist 的 TypeError。
    """
    torchvision.set_video_backend(backend)
    reader = torchvision.io.VideoReader(str(video_path), "video")

    timestamps = [float(t) for t in timestamps]
    first_ts = min(timestamps)
    last_ts = max(timestamps)

    keyframes_only = (backend == "pyav")
    reader.seek(first_ts, keyframes_only=keyframes_only)

    loaded_frames: list = []
    loaded_ts: list = []
    for frame in reader:
        current_ts = float(frame["pts"])
        loaded_frames.append(frame["data"])
        loaded_ts.append(current_ts)
        if current_ts >= last_ts:
            break

    if backend == "pyav":
        try:
            reader.container.close()
        except Exception:
            pass
    reader = None

    query_ts = torch.as_tensor(timestamps, dtype=torch.float64)
    loaded_ts_t = torch.as_tensor(loaded_ts, dtype=torch.float64)

    dist = torch.cdist(query_ts[:, None], loaded_ts_t[:, None], p=1)
    min_, argmin_ = dist.min(1)

    is_within_tol = min_ < tolerance_s
    assert is_within_tol.all(), (
        f"Timestamps violate tolerance ({min_[~is_within_tol]} > {tolerance_s=}). "
        f"video: {video_path}"
    )

    closest_frames = torch.stack([loaded_frames[idx] for idx in argmin_])
    closest_frames = closest_frames.type(torch.float32) / 255.0
    return closest_frames


# ---------------------------------------------------------------------------
# 兼容层（保持子类 CalibLeRobotDataset 可用）
# ---------------------------------------------------------------------------

class _DatasetMeta:
    """轻量包装 meta/info.json，兼容原有 self.meta 的属性访问。

    参考 starVLA 从 info.json 和 episodes.jsonl 提取元信息的方式。
    """

    def __init__(self, info: dict, dataset_path: Path):
        self.fps = info.get("fps", 30)
        self.robot_type = info.get("robot_type", "unknown")
        self._info = info
        self._dataset_path = dataset_path
        self._chunks_size = info.get("chunks_size", 1000)
        self._video_path_pattern = info.get("video_path", "")

        # 从 features 中提取视频相关 key
        features = info.get("features", {})
        self.video_keys: set = set()
        self.camera_keys: list = []
        for k, v in features.items():
            if v.get("dtype") == "video":
                self.video_keys.add(k)
                self.camera_keys.append(k)

    def get_video_file_path(self, episode_index: int, video_key: str) -> str:
        """根据 episode 索引和视频 key 生成视频文件相对路径。"""
        chunk = episode_index // self._chunks_size
        return self._video_path_pattern.format(
            episode_chunk=chunk,
            video_key=video_key,
            episode_index=episode_index,
        )


class _ParquetDataStore:
    """按 episode 懒加载 parquet 数据，支持全局索引切片。

    兼容 ``hf_dataset[start:end]`` 的访问模式。
    参考 starVLA 的 trajectory 缓存设计（curr_traj_data / curr_traj_id）。
    """

    def __init__(
        self,
        dataset_path: Path,
        data_path_pattern: str,
        chunks_size: int,
        episodes_info: list,
    ):
        self._dataset_path = dataset_path
        self._data_path_pattern = data_path_pattern
        self._chunks_size = chunks_size
        self._episodes = episodes_info

        # 构建全局索引 → episode 映射
        self._episode_starts: list = []
        total = 0
        for ep in episodes_info:
            self._episode_starts.append(total)
            total += ep["length"]
        self._total_frames = total

        # episode 缓存
        self._cached_ep_index: Optional[int] = None
        self._cached_df: Optional[pd.DataFrame] = None

    def _load_episode(self, episode_index: int) -> pd.DataFrame:
        """加载指定 episode 的 parquet，带缓存。"""
        if self._cached_ep_index == episode_index and self._cached_df is not None:
            return self._cached_df
        chunk = episode_index // self._chunks_size
        path = self._dataset_path / self._data_path_pattern.format(
            episode_chunk=chunk, episode_index=episode_index,
        )
        if not path.exists():
            raise FileNotFoundError(f"Parquet file not found: {path}")
        self._cached_df = pd.read_parquet(path)
        self._cached_ep_index = episode_index
        return self._cached_df

    def __getitem__(self, key):
        """支持 int 和 slice 索引，返回 dict（列名 → tensor/list）。"""
        if isinstance(key, slice):
            start = key.start or 0
            stop = key.stop or self._total_frames
        else:
            start, stop = key, key + 1

        # 根据全局索引找到对应的 episode
        ep_local_idx = bisect.bisect_right(self._episode_starts, start) - 1
        ep_info = self._episodes[ep_local_idx]
        ep_index = ep_info["episode_index"]
        local_start = start - self._episode_starts[ep_local_idx]
        local_stop = stop - self._episode_starts[ep_local_idx]

        df = self._load_episode(ep_index)
        sub = df.iloc[local_start:local_stop]

        result: dict = {}
        for col in sub.columns:
            vals = sub[col].values
            try:
                arr = np.stack(vals)
                result[col] = torch.from_numpy(np.ascontiguousarray(arr))
            except (ValueError, TypeError):
                # 非数值列（str 等）保持为 list
                result[col] = list(vals)
        return result

    def __len__(self):
        return self._total_frames


class _LeRobotCompat:
    """兼容 ``self.lerobot_ds`` 的属性访问。

    子类 CalibLeRobotDataset 通过 ``self.lerobot_ds.hf_dataset[start:end]``
    访问数据，此类提供兼容接口。
    """

    def __init__(self, root: Path, hf_dataset: _ParquetDataStore,
                 tolerance_s: float = 0.04):
        self.root = root
        self.hf_dataset = hf_dataset
        self.tolerance_s = tolerance_s


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def _scan_available_episodes(repo_id: str) -> Optional[List[int]]:
    """扫描本地 data/chunk-*/episode_*.parquet 文件，返回实际存在的 episode 索引列表。"""
    data_dir = Path(repo_id) / "data"
    if not data_dir.is_dir():
        return None
    available = []
    for p in sorted(data_dir.rglob("episode_*.parquet")):
        m = re.match(r"episode_(\d+)\.parquet$", p.name)
        if m:
            available.append(int(m.group(1)))
    return sorted(available) if available else None


def _load_info(dataset_path: Path) -> dict:
    """读取 meta/info.json。"""
    info_path = dataset_path / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"meta/info.json not found in {dataset_path}")
    with open(info_path, "r") as f:
        return json.load(f)


def _load_episodes(dataset_path: Path, episodes_filter: Optional[List[int]] = None) -> list:
    """读取 meta/episodes.jsonl，返回 episode 信息列表。

    每个元素为 dict: {episode_index, length, tasks, ...}
    """
    episodes_path = dataset_path / "meta" / "episodes.jsonl"
    if not episodes_path.exists():
        raise FileNotFoundError(f"meta/episodes.jsonl not found in {dataset_path}")
    episodes = []
    with open(episodes_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ep = json.loads(line)
            if episodes_filter is not None and ep["episode_index"] not in episodes_filter:
                continue
            episodes.append(ep)
    return episodes


# ---------------------------------------------------------------------------
# 主类
# ---------------------------------------------------------------------------

class LeRobotDataset(Dataset):
    """
    LeRobot 2.1 数据集读取器（无 lerobot 库依赖）

    直接读取本地 meta/info.json + data/*.parquet + videos/ 文件，
    参考 starVLA/LeRobotSingleDataset 的设计。

    Args:
        repo_id: 数据集本地目录路径
        episodes: 要加载的 episode 索引列表，None 表示加载所有
        split: 数据集分割，默认 "train"
        root_dir: 未使用，保留兼容
        image_transforms: 图像变换函数
        state_key: 状态数据的键名
        action_key: 动作数据的键名
        decode_video_keys: 要解码的视频 key 列表，None 则解码全部
        video_backend: 视频解码后端，默认 "pyav"

    按 episode 读取 parquet，每路视频调用一次 _decode_video_frames。
    """

    def __init__(
        self,
        repo_id: str,
        episodes: Optional[List[int]] = None,
        split: str = "train",
        root_dir: Optional[str] = None,
        image_transforms=None,
        state_key: str = "observation.states.joint_position",
        action_key: str = "observation.states.end_effector",
        decode_video_keys: Optional[List[str]] = None,
        video_backend: str = "pyav",
    ) -> None:
        super().__init__()

        self.repo_id = repo_id
        self.split = split
        self.root_dir = root_dir
        self.image_transforms = image_transforms
        self.state_key = state_key
        self.action_key = action_key
        self.decode_video_keys = decode_video_keys
        self.video_backend = video_backend

        dataset_path = Path(repo_id)
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset path not found: {dataset_path}")

        # --- 读取 meta/info.json ---
        info = _load_info(dataset_path)
        self.meta = _DatasetMeta(info, dataset_path)
        self.fps = self.meta.fps
        self.robot_type = self.meta.robot_type
        self.camera_keys = self.meta.camera_keys

        data_path_pattern = info.get(
            "data_path",
            "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        )
        chunks_size = info.get("chunks_size", 1000)

        # --- 读取 episodes ---
        if episodes is not None:
            episodes_filter = set(episodes)
        else:
            scanned = _scan_available_episodes(repo_id)
            episodes_filter = set(scanned) if scanned is not None else None

        episodes_info = _load_episodes(
            dataset_path,
            episodes_filter=list(episodes_filter) if episodes_filter else None,
        )

        # --- 构建 episode_data_index（兼容子类访问）---
        from_indices = []
        to_indices = []
        running = 0
        for ep in episodes_info:
            from_indices.append(running)
            running += ep["length"]
            to_indices.append(running)

        self.episode_data_index = {
            "from": torch.tensor(from_indices, dtype=torch.long),
            "to": torch.tensor(to_indices, dtype=torch.long),
        }
        self.total_episodes = len(episodes_info)
        self._episodes_info = episodes_info

        # --- 构建懒加载 parquet 存储 ---
        hf_dataset = _ParquetDataStore(
            dataset_path=dataset_path,
            data_path_pattern=data_path_pattern,
            chunks_size=chunks_size,
            episodes_info=episodes_info,
        )

        # --- 兼容 self.lerobot_ds 访问 ---
        self.lerobot_ds = _LeRobotCompat(
            root=dataset_path,
            hf_dataset=hf_dataset,
            tolerance_s=1.0 / self.fps,
        )

    def __len__(self):
        """获取数据集中的 episode 数量。"""
        return self.total_episodes

    @staticmethod
    def _tensor_batch_to_numpy(x):
        if x is None:
            return None
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    @staticmethod
    def _chw_float_to_hwc_uint8(frames: torch.Tensor) -> np.ndarray:
        """(T,C,H,W) float [0,1] -> (T,H,W,C) uint8"""
        x = frames.detach().cpu().numpy()
        x = np.transpose(x, (0, 2, 3, 1))
        if x.dtype in (np.float32, np.float64):
            x = (x * 255.0).clip(0, 255).astype(np.uint8)
        return x

    def _apply_image_transforms(self, img: np.ndarray) -> np.ndarray:
        if self.image_transforms is None:
            return img
        return self.image_transforms(img)

    def _getitem_episode(self, idx: int) -> dict:
        """一次读取 parquet + 每路视频一次解码（整段 timestamps）。

        参考 starVLA 的 get_trajectory_data 按 episode 懒加载设计。
        """
        start_idx = int(self.episode_data_index["from"][idx].item())
        end_idx = int(self.episode_data_index["to"][idx].item())
        if start_idx >= end_idx:
            return dict(
                video=None,
                videos={},
                states=None,
                actions=None,
                name=f"{self.repo_id}_episode_{idx}",
            )

        batch = self.lerobot_ds.hf_dataset[start_idx:end_idx]
        ep_idx = int(self._tensor_batch_to_numpy(batch["episode_index"]).reshape(-1)[0])

        ts = batch["timestamp"]
        if isinstance(ts, torch.Tensor):
            timestamps = ts.flatten().tolist()
        else:
            timestamps = list(ts)

        # --- 视频解码 ---
        videos = {}
        vid_keys = self.decode_video_keys
        if vid_keys is None:
            vid_keys = list(self.meta.video_keys)
        keys_to_decode = [k for k in vid_keys if k in self.meta.video_keys]

        if _VIDEO_DECODE_AVAILABLE and len(keys_to_decode) > 0:
            for vid_key in keys_to_decode:
                video_path = (
                    self.lerobot_ds.root
                    / self.meta.get_video_file_path(ep_idx, vid_key)
                )
                frames_t = _decode_video_frames(
                    video_path,
                    timestamps,
                    self.lerobot_ds.tolerance_s,
                    self.video_backend,
                )
                arr = self._chw_float_to_hwc_uint8(frames_t)
                if self.image_transforms is not None:
                    arr = np.stack(
                        [self._apply_image_transforms(arr[i]) for i in range(len(arr))],
                        axis=0,
                    )
                videos[vid_key] = arr
        elif len(keys_to_decode) > 0:
            global _DECODE_VIDEO_WARNED
            if not _DECODE_VIDEO_WARNED:
                _DECODE_VIDEO_WARNED = True
                warnings.warn(
                    "torchvision 不可用，将跳过视频解码，仅返回 parquet 中的 states/actions。",
                    UserWarning,
                    stacklevel=2,
                )

        # --- states / actions ---
        states = None
        actions = None
        # 支持 state_key 为 list（双臂数据集拼接 left + right）
        if isinstance(self.state_key, (list, tuple)):
            parts = []
            for sk in self.state_key:
                if sk in batch:
                    parts.append(self._tensor_batch_to_numpy(batch[sk]))
            if parts:
                states = np.concatenate(parts, axis=-1)
        elif self.state_key in batch:
            states = self._tensor_batch_to_numpy(batch[self.state_key])
        if self.action_key in batch:
            actions = self._tensor_batch_to_numpy(batch[self.action_key])

        main_video = None
        if videos:
            if self.camera_keys and self.camera_keys[0] in videos:
                main_video = videos[self.camera_keys[0]]
            else:
                main_video = next(iter(videos.values()))

        return dict(
            video=main_video,
            videos=videos,
            states=states,
            actions=actions,
            name=f"{self.repo_id}_episode_{idx}",
        )

    def __getitem__(self, idx):
        """
        获取指定索引的 episode 数据

        Args:
            idx: episode 索引

        Returns:
            dict: 包含以下键的字典
                - video: 主相机的视频帧数组 (T, H, W, C)
                - videos: 所有相机的视频帧字典
                - states: 状态序列 (T, state_dim)
                - actions: 动作序列 (T, action_dim)
                - name: episode 名称
        """
        return self._getitem_episode(idx)


if __name__ == "__main__":
    dataset = LeRobotDataset(
        "/cpfs02/user/xiesicheng.xsc/project/CalibAll/data/rdt_aloha_lerobot2.1/airpods_on_second_layer",
        state_key="observation.state",
    )

    import pdb

    pdb.set_trace()
    sample = dataset[0]

    print("LeRobot数据集加载器已准备就绪！")
    print("\n使用方法:")
    print("1. 加载数据集:")
    print('   dataset = LeRobotDataset("path/to/dataset")')
    print("\n2. 加载指定episodes:")
    print('   dataset = LeRobotDataset("path/to/dataset", episodes=[0, 1, 2])')
    print("\n3. 获取数据:")
    print("   sample = dataset[0]")
    print("   video = sample['video']")
    print("   states = sample['states']")
    print("   actions = sample['actions']")
