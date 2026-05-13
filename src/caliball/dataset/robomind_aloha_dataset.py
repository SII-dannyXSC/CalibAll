import warnings
from typing import List, Optional, Tuple

import numpy as np
import torch
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset as _LeRobotDataset
from torch.utils.data import Dataset

from src.caliball.dataset.lerobot_video_decode_fix import apply_lerobot_pyav_pts_fraction_fix

apply_lerobot_pyav_pts_fraction_fix()

try:
    from lerobot.common.datasets.video_utils import decode_video_frames
except ImportError:
    decode_video_frames = None

_DECODE_VIDEO_WARNED = False


class RobomindAlohaDataset(Dataset):
    """
    RoboMIND agilex_3rgb / 双臂 LeRobot：episode 级读 parquet + 按时间戳解码视频。

    与通用 ``LeRobotDataset`` 不同，**状态列在这里显式处理**：
    - 默认从 ``joint_position_left`` / ``joint_position_right`` 拼成 ``(T, 14)`` 的 ``states``；
    - 若 parquet 中有 ``action_key_left`` / ``action_key_right`` 则同样拼接为 ``actions``，否则 ``actions`` 为 ``states`` 的拷贝；
    - 若指定 ``state_key``（单列），则只读该列，行为接近单臂/已合并布局。
    """

    def __init__(
        self,
        repo_id: str,
        episodes: Optional[List[int]] = None,
        split: str = "train",
        root_dir: Optional[str] = None,
        image_transforms=None,
        state_key: Optional[str] = None,
        action_key: Optional[str] = None,
        state_key_left: str = "observation.states.joint_position_left",
        state_key_right: str = "observation.states.joint_position_right",
        action_key_left: Optional[str] = "actions.joint_position_left",
        action_key_right: Optional[str] = "actions.joint_position_right",
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
        self.state_key_left = state_key_left
        self.state_key_right = state_key_right
        self.action_key_left = action_key_left
        self.action_key_right = action_key_right
        self.decode_video_keys = decode_video_keys
        self.video_backend = video_backend

        _vb = video_backend
        if episodes is not None:
            self.lerobot_ds = _LeRobotDataset(
                repo_id=repo_id,
                episodes=episodes,
                root=root_dir,
                video_backend=_vb,
            )
        else:
            self.lerobot_ds = _LeRobotDataset(
                repo_id=repo_id,
                root=root_dir,
                video_backend=_vb,
            )

        self.meta = self.lerobot_ds.meta
        self.fps = self.meta.fps
        self.robot_type = self.meta.robot_type
        self.camera_keys = self.meta.camera_keys if hasattr(self.meta, "camera_keys") else []
        self.episode_data_index = self.lerobot_ds.episode_data_index
        self.total_episodes = len(self.episode_data_index["from"])

    def __len__(self) -> int:
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
        x = frames.detach().cpu().numpy()
        x = np.transpose(x, (0, 2, 3, 1))
        if x.dtype in (np.float32, np.float64):
            x = (x * 255.0).clip(0, 255).astype(np.uint8)
        return x

    def _apply_image_transforms(self, img: np.ndarray) -> np.ndarray:
        if self.image_transforms is None:
            return img
        return self.image_transforms(img)

    def _column_to_numpy(self, batch, key: str):
        if key not in batch:
            return None
        return self._tensor_batch_to_numpy(batch[key])

    def _states_actions_from_batch(self, batch) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """从 episode 的 parquet 切片构造 states / actions（本类与 LeRobotDataset 的主要差异）。"""
        if self.state_key is not None:
            states = self._column_to_numpy(batch, self.state_key)
            if self.action_key is not None:
                actions = self._column_to_numpy(batch, self.action_key)
            else:
                actions = np.copy(states) if states is not None else None
            return states, actions

        sl = self._column_to_numpy(batch, self.state_key_left)
        sr = self._column_to_numpy(batch, self.state_key_right)
        if sl is not None and sr is not None:
            states = np.concatenate([np.asarray(sl), np.asarray(sr)], axis=-1)
        elif sl is not None:
            states = np.asarray(sl)
        elif sr is not None:
            states = np.asarray(sr)
        else:
            states = None

        al = (
            self._column_to_numpy(batch, self.action_key_left)
            if self.action_key_left
            else None
        )
        ar = (
            self._column_to_numpy(batch, self.action_key_right)
            if self.action_key_right
            else None
        )
        if al is not None and ar is not None:
            actions = np.concatenate([np.asarray(al), np.asarray(ar)], axis=-1)
        elif states is not None:
            actions = np.copy(states)
        else:
            actions = None
        return states, actions

    def _decode_videos_for_episode(self, ep_idx: int, timestamps: list) -> dict:
        videos = {}
        vid_keys = self.decode_video_keys
        if vid_keys is None:
            vid_keys = list(self.meta.video_keys)
        keys_to_decode = [k for k in vid_keys if k in self.meta.video_keys]

        if decode_video_frames is not None and len(keys_to_decode) > 0:
            for vid_key in keys_to_decode:
                video_path = self.lerobot_ds.root / self.meta.get_video_file_path(ep_idx, vid_key)
                frames_t = decode_video_frames(
                    video_path,
                    timestamps,
                    self.lerobot_ds.tolerance_s,
                    self.video_backend,
                )
                arr = self._chw_float_to_hwc_uint8(frames_t)
                if self.image_transforms is not None:
                    arr = np.stack([self._apply_image_transforms(arr[i]) for i in range(len(arr))], axis=0)
                videos[vid_key] = arr
        elif len(keys_to_decode) > 0:
            global _DECODE_VIDEO_WARNED
            if not _DECODE_VIDEO_WARNED:
                _DECODE_VIDEO_WARNED = True
                warnings.warn(
                    "decode_video_frames 不可用，跳过视频解码。",
                    UserWarning,
                    stacklevel=2,
                )
        return videos

    def __getitem__(self, idx: int) -> dict:
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
        _ei = batch["episode_index"]
        if isinstance(_ei, torch.Tensor):
            ep_idx = int(_ei.flatten()[0].item())
        else:
            ep_idx = int(np.asarray(_ei).reshape(-1)[0])

        ts = batch["timestamp"]
        if isinstance(ts, torch.Tensor):
            timestamps = ts.flatten().tolist()
        else:
            timestamps = list(ts)

        videos = self._decode_videos_for_episode(ep_idx, timestamps)
        states, actions = self._states_actions_from_batch(batch)

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
            # actions=actions,
            name=f"{self.repo_id}_episode_{idx}",
        )


if __name__ == "__main__":
    dataset = RobomindAlohaDataset(
        "/cpfs02/user/xiesicheng.xsc/data/RoboMIND_lerobot_v2.1/benchmark1_0_compressed/agilex_3rgb/1_potatooven"
    )
    sample = dataset[0]
    print(f"video: {sample['video'].shape if sample['video'] is not None else None}")
    print(f"states: {sample['states'].shape if sample['states'] is not None else None}")
    print(f"actions: {sample['actions'].shape if sample['actions'] is not None else None}")
    print(f"cameras: {list(sample['videos'].keys())}")
