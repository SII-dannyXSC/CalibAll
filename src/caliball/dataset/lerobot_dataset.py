import re
import warnings
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset as _LeRobotDataset
from typing import Optional, List

# torchvision VideoReader 的 pts 可能为 Fraction，需在 lerobot 解码里转为 float，见 lerobot_video_decode_fix
from src.caliball.dataset.lerobot_video_decode_fix import apply_lerobot_pyav_pts_fraction_fix

apply_lerobot_pyav_pts_fraction_fix()


def _apply_datasets_generate_from_dict_fix():
    """兼容旧版 annotation 写入的 parquet feature 格式。

    旧格式（无 _type 字段）: {"dtype": "float32", "shape": [N], "names": null}
    datasets 4.x 的 generate_from_dict 遇到无 _type 的 dict 会递归处理其 value，
    将 "float32" 等字符串传入后调 .items() 报 AttributeError。
    补丁：检测到 shape+dtype 组合时，直接返回对应的 Sequence/Value feature。
    """
    import datasets.features.features as _feat

    _orig = _feat.generate_from_dict

    def _patched(obj):
        if isinstance(obj, dict) and "_type" not in obj and "dtype" in obj and "shape" in obj:
            shape = obj.get("shape") or []
            dtype = obj.get("dtype", "float32")
            feature = _feat.Value(dtype)
            for dim in reversed(shape):
                feature = _feat.Sequence(feature, length=dim if dim else -1)
            return feature
        return _orig(obj)

    _feat.generate_from_dict = _patched


_apply_datasets_generate_from_dict_fix()

try:
    from lerobot.common.datasets.video_utils import decode_video_frames
except ImportError:
    decode_video_frames = None

_DECODE_VIDEO_WARNED = False


def _scan_available_episodes(repo_id: str) -> Optional[List[int]]:
    """
    扫描本地 data/chunk-*/episode_*.parquet 文件，返回实际存在的 episode 索引列表。
    若目录不存在或无法扫描（HF repo），返回 None（交由 lerobot 自行处理）。
    """
    data_dir = Path(repo_id) / "data"
    if not data_dir.is_dir():
        return None
    available = []
    for p in sorted(data_dir.rglob("episode_*.parquet")):
        m = re.match(r"episode_(\d+)\.parquet$", p.name)
        if m:
            available.append(int(m.group(1)))
    return sorted(available) if available else None


class LeRobotDataset(Dataset):
    """
    LeRobot 2.1数据集读取器

    支持从HuggingFace Hub加载LeRobot格式的机器人数据集

    Args:
        repo_id: HuggingFace上的数据集repo ID，例如 "lerobot/aloha_mobile_cabinet"
        episodes: 要加载的episode索引列表，None表示加载所有episodes
        split: 数据集分割，默认为"train"
        root_dir: 本地缓存目录
        image_transforms: 图像变换函数
        state_key: 状态数据的键名
        action_key: 动作数据的键名
        decode_video_keys: 若为 None 则解码 meta.video_keys 中全部相机；可传入子集以进一步加速
        video_backend: 视频解码后端，默认 "pyav"（torchvision VideoReader + PyAV），避免 torchcodec 依赖系统 FFmpeg/libavutil。
            需要时可改为 "torchcodec" 等 lerobot 支持的值。

    按 episode 批量读 parquet，每路视频对整段 timestamps 调用一次 decode_video_frames。
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
        _vb = video_backend
        if episodes is not None:
            episodes_to_load = episodes
        else:
            # 扫描实际存在的 parquet 文件，避免不完整数据集触发 lerobot assert
            episodes_to_load = _scan_available_episodes(repo_id)

        if episodes_to_load is not None:
            self.lerobot_ds = _LeRobotDataset(
                repo_id=repo_id,
                episodes=episodes_to_load,
                root=root_dir,
                video_backend=_vb,
            )
        else:
            self.lerobot_ds = _LeRobotDataset(
                repo_id=repo_id,
                root=root_dir,
                video_backend=_vb,
            )

        # 获取数据集元信息
        self.meta = self.lerobot_ds.meta
        self.fps = self.meta.fps
        self.robot_type = self.meta.robot_type
        self.camera_keys = self.meta.camera_keys if hasattr(self.meta, "camera_keys") else []

        # 获取episode索引信息（不加载实际数据）
        self.episode_data_index = self.lerobot_ds.episode_data_index
        self.total_episodes = len(self.episode_data_index["from"])

    def __len__(self):
        """获取数据集中的episode数量"""
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
        """一次切片 parquet + 每路视频一次 decode_video_frames（整段 timestamps）。"""
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
                    "decode_video_frames 不可用（lerobot video_utils 未导入），将跳过视频解码，"
                    "仅返回 parquet 中的 states/actions。",
                    UserWarning,
                    stacklevel=2,
                )

        states = None
        actions = None
        if self.state_key in batch:
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
        获取指定索引的episode数据（懒加载模式）

        Args:
            idx: episode索引

        Returns:
            dict: 包含以下键的字典
                - video: 主相机的视频帧数组 (T, H, W, C)
                - videos: 所有相机的视频帧字典
                - states: 状态序列 (T, state_dim)
                - actions: 动作序列 (T, action_dim)
                - name: episode名称
        """
        return self._getitem_episode(idx)


if __name__ == "__main__":
    dataset = LeRobotDataset(
        "/cpfs02/user/xiesicheng.xsc/project/CalibAll/data/rdt_aloha_lerobot2.1/airpods_on_second_layer",
        state_key = "observation.state",
    )

    import pdb

    pdb.set_trace()
    sample = dataset[0]

    print("LeRobot数据集加载器已准备就绪！")
    print("\n使用方法:")
    print("1. 加载数据集:")
    print('   dataset = LeRobotDataset("lerobot/aloha_mobile_cabinet")')
    print("\n2. 加载指定episodes:")
    print('   dataset = LeRobotDataset("lerobot/pusht", episodes=[0, 1, 2])')
    print("\n3. 获取数据:")
    print("   sample = dataset[0]")
    print("   video = sample['video']")
    print("   states = sample['states']")
    print("   actions = sample['actions']")
