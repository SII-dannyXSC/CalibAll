import os
from typing import List, Optional

import numpy as np

from src.caliball.dataset.lerobot_dataset import LeRobotDataset

# 原始 observation.state: [j0..j5, x,y,z, qx,qy,qz,qw, gripper_is_closed, action_blocked]
_STATE_JOINT_SLICE = slice(0, 6)
_GRIPPER_IS_CLOSED_IDX = 13
_GRIPPER_CLOSED = 0.8  # 与 Robotiq URDF 全闭一致
_GRIPPER_OPEN = 0.0


class BerkeleyUr5Dataset(LeRobotDataset):
    """
    Berkeley Autolab UR5（LeRobot 2.1 本地目录），默认键与 `berkeley_autolab_ur5.yaml` 一致。

    提供 `repo_id` 或 `root_dir`+`name`（拼为 ``os.path.join(root_dir, name)``）。

    在父类输出的 episode 上可选：`step_stride`、`max_steps` 下采样时间维。

    ``states``：若最后一维为 15（原始 Berkeley 布局），则压缩为 7 维
    ``[j0..j5, gripper]``，其中 ``gripper`` 由 ``gripper_is_closed`` 映射（1→0.8，0→0），
    与 Robotiq 开合标量一致；若已是 7 维则不再变换。
    可选 ``qpos_dim`` 在变换之后对最后一维截断（如仅要 6 维臂角）。

    ``__getitem__`` / ``__iter__`` 与 `LeRobotDataset` 一致；额外设置 ``action`` 为 ``actions`` 的别名（兼容旧脚本）。
    """

    def __init__(
        self,
        repo_id: Optional[str] = None,
        episodes: Optional[List[int]] = None,
        split: str = "train",
        root_dir: Optional[str] = None,
        *,
        name: Optional[str] = None,
        image_transforms=None,
        state_key: str = "observation.state",
        action_key: str = "action",
        decode_video_keys: Optional[List[str]] = None,
        video_backend: str = "pyav",
        max_episodes: Optional[int] = None,
        max_steps: Optional[int] = None,
        step_stride: int = 1,
        qpos_dim: Optional[int] = None,
    ) -> None:
        if repo_id is None:
            if not root_dir or not name:
                raise TypeError(
                    "BerkeleyUr5Dataset 需要 repo_id（LeRobot 数据集目录），或同时提供 root_dir 与 name"
                )
            repo_id = os.path.join(os.path.expanduser(str(root_dir)), str(name))

        if episodes is None and max_episodes is not None:
            episodes = list(range(int(max_episodes)))

        super().__init__(
            repo_id=repo_id,
            episodes=episodes,
            split=split,
            root_dir=None,
            image_transforms=image_transforms,
            state_key=state_key,
            action_key=action_key,
            decode_video_keys=decode_video_keys,
            video_backend=video_backend,
        )

        self.max_steps = max_steps
        self.step_stride = int(step_stride)
        self.qpos_dim = qpos_dim

    def _slice_time(self, x):
        if x is None:
            return None
        y = x[:: self.step_stride]
        if self.max_steps is not None:
            y = y[: self.max_steps]
        return y

    @staticmethod
    def _raw_state_to_arm_gripper(states: np.ndarray) -> np.ndarray:
        """(T,15)→(T,7) 或 (15,)→(7,)；已为 (…,7) 则原样返回。"""
        s = np.asarray(states, dtype=np.float32)
        if s.size == 0:
            return s
        d = s.shape[-1]
        if d == 7:
            return s
        if d < _GRIPPER_IS_CLOSED_IDX + 1:
            return s
        closed = s[..., _GRIPPER_IS_CLOSED_IDX : _GRIPPER_IS_CLOSED_IDX + 1]
        grip = np.where(closed > 0.5, _GRIPPER_CLOSED, _GRIPPER_OPEN).astype(np.float32)
        joints = s[..., _STATE_JOINT_SLICE]
        return np.concatenate([joints, grip], axis=-1)

    def _berkeley_postprocess(self, out: dict) -> dict:
        out = dict(out)
        out["video"] = self._slice_time(out.get("video"))
        out["videos"] = {k: self._slice_time(v) for k, v in out.get("videos", {}).items()}
        out["states"] = self._slice_time(out.get("states"))
        out["actions"] = self._slice_time(out.get("actions"))
        if out["states"] is not None:
            out["states"] = self._raw_state_to_arm_gripper(out["states"])
        if self.qpos_dim is not None and out["states"] is not None:
            out["states"] = np.asarray(out["states"], dtype=np.float32)[..., : self.qpos_dim]
        if out.get("actions") is not None:
            out["action"] = out["actions"]
        return out

    def __getitem__(self, idx):
        return self._berkeley_postprocess(super().__getitem__(idx))

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]


if __name__ == "__main__":
    # 示例：本地 LeRobot 2.1 转换目录（见 config/berkeley_autolab_ur5.yaml）
    BASE = "/cpfs02/user/xiesicheng.xsc/convert/oxe/lerobot_2.1"
    ds = BerkeleyUr5Dataset(root_dir=BASE, name="berkeley_autolab_ur5", max_episodes=1, max_steps=50)

    ep = ds[0]
    print("name:", ep["name"])
    print("video", None if ep["video"] is None else ep["video"].shape, ep["video"].dtype if ep["video"] is not None else None)
    print("states", ep["states"].shape, ep["states"].dtype)
    print("actions", None if ep["actions"] is None else ep["actions"].shape)
