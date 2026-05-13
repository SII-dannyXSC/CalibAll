"""
CalibLeRobotDataset — 统一标定数据集
=====================================

替代各个数据集子类（BerkeleyUr5Dataset、TotoDataset 等），通过配置驱动。

关键参数：
- ``state_keys``：单个 str 或 List[str]，多个 key 按最后维 concat 后传给 state_fn
- ``state_fn``：可选 callable ``(raw_states: ndarray, raw_actions: ndarray) -> ndarray``
  若为 None，直接返回 concat 后的 raw_states
  可传入 :class:`StateProcessor` 实例（YAML 通过 ``_target_`` 实例化）

StateProcessor
--------------
配置驱动的 state_fn 实现，覆盖所有已知数据集的后处理逻辑：

.. code-block:: yaml

    state_fn:
      _target_: src.caliball.dataset.calib_lerobot_dataset.StateProcessor
      arm_joint_stop: 7
      gripper_mode: from_action
      gripper_action_idx: 6
      gripper_invert: true
      joint_offset: [0.0, 0.0, 0.0, 0.0, 0.0, 1.5708, 0.7854]
"""

from __future__ import annotations

import math
from typing import Callable, List, Optional, Union

import numpy as np

from src.caliball.dataset.lerobot_dataset import LeRobotDataset


# ---------------------------------------------------------------------------
# StateProcessor（可通过 Hydra _target_ 直接实例化）
# ---------------------------------------------------------------------------

class StateProcessor:
    """
    可配置的 state 后处理 callable，实现以下流水线::

        raw_state = concat(state_keys 读出的列, axis=-1)
        arm_joints = raw_state[..., start:stop:step]
        arm_joints += joint_offset  （若有）
        gripper = <按 gripper_mode 计算>
        if gripper_mode == "in_state":
            return arm_joints
        else:
            return concat([arm_joints, gripper], axis=-1)

    Parameters
    ----------
    arm_joint_start, arm_joint_stop, arm_joint_step
        等价于 ``raw_state[..., start:stop:step]``；默认全取。
        KAIST 交错格式用 ``stop=14, step=2``。
    joint_offset
        与 arm_joints 等长的加性偏置（列表或 ndarray），TOTO 数据集用。
    gripper_mode
        ``"in_state"``   – arm_joints 已含夹爪，直接返回（默认）
        ``"from_action"``– 从 actions[..., gripper_action_idx] 二值化
        ``"from_state_idx"``– 从 raw_state[..., gripper_state_idx] 二值化
        ``"constant"``   – 所有帧用 default_gripper_q
    gripper_invert
        False（默认）：原始值 > threshold → closed_val（Berkeley/UCSD）
        True：原始值 > threshold → open_val（TOTO）
    """

    def __init__(
        self,
        *,
        arm_joint_start: Optional[int] = None,
        arm_joint_stop: Optional[int] = None,
        arm_joint_step: int = 1,
        joint_offset: Optional[List[float]] = None,
        gripper_mode: str = "in_state",
        gripper_state_idx: Optional[int] = None,
        gripper_action_idx: int = -1,
        gripper_open_val: float = 0.0,
        gripper_closed_val: float = 0.8,
        gripper_threshold: float = 0.5,
        gripper_invert: bool = False,
        default_gripper_q: float = 0.0,
    ) -> None:
        self._arm_sl = slice(arm_joint_start, arm_joint_stop, arm_joint_step)
        self._offset = (
            np.asarray(joint_offset, dtype=np.float32) if joint_offset is not None else None
        )
        assert gripper_mode in ("in_state", "from_action", "from_state_idx", "constant"), \
            f"未知 gripper_mode: {gripper_mode!r}"
        self._gmode = gripper_mode
        self._gstate_idx = gripper_state_idx
        self._gaction_idx = gripper_action_idx
        self._gopen = float(gripper_open_val)
        self._gclosed = float(gripper_closed_val)
        self._gthr = float(gripper_threshold)
        self._ginvert = bool(gripper_invert)
        self._gdefault = float(default_gripper_q)

    def _binary_grip(self, raw: np.ndarray) -> np.ndarray:
        """将标量值二值化为 open_val / closed_val，支持 invert。"""
        raw = np.asarray(raw, dtype=np.float32)
        if raw.ndim == 1:
            raw = raw[:, np.newaxis]
        high_val = self._gopen if self._ginvert else self._gclosed
        low_val = self._gclosed if self._ginvert else self._gopen
        return np.where(raw > self._gthr, np.float32(high_val), np.float32(low_val))

    def __call__(
        self,
        raw_states: np.ndarray,
        raw_actions: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Parameters
        ----------
        raw_states  : (T, D) — concat 后的原始状态
        raw_actions : (T, A) 或 None
        Returns
        -------
        (T, n_joints) joint angles（含夹爪）
        """
        s = np.asarray(raw_states, dtype=np.float32)
        arm = s[..., self._arm_sl].copy()

        if self._offset is not None:
            arm = arm + self._offset

        if self._gmode == "in_state":
            return arm

        T = arm.shape[0]

        if self._gmode == "constant":
            g = np.full((T, 1), self._gdefault, dtype=np.float32)

        elif self._gmode == "from_state_idx":
            assert self._gstate_idx is not None, "gripper_state_idx 未设置"
            g = self._binary_grip(s[..., self._gstate_idx])

        elif self._gmode == "from_action":
            if raw_actions is None:
                g = np.full((T, 1), self._gdefault, dtype=np.float32)
            else:
                a = np.asarray(raw_actions, dtype=np.float32)
                idx = self._gaction_idx
                g = self._binary_grip(a[..., idx])

        else:
            raise ValueError(f"未知 gripper_mode: {self._gmode!r}")

        return np.concatenate([arm, g], axis=-1)


# ---------------------------------------------------------------------------
# CalibLeRobotDataset
# ---------------------------------------------------------------------------

class CalibLeRobotDataset(LeRobotDataset):
    """
    统一标定数据集，替代各个数据集子类。

    Parameters
    ----------
    repo_id
        LeRobot 2.1 数据集本地目录（绝对路径）。
    state_keys
        单个字符串或字符串列表；多个 key 按最后维 concat 后传给 ``state_fn``。
    action_key
        动作列名，默认 ``"action"``。
    state_fn
        可选 callable ``(raw_states: ndarray, raw_actions: ndarray) -> ndarray``。
        若为 None，直接返回 concat 后的 raw_states（即 state 已是最终关节角）。
        典型用法：传入 :class:`StateProcessor` 实例（YAML 通过 ``_target_`` 实例化）。
    step_stride
        时间下采样步长，默认 1（不下采样）。
    max_steps
        每个 episode 最多保留的帧数（下采样后计）。
    """

    def __init__(
        self,
        repo_id: str,
        state_keys: Union[str, List[str]] = "observation.state",
        action_key: str = "action",
        state_fn: Optional[Callable] = None,
        decode_video_keys: Optional[List[str]] = None,
        step_stride: int = 1,
        max_steps: Optional[int] = None,
        # 以下透传给父类
        episodes: Optional[List[int]] = None,
        split: str = "train",
        root_dir: Optional[str] = None,
        image_transforms=None,
        video_backend: str = "pyav",
    ) -> None:
        if isinstance(state_keys, str):
            state_keys = [state_keys]
        self._state_keys = list(state_keys)
        self._state_fn = state_fn
        self.step_stride = int(step_stride)
        self.max_steps = max_steps

        # 父类用第一个 key 作 state_key（用于兼容；实际读取在 _postprocess 里重载）
        super().__init__(
            repo_id=repo_id,
            episodes=episodes,
            split=split,
            root_dir=root_dir,
            image_transforms=image_transforms,
            state_key=self._state_keys[0],
            action_key=action_key,
            decode_video_keys=decode_video_keys,
            video_backend=video_backend,
        )

    # ------------------------------------------------------------------
    def _slice_time(self, x):
        if x is None:
            return None
        y = x[:: self.step_stride]
        if self.max_steps is not None:
            y = y[: self.max_steps]
        return y

    def _read_states(self, batch) -> Optional[np.ndarray]:
        """从 parquet batch 读取并 concat 所有 state_keys。"""
        parts = []
        for key in self._state_keys:
            if key in batch:
                arr = self._tensor_batch_to_numpy(batch[key])
                if arr is not None:
                    parts.append(np.asarray(arr, dtype=np.float32))
        if not parts:
            return None
        if len(parts) == 1:
            return parts[0]
        return np.concatenate(parts, axis=-1)

    # ------------------------------------------------------------------
    def _getitem_episode(self, idx: int) -> dict:
        """覆盖父类：额外读取多 state_key 并执行 state_fn。"""
        start_idx = int(self.episode_data_index["from"][idx].item())
        end_idx = int(self.episode_data_index["to"][idx].item())
        if start_idx >= end_idx:
            return dict(video=None, videos={}, states=None, actions=None,
                        name=f"{self.repo_id}_episode_{idx}")

        # 调用父类获取 videos / actions（父类已处理视频解码）
        base = super()._getitem_episode(idx)

        # 重新读 parquet 以获取多 state_keys
        batch = self.lerobot_ds.hf_dataset[start_idx:end_idx]
        raw_states = self._read_states(batch)

        base["states"] = raw_states
        return base

    # ------------------------------------------------------------------
    def _postprocess(self, out: dict) -> dict:
        out = dict(out)
        out["video"] = self._slice_time(out.get("video"))
        out["videos"] = {k: self._slice_time(v) for k, v in out.get("videos", {}).items()}
        raw_st = self._slice_time(out.get("states"))
        raw_act = self._slice_time(out.get("actions"))
        out["actions"] = raw_act
        if out.get("actions") is not None:
            out["action"] = out["actions"]

        if raw_st is None:
            out["states"] = None
        elif self._state_fn is not None:
            out["states"] = self._state_fn(raw_st, raw_act)
        else:
            out["states"] = np.asarray(raw_st, dtype=np.float32)

        return out

    def __getitem__(self, idx: int) -> dict:
        return self._postprocess(self._getitem_episode(idx))

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]


# ---------------------------------------------------------------------------
# 预定义 state_fn 工厂（TOTO 偏置常量）
# ---------------------------------------------------------------------------

TOTO_ARM_Q_OFFSET = [0.0, 0.0, 0.0, 0.0, 0.0, math.pi / 2, math.pi / 4]
