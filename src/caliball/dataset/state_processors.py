"""预定义的 StateProcessor 实现。

StateProcessor 从 parquet 列字典中提取关节角。
LeRobotDataset 读取 state_keys 指定的列，传给 processor 处理。

用法：
    - 默认（无 processor）：concat 所有列
    - SliceProcessor：concat → slice → offset
    - 自定义：继承 StateProcessor 覆盖 __call__
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np


class StateProcessor:
    """从 columns dict 提取关节角的基类。

    子类覆盖 __call__ 实现自定义逻辑。
    可通过 Hydra _target_ 在 YAML 中配置。
    """

    def __call__(self, columns: dict[str, np.ndarray]) -> np.ndarray:
        """处理列字典，返回关节角数组。

        Args:
            columns: {parquet列名: (T, D) ndarray}
        Returns:
            (T, n_joints) 关节角数组
        """
        return np.concatenate(list(columns.values()), axis=-1)


class SliceProcessor(StateProcessor):
    """concat 所有列 → slice[start:stop:step] → 加偏置。

    YAML 示例::

        state_processor:
          _target_: caliball.dataset.state_processors.SliceProcessor
          stop: 7
          offset: [0, 0, 0, 0, 0, 1.5708, 0.7854]
    """

    def __init__(
        self,
        start: Optional[int] = None,
        stop: Optional[int] = None,
        step: int = 1,
        offset: Optional[List[float]] = None,
    ):
        self._sl = slice(start, stop, step)
        self._offset = np.asarray(offset, dtype=np.float32) if offset is not None else None

    def __call__(self, columns: dict[str, np.ndarray]) -> np.ndarray:
        raw = np.concatenate(list(columns.values()), axis=-1)
        result = raw[..., self._sl].copy()
        if self._offset is not None:
            result = result + self._offset
        return result


class DualArmSliceProcessor(StateProcessor):
    """双臂：分别 slice 两个 state_key，然后拼接。

    YAML 示例::

        state_processor:
          _target_: caliball.dataset.state_processors.DualArmSliceProcessor
          start: 0
          stop: 6
          start_2: 0
          stop_2: 6

    Web 端传入 state_keys=["obs.left", "obs.right"] 和 start/stop 参数。
    """

    def __init__(
        self,
        start: Optional[int] = None,
        stop: Optional[int] = None,
        step: int = 1,
        offset: Optional[List[float]] = None,
        start_2: Optional[int] = None,
        stop_2: Optional[int] = None,
        step_2: int = 1,
        offset_2: Optional[List[float]] = None,
    ):
        self._sl1 = slice(start, stop, step)
        self._sl2 = slice(start_2, stop_2, step_2)
        self._offset = np.asarray(offset, dtype=np.float32) if offset else None
        self._offset_2 = np.asarray(offset_2, dtype=np.float32) if offset_2 else None

    def __call__(self, columns: dict[str, np.ndarray]) -> np.ndarray:
        # 依赖 dict 插入顺序（Python 3.7+），与 LeRobotDataset.state_keys 顺序一致
        keys = list(columns.keys())
        left = columns[keys[0]][..., self._sl1].copy()
        right = columns[keys[1]][..., self._sl2].copy()
        if self._offset is not None:
            left += self._offset
        if self._offset_2 is not None:
            right += self._offset_2
        return np.concatenate([left, right], axis=-1)
