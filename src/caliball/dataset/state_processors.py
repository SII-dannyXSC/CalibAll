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
