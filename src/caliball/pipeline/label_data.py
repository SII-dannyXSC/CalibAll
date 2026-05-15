"""
标注结果数据结构，支持单臂和双臂、多相机。

层级：
  LabelData          —— episode 级元信息 + 按相机存储的帧列表
    cameras            —— dict[camera_name, list[FrameLabel]]
      FrameLabel       —— 单帧，各臂数据（含各自 mask/bbox）
        ArmLabel       —— 单臂 grip point 2D/3D 轨迹 + mask/bbox

序列化格式：pickle（默认）。
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# 单臂 grip point
# ---------------------------------------------------------------------------

@dataclass
class ArmLabel:
    """单臂 grip point 标注（相机坐标系）。"""

    uv: list[int]               # [u, v]

    xyz_euler_g: list[float]    # [x, y, z, rx, ry, rz, g]          (7,)
    xyz_quat_g: list[float]     # [x, y, z, qw, qx, qy, qz, g]      (8,)
    xyz_mat_g: list[float]      # [x, y, z, r00..r22, g]             (13,)
    uvd: list[float]            # [u, v, d]

    # mask —— COCO RLE dict {"size": [H, W], "counts": str}
    mask_with_gripper: Optional[dict] = None     # 该臂 arm + gripper
    mask_without_gripper: Optional[dict] = None  # 该臂 arm only
    mask_gripper: Optional[dict] = None          # 该臂 gripper only

    # bbox —— [x1, y1, x2, y2]，无前景时为 None
    bbox_with_gripper: Optional[list[int]] = None     # 该臂 arm + gripper
    bbox_without_gripper: Optional[list[int]] = None  # 该臂 arm only
    bbox_gripper: Optional[list[int]] = None          # 该臂 gripper only

    is_placeholder: bool = False  # 单臂时对侧填充的占位臂标记


# ---------------------------------------------------------------------------
# 单帧
# ---------------------------------------------------------------------------

@dataclass
class FrameLabel:
    """单帧标注：各臂数据（含各自 mask/bbox）。"""

    index: int
    arms: dict[str, ArmLabel]   # 键为臂名（如 "left" / "right" / "single"）


# ---------------------------------------------------------------------------
# 单个 episode（多相机）
# ---------------------------------------------------------------------------

@dataclass
class LabelData:
    """存储一个 episode 的完整标注结果，支持多相机。"""

    dataset_name: str
    episode_id: str
    arm_names: list[str]        # 如 ["left"] 或 ["left", "right"]

    dataset_root: Optional[str] = None  # 可选，数据集根目录，仅供参考

    # camera_name -> 该相机的帧列表（各相机帧数应相同）
    cameras: dict[str, list[FrameLabel]] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # 基本操作
    # ------------------------------------------------------------------

    def add_frame(self, camera_name: str, frame: FrameLabel) -> None:
        if camera_name not in self.cameras:
            self.cameras[camera_name] = []
        self.cameras[camera_name].append(frame)

    @property
    def camera_names(self) -> list[str]:
        return list(self.cameras.keys())

    def __len__(self) -> int:
        """返回第一个相机的帧数（各相机帧数应相同）。"""
        if not self.cameras:
            return 0
        return len(next(iter(self.cameras.values())))

    def __repr__(self) -> str:
        cam_info = {k: len(v) for k, v in self.cameras.items()}
        return (
            f"LabelData(dataset={self.dataset_name!r}, episode={self.episode_id!r}, "
            f"arms={self.arm_names}, cameras={cam_info})"
        )

    # ------------------------------------------------------------------
    # 序列化
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str | Path) -> LabelData:
        with open(Path(path), "rb") as f:
            return pickle.load(f)

    # ------------------------------------------------------------------
    # 转换为列字典（用于写入 parquet）
    # ------------------------------------------------------------------

    ARRAY_FIELDS: ClassVar[List[Tuple[str, int]]] = [
        ("uv",                    2),
        ("uvd",                   3),
        ("xyz_euler_g",           7),
        ("xyz_quat_g",            8),
        ("xyz_mat_g",            13),
        ("bbox_with_gripper",     4),
        ("bbox_without_gripper",  4),
        ("bbox_gripper",          4),
    ]
    MASK_FIELDS: ClassVar[List[str]] = [
        "mask_with_gripper",
        "mask_without_gripper",
        "mask_gripper",
    ]

    @staticmethod
    def _rle_to_str(rle: Optional[dict]) -> str:
        if rle is None:
            return ""
        counts = rle.get("counts", "")
        if isinstance(counts, (bytes, bytearray)):
            counts = counts.decode("ascii")
        return json.dumps({"size": rle["size"], "counts": counts}, ensure_ascii=True)

    def to_columns(self, prefix: str = "annotation") -> Dict[str, list]:
        """将标注数据转换为 ``{column_name: [values...]}`` 字典。

        列名格式: ``{prefix}.{camera}.{arm}.{field}``
        """
        cols: Dict[str, list] = {}

        def _col(cam: str, arm: str, fld: str) -> str:
            return f"{prefix}.{cam}.{arm}.{fld}"

        def _bbox_or_zero(bbox) -> list:
            return list(bbox) if bbox is not None else [0, 0, 0, 0]

        for cam, frames in self.cameras.items():
            T = len(frames)
            if T == 0:
                continue
            for arm_name in self.arm_names:
                for fld, _ in self.ARRAY_FIELDS:
                    cols[_col(cam, arm_name, fld)] = [None] * T
                for fld in self.MASK_FIELDS:
                    cols[_col(cam, arm_name, fld)] = [""] * T

            for t, frame in enumerate(frames):
                for arm_name, arm_label in frame.arms.items():
                    if arm_label.is_placeholder:
                        continue
                    cols[_col(cam, arm_name, "uv")][t]                   = list(map(float, arm_label.uv))
                    cols[_col(cam, arm_name, "uvd")][t]                  = list(map(float, arm_label.uvd))
                    cols[_col(cam, arm_name, "xyz_euler_g")][t]          = list(map(float, arm_label.xyz_euler_g))
                    cols[_col(cam, arm_name, "xyz_quat_g")][t]           = list(map(float, arm_label.xyz_quat_g))
                    cols[_col(cam, arm_name, "xyz_mat_g")][t]            = list(map(float, arm_label.xyz_mat_g))
                    cols[_col(cam, arm_name, "bbox_with_gripper")][t]    = _bbox_or_zero(arm_label.bbox_with_gripper)
                    cols[_col(cam, arm_name, "bbox_without_gripper")][t] = _bbox_or_zero(arm_label.bbox_without_gripper)
                    cols[_col(cam, arm_name, "bbox_gripper")][t]         = _bbox_or_zero(arm_label.bbox_gripper)
                    cols[_col(cam, arm_name, "mask_with_gripper")][t]    = self._rle_to_str(arm_label.mask_with_gripper)
                    cols[_col(cam, arm_name, "mask_without_gripper")][t] = self._rle_to_str(arm_label.mask_without_gripper)
                    cols[_col(cam, arm_name, "mask_gripper")][t]         = self._rle_to_str(arm_label.mask_gripper)

        for col, vals in cols.items():
            fill = None
            for v in vals:
                if v is not None:
                    fill = [0.0] * len(v) if isinstance(v, list) else ""
                    break
            cols[col] = [fill if v is None else v for v in vals] if fill is not None else vals
        return cols

