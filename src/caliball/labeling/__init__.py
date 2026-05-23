"""
labeling 子包：机器人轨迹标注（位姿计算、mask 渲染、标注编排）。

主要导出：
  PoseCalculator   — 3D 位姿计算 + 2D 投影
  MaskRenderer     — Mask/BBox 渲染
  LabelOrchestrator — 单帧/episode 标注编排
  LabelData / FrameLabel / ArmLabel — 标注数据结构
  EEFPose          — 末端执行器位姿容器
"""

from caliball.labeling.eef_pose import EEFPose
from caliball.labeling.label_data import ArmLabel, FrameLabel, LabelData
from caliball.labeling.mask_renderer import MaskBboxResult, MaskRenderer
from caliball.labeling.orchestrator import LabelOrchestrator
from caliball.labeling.pose_calculator import PoseAllRepr, PoseCalculator, PoseDelta

__all__ = [
    "PoseCalculator",
    "PoseAllRepr",
    "PoseDelta",
    "MaskRenderer",
    "MaskBboxResult",
    "LabelOrchestrator",
    "LabelData",
    "FrameLabel",
    "ArmLabel",
    "EEFPose",
]
