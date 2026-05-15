"""标注可视化绘图工具和帧渲染。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

# ─────────────────────────────── 渲染数据结构 ─────────────────────────────────


@dataclass
class MaskItem:
    mask: np.ndarray        # (H, W) uint8
    color: tuple            # BGR
    alpha: float = 0.45


@dataclass
class BboxItem:
    bbox: list              # [x1, y1, x2, y2]
    color: tuple            # BGR
    label: str = ""


@dataclass
class ArmRenderData:
    """单臂渲染所需的标准化数据。颜色和类型由调用方决定。"""
    name: str
    uv: Optional[list] = None
    uv_color: tuple = (255, 0, 0)
    masks: List[MaskItem] = field(default_factory=list)
    bboxes: List[BboxItem] = field(default_factory=list)
    xyz_cam: Optional[list] = None
    rot_flat: Optional[list] = None
    gripper_val: Optional[float] = None


# ─────────────────────────────── 基础绘图 ──────────────────────────────────────


def decode_mask(rle, shape):
    """将 RLE 编码解码为 (H, W) uint8 二值图。"""
    if rle is None:
        return None
    mask = None
    if isinstance(rle, dict) and rle.get("format") == "simple_rle":
        rle_shape = tuple(rle["size"])
        arr = np.zeros(rle_shape[0] * rle_shape[1], dtype=np.uint8)
        idx = 0
        for val, cnt in rle["counts"]:
            arr[idx: idx + cnt] = val
            idx += cnt
        mask = arr.reshape(rle_shape)
    else:
        try:
            from pycocotools import mask as coco_mask
            rle_bytes = dict(rle)
            if isinstance(rle_bytes["counts"], str):
                rle_bytes["counts"] = rle_bytes["counts"].encode("utf-8")
            mask = coco_mask.decode(rle_bytes).astype(np.uint8)
        except ImportError:
            return None
    if mask is None:
        return None
    tH, tW = shape
    if mask.shape != (tH, tW):
        mask = cv2.resize(mask, (tW, tH), interpolation=cv2.INTER_NEAREST)
    return mask


def overlay_mask(img_bgr, mask, color_bgr, alpha=0.45):
    if mask is None or not np.any(mask):
        return img_bgr
    overlay = img_bgr.copy()
    overlay[mask > 0] = color_bgr
    return cv2.addWeighted(overlay, alpha, img_bgr, 1 - alpha, 0)


def draw_bbox(img_bgr, bbox, color_bgr, label="", thickness=2):
    if bbox is None:
        return img_bgr
    x1, y1, x2, y2 = [int(v) for v in bbox]
    cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color_bgr, thickness)
    if label:
        cv2.putText(img_bgr, label, (x1, max(y1 - 4, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color_bgr, 1, cv2.LINE_AA)
    return img_bgr


def draw_point(img_bgr, uv, color_bgr, label="", radius=6, thickness=-1):
    if uv is None:
        return img_bgr
    u, v = int(uv[0]), int(uv[1])
    H, W = img_bgr.shape[:2]
    if not (0 <= u < W and 0 <= v < H):
        return img_bgr
    cv2.circle(img_bgr, (u, v), radius, color_bgr, thickness)
    cv2.circle(img_bgr, (u, v), radius, (255, 255, 255), 1)
    if label:
        cv2.putText(img_bgr, label, (u + 8, v - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color_bgr, 1, cv2.LINE_AA)
    return img_bgr


def draw_axes(img_bgr, K, xyz_cam, rot_flat, scale=0.15, thickness=6):
    try:
        K = np.array(K, dtype=np.float64)
        origin = np.array(xyz_cam, dtype=np.float64)
        R = np.array(rot_flat, dtype=np.float64).reshape(3, 3)
        pts = np.vstack([origin, origin + scale * R[:, 0],
                         origin + scale * R[:, 1], origin + scale * R[:, 2]])
        H, W = img_bgr.shape[:2]

        def project(p):
            if p[2] <= 1e-4:
                return None
            uv = K @ p
            u, v = int(round(uv[0] / uv[2])), int(round(uv[1] / uv[2]))
            return (u, v) if (0 <= u < W and 0 <= v < H) else None

        o = project(pts[0])
        if o is None:
            return img_bgr
        for i, color in enumerate([(0, 0, 255), (0, 255, 0), (255, 0, 0)]):
            tip = project(pts[i + 1])
            if tip:
                cv2.arrowedLine(img_bgr, o, tip, color, thickness,
                                tipLength=0.15, line_type=cv2.LINE_AA)
    except Exception:
        pass
    return img_bgr


def draw_gripper_state(img_bgr, gripper_val, pos=(10, 20)):
    if gripper_val is None:
        return img_bgr
    cv2.putText(img_bgr, f"gripper: {gripper_val:.2f}", pos,
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
    return img_bgr


# ─────────────────────────────── 通用帧渲染 ──────────────────────────────────


def render_frame(
    bgr: np.ndarray,
    arms: List[ArmRenderData],
    *,
    K: Optional[np.ndarray] = None,
    show_mask: bool = True,
    show_bbox: bool = True,
    show_point: bool = True,
    show_axes: bool = True,
    axes_scale: float = 0.05,
) -> np.ndarray:
    """渲染单帧标注，返回合成图像。"""
    out = bgr.copy()
    for arm_i, arm in enumerate(arms):
        tag = arm.name[0].upper()
        if show_mask:
            for item in arm.masks:
                out = overlay_mask(out, item.mask, item.color, alpha=item.alpha)
        if show_bbox:
            for item in arm.bboxes:
                out = draw_bbox(out, item.bbox, item.color, item.label)
        if show_point:
            out = draw_point(out, arm.uv, arm.uv_color, tag)
        out = draw_gripper_state(out, arm.gripper_val, pos=(10, 20 + arm_i * 18))
        if show_axes and K is not None and arm.xyz_cam and arm.rot_flat:
            draw_axes(out, K, arm.xyz_cam, arm.rot_flat, scale=axes_scale)
    return out


def render_frame_split(
    bgr: np.ndarray,
    arms: List[ArmRenderData],
    *,
    K: Optional[np.ndarray] = None,
    show_mask: bool = True,
    show_bbox: bool = True,
    show_point: bool = True,
    show_axes: bool = True,
    axes_scale: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """渲染单帧标注，返回 (full, mask_vis, bbox_vis, pts_vis) 四张图。"""
    common = dict(K=K, axes_scale=axes_scale)
    full     = render_frame(bgr, arms, show_mask=show_mask, show_bbox=show_bbox,
                            show_point=show_point, show_axes=show_axes, **common)
    mask_vis = render_frame(bgr, arms, show_mask=True, show_bbox=False,
                            show_point=False, show_axes=False, **common)
    bbox_vis = render_frame(bgr, arms, show_mask=False, show_bbox=True,
                            show_point=False, show_axes=False, **common)
    pts_vis  = render_frame(bgr, arms, show_mask=False, show_bbox=False,
                            show_point=True, show_axes=show_axes, **common)
    return full, mask_vis, bbox_vis, pts_vis
