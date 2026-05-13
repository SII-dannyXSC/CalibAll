#!/usr/bin/env python3
"""
visualize_droid.py

对 DROID (droid_1.0.1) 数据集做 FK 投影可视化：
  - 从 parquet 读取 joint_position + gripper + camera_extrinsics
  - 用 FrankaPandaHandTF 计算 grip point（TCP）在 robot base 坐标系下的位置
  - 将 extrinsic [x,y,z,roll,pitch,yaw] 转为 4×4 矩阵（支持多种 euler 约定）
  - 用内参 K 将 grip point 投影到像素坐标
  - 叠加到视频帧上，输出 MP4

用法：
    # 默认：episode 0，exterior_1_left 相机，xyz euler 约定
    python scripts/visualize_droid.py --dataset_dir data/droid_1.0.1

    # 指定多个 episode 和相机
    python scripts/visualize_droid.py \\
        --dataset_dir data/droid_1.0.1 \\
        --episodes 0 1 2 \\
        --cams exterior_1_left exterior_2_left \\
        --output_dir /tmp/droid_vis

    # 调整内参（fx,fy,cx,cy）和旋转约定
    python scripts/visualize_droid.py \\
        --dataset_dir data/droid_1.0.1 \\
        --intrinsic 229 228 160 90 \\
        --rot_conv XYZ

    # 改变 extrinsic 语义：world-to-cam（默认 cam-to-world）
    python scripts/visualize_droid.py --extrinsic_inv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation
from tqdm import tqdm

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from src.caliball.robot.composite.franka_robotiq import FrankaRobotiqTF

# ── 默认内参（RealSense，原始 1280×720 标定值缩放到 320×180，比例 0.25）────
_DEFAULT_FX = 130.61   # 522.42 * (320/1280)
_DEFAULT_FY = 130.61   # 522.42 * (180/720)
_DEFAULT_CX = 163.49   # 653.96 * (320/1280)
_DEFAULT_CY = 89.70    # 358.79 * (180/720)

# ── FrankaRobotiqTF 固定参数 ──────────────────────────────────────────────────
_ARM_NAMES  = ["panda_link1", "panda_link2", "panda_link3", "panda_link4",
               "panda_link5", "panda_link6", "panda_link7", "panda_link8"]
_ARM_EEF    = "panda_link8"
_GRIP_NAMES = ["robotiq_arg2f_base_link", "left_outer_knuckle", "left_outer_finger",
               "left_inner_finger", "left_inner_knuckle", "right_outer_knuckle",
               "right_outer_finger", "right_inner_finger", "right_inner_knuckle"]
_GRIP_EEF   = "left_inner_finger"

# ── observation.state 列名 ────────────────────────────────────────────────────
_STATE_KEY  = "observation.state"   # (8,): 7 臂关节角 + 1 夹爪开合 [0, 0.8]

# ── 视频工具 ──────────────────────────────────────────────────────────────────

def _ffmpeg_exe() -> str:
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return "ffmpeg"


class VideoReader:
    def __init__(self, path: str):
        import subprocess, shlex
        ff = _ffmpeg_exe()
        cmd = f'{ff} -loglevel error -i "{path}" -f rawvideo -pix_fmt bgr24 pipe:1'
        self._proc = subprocess.Popen(
            shlex.split(cmd), stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
        )
        cap = cv2.VideoCapture(path)
        self.w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        self._frame_bytes = self.w * self.h * 3

    def read(self):
        raw = self._proc.stdout.read(self._frame_bytes)
        if len(raw) < self._frame_bytes:
            return False, None
        return True, np.frombuffer(raw, np.uint8).reshape(self.h, self.w, 3)

    def release(self):
        self._proc.stdout.close()
        self._proc.wait()


class VideoWriter:
    def __init__(self, path: str, fps: int, w: int, h: int):
        import subprocess, shlex
        ff = _ffmpeg_exe()
        cmd = (f'{ff} -loglevel error -y -f rawvideo -vcodec rawvideo '
               f'-s {w}x{h} -pix_fmt bgr24 -r {fps} -i pipe:0 '
               f'-vcodec libx264 -pix_fmt yuv420p "{path}"')
        self._proc = subprocess.Popen(
            shlex.split(cmd), stdin=subprocess.PIPE, stderr=subprocess.DEVNULL
        )

    def write(self, frame: np.ndarray):
        self._proc.stdin.write(frame.tobytes())

    def release(self):
        self._proc.stdin.close()
        self._proc.wait()


# ── Extrinsic 转换 ────────────────────────────────────────────────────────────

def extrinsic_to_mat(xyzrpy: list, rot_conv: str, invert: bool) -> np.ndarray:
    """
    [x,y,z,roll,pitch,yaw] → 4×4 numpy matrix.

    rot_conv: scipy Rotation.from_euler 的 seq 参数，如 'xyz'、'XYZ'、'zyx'
    invert:   True 表示存储的是 world-to-cam，需要取逆得到 cam-to-world
              False 表示存储的是 cam-to-world（默认假设）
    返回值语义：T_world_cam（cam-to-world），用于将 base 坐标系下的点变换到
              相机坐标系时用 T_world_cam^{-1}。
    """
    t = np.array(xyzrpy[:3], dtype=np.float64)
    rpy = np.array(xyzrpy[3:], dtype=np.float64)
    R = Rotation.from_euler(rot_conv, rpy).as_matrix()
    T = np.eye(4)
    T[:3, :3] = R
    T[:3,  3] = t
    # result = np.linalg.inv(T)
    return T


def project_point(p_world: np.ndarray, T_world_cam: np.ndarray, K: np.ndarray):
    """
    将 robot base 坐标系下的点 p_world (3,) 投影到像素坐标。
    T_world_cam: cam-to-world 4×4；取逆得 world-to-cam。
    返回 (u, v, depth)；depth<=0 表示点在相机后面。
    """
    T_cam_world = np.linalg.inv(T_world_cam)
    p_h = np.array([*p_world, 1.0])
    p_cam = T_cam_world @ p_h          # (4,)
    z = p_cam[2]
    if z <= 0:
        return None
    uv = K @ p_cam[:3]
    return float(uv[0] / uv[2]), float(uv[1] / uv[2]), z


# ── 绘制工具 ──────────────────────────────────────────────────────────────────

def draw_grip_point(img: np.ndarray, uv, color=(0, 255, 0), radius=6, label="grip"):
    if uv is None:
        return img
    u, v = int(round(uv[0])), int(round(uv[1]))
    H, W = img.shape[:2]
    if 0 <= u < W and 0 <= v < H:
        cv2.circle(img, (u, v), radius, color, -1)
        cv2.putText(img, label, (u + 8, v + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
    return img


def draw_axes(img: np.ndarray, K: np.ndarray, T_world_cam: np.ndarray,
              origin_world: np.ndarray, scale: float = 0.05,
              R_obj: Optional[np.ndarray] = None):
    """在 origin_world 处绘制 XYZ 三轴（红/绿/蓝）。
    R_obj: 若提供，使用该旋转矩阵（base frame）决定轴方向；否则用世界坐标轴方向。
    """
    if R_obj is not None:
        axes = R_obj * scale          # (3,3)，每列是一个轴方向 × scale
    else:
        axes = np.eye(3) * scale
    T_cam_world = np.linalg.inv(T_world_cam)
    pts = []
    for p in [origin_world,
              origin_world + axes[:, 0],
              origin_world + axes[:, 1],
              origin_world + axes[:, 2]]:
        pc = T_cam_world @ np.array([*p, 1.0])
        if pc[2] <= 0:
            pts.append(None)
        else:
            uv = K @ pc[:3]
            pts.append((int(uv[0]/uv[2]), int(uv[1]/uv[2])))
    if pts[0] is None:
        return img
    colors = [(0,0,255), (0,255,0), (255,0,0)]   # X红 Y绿 Z蓝
    for i, col in enumerate(colors):
        if pts[i+1] is not None:
            cv2.arrowedLine(img, pts[0], pts[i+1], col, 2, tipLength=0.3)
    return img


# ── 主处理 ────────────────────────────────────────────────────────────────────

def find_video(dataset_dir: Path, cam: str, ep_idx: int, chunks_size: int) -> Optional[Path]:
    chunk = ep_idx // chunks_size
    ep_str = f"episode_{ep_idx:06d}.mp4"
    for p in [
        dataset_dir / "videos" / f"chunk-{chunk:03d}" / cam / ep_str,
        dataset_dir / "videos" / cam / f"chunk-{chunk:03d}" / ep_str,
    ]:
        if p.exists():
            return p
    return None


def process_episode(
    dataset_dir: Path,
    ep_idx: int,
    chunks_size: int,
    cams: List[str],
    tf_model: FrankaRobotiqTF,
    K: np.ndarray,
    rot_conv: str,
    extrinsic_inv: bool,
    output_dir: Path,
    fps: int,
    axes_scale: float,
):
    chunk = ep_idx // chunks_size
    pq_path = dataset_dir / "data" / f"chunk-{chunk:03d}" / f"episode_{ep_idx:06d}.parquet"
    if not pq_path.exists():
        print(f"[WARN] parquet 不存在: {pq_path}")
        return

    table = pq.read_table(pq_path)
    n_rows = len(table)
    col = {name: table.column(name).to_pylist() for name in table.schema.names}

    ep_out = output_dir / f"episode_{ep_idx:06d}"
    ep_out.mkdir(parents=True, exist_ok=True)

    for cam in cams:
        ext_col = f"camera_extrinsics.{cam}"
        if ext_col not in col:
            print(f"[WARN] 列 {ext_col!r} 不存在，跳过")
            continue

        vid_path = find_video(dataset_dir, f"observation.images.{cam}", ep_idx, chunks_size)
        if vid_path is None:
            print(f"[WARN] 找不到视频: cam={cam} ep={ep_idx}")
            continue

        # 取 episode 内第一帧的 extrinsic（episode 内通常固定，此处不单独使用）

        reader = VideoReader(str(vid_path))
        out_mp4 = ep_out / f"ep{ep_idx:06d}_{cam}.mp4"
        writer: Optional[VideoWriter] = None

        for t in range(n_rows):
            ok, bgr = reader.read()
            if not ok:
                break
            if writer is None:
                writer = VideoWriter(str(out_mp4), fps, reader.w, reader.h)

            # 从 observation.state 取关节角，经 FK 得 TCP 位姿
            state = col.get(_STATE_KEY, [None] * n_rows)[t]
            if state is None:
                writer.write(bgr)
                continue
            q = np.array(state, dtype=np.float64)   # (8,): 7 关节 + 1 夹爪
            T_eef = tf_model.fkine_gripper(q)        # (1, 4, 4)
            p_grip = T_eef[0, :3, 3]                 # [x, y, z] in robot base
            R_grip = T_eef[0, :3, :3]

            # 也可以用帧级 extrinsic（若帧间有变化）
            T_world_cam = extrinsic_to_mat(col[ext_col][t], rot_conv, extrinsic_inv)

            uv = project_point(p_grip, T_world_cam, K)

            frame = bgr.copy()
            if uv is not None:
                draw_grip_point(frame, uv, color=(0, 255, 0))
                # 画 EEF 坐标轴（旋转来自 FK）
                draw_axes(frame, K, T_world_cam, p_grip, scale=axes_scale,
                          R_obj=R_grip)
            xyzrpy = col[ext_col][t]
            cv2.putText(frame,
                f"ext_t=[{xyzrpy[0]:.2f},{xyzrpy[1]:.2f},{xyzrpy[2]:.2f}]",
                (5, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200,200,200), 1, cv2.LINE_AA)
            cv2.putText(frame,
                f"q=[{q[0]:.2f},{q[1]:.2f},{q[2]:.2f},{q[3]:.2f}]",
                (5, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200,200,200), 1, cv2.LINE_AA)
            # FK grip world pos
            cv2.putText(frame,
                f"fk=[{p_grip[0]:.2f},{p_grip[1]:.2f},{p_grip[2]:.2f}]",
                (5, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,255,0), 1, cv2.LINE_AA)

            writer.write(frame)

        reader.release()
        if writer:
            writer.release()
            print(f"  → {out_mp4.relative_to(output_dir)}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="DROID FK 投影可视化")
    p.add_argument("--dataset_dir", default="data/droid_1.0.1",
                   help="droid_1.0.1 数据集根目录")
    p.add_argument("--output_dir",  default="/tmp/droid_vis")
    p.add_argument("--episodes",    nargs="+", type=int, default=[0],
                   help="要处理的 episode 编号（默认 0）")
    p.add_argument("--cams",        nargs="+",
                   default=["exterior_1_left", "exterior_2_left"],
                   help="相机名（不含 observation.images. 前缀）")
    p.add_argument("--intrinsic",   nargs=4, type=float,
                   metavar=("FX", "FY", "CX", "CY"),
                   default=[_DEFAULT_FX, _DEFAULT_FY, _DEFAULT_CX, _DEFAULT_CY],
                   help="相机内参 fx fy cx cy（默认 RealSense D415 @ 320×180）")
    p.add_argument("--rot_conv",    default="xyz",
                   help="scipy Rotation.from_euler seq（默认 xyz；可试 XYZ / zyx / ZYX）")
    p.add_argument("--no_extrinsic_inv", action="store_true",
                   help="关闭默认的 extrinsic 取逆（默认认为存储的是 world-to-cam，自动取逆）")
    p.add_argument("--fps",         type=int,   default=15)
    p.add_argument("--axes_scale",  type=float, default=0.05,
                   help="坐标轴箭头长度（米，默认 0.05）")
    return p.parse_args()


def main():
    args = parse_args()

    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.is_absolute():
        dataset_dir = _ROOT / dataset_dir
    dataset_dir = dataset_dir.resolve()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    import json
    info = json.loads((dataset_dir / "meta" / "info.json").read_text())
    chunks_size = info.get("chunks_size", 1000)

    # 构建内参矩阵
    fx, fy, cx, cy = args.intrinsic
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    print(f"[INFO] 内参 K:\n{K}")
    print(f"[INFO] Euler 约定: {args.rot_conv!r}  extrinsic_inv={not args.no_extrinsic_inv}")

    # 初始化 FK 模型
    tf_model = FrankaRobotiqTF(
        arm_names=_ARM_NAMES,
        arm_eef_name=_ARM_EEF,
        gripper_names=_GRIP_NAMES,
        gripper_eef_name=_GRIP_EEF,
    )
    print(f"[INFO] FrankaRobotiqTF 初始化完成")

    for ep_idx in tqdm(args.episodes, desc="Episodes"):
        process_episode(
            dataset_dir, ep_idx, chunks_size,
            args.cams, tf_model, K,
            args.rot_conv, not args.no_extrinsic_inv,
            output_dir, args.fps, args.axes_scale,
        )

    print(f"\n[DONE] 输出: {output_dir}")


if __name__ == "__main__":
    main()
