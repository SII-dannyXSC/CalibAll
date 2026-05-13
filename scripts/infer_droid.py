"""
infer_droid.py

对 droid_1.0.1 数据集运行 CalibAll 推理管道：
  - 从 parquet 读取 joint_angles（observation.state[:7]）
  - 从 parquet 读取 camera_extrinsics，作为 solve_pnp 的 init_w2c 初始解
  - 从 mp4 读取视频帧（ffmpeg，兼容 AV1 编码）
  - CoarseInit.get_extrinsic → Refinement.refine

用法：
    python scripts/infer_droid.py \\
        --dataset_dir data/droid_1.0.1 \\
        --episodes 0 1 \\
        --cam exterior_1_left \\
        --output_dir results/droid
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pyarrow.parquet as pq
from PIL import Image
from scipy.spatial.transform import Rotation
from tqdm import tqdm

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from src.caliball.coarse_init import CoarseInit
from src.caliball.refinement import Refinement

# ── DROID 默认内参（RealSense D415，320×180）────────────────────────────────────
_DEFAULT_FX, _DEFAULT_FY = 130.61, 130.61
_DEFAULT_CX, _DEFAULT_CY = 163.49, 89.70

# parquet 列名
_STATE_KEY = "observation.state"   # (8,): 7 关节角 + 1 夹爪


# ── 视频读取（ffmpeg，兼容 AV1）───────────────────────────────────────────────

def _ffmpeg_exe() -> str:
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return "ffmpeg"


def read_video_ffmpeg(path: str) -> np.ndarray:
    """用 ffmpeg 解码整段视频，返回 (T, H, W, 3) uint8 RGB。"""
    ff = _ffmpeg_exe()
    cap = cv2.VideoCapture(path)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None
    cap.release()

    cmd = f'{ff} -loglevel error -i "{path}" -f rawvideo -pix_fmt rgb24 pipe:1'
    proc = subprocess.Popen(shlex.split(cmd), stdout=subprocess.PIPE,
                            stderr=subprocess.DEVNULL)
    frame_bytes = h * w * 3
    frames = []
    while True:
        raw = proc.stdout.read(frame_bytes)
        if len(raw) < frame_bytes:
            break
        frames.append(np.frombuffer(raw, np.uint8).reshape(h, w, 3).copy())
    proc.stdout.close()
    proc.wait()
    return np.stack(frames)   # (T, H, W, 3)


# ── DROID extrinsic → w2c ──────────────────────────────────────────────────────

def droid_extrinsic_to_w2c(xyzrpy: list, rot_conv: str = "xyz") -> np.ndarray:
    """
    DROID camera_extrinsics [x, y, z, roll, pitch, yaw] → 4×4 world-to-cam。

    DROID 以 cam-to-world (c2w) 格式存储：取逆得 world-to-cam (w2c)。
    solve_pnp 的 init_w2c 参数期望 w2c。
    """
    t = np.array(xyzrpy[:3], dtype=np.float64)
    R = Rotation.from_euler(rot_conv, xyzrpy[3:]).as_matrix()
    c2w = np.eye(4)
    c2w[:3, :3] = R
    c2w[:3, 3] = t
    return np.linalg.inv(c2w)


# ── 投影可视化 ────────────────────────────────────────────────────────────────

def _project(p_world: np.ndarray, w2c: np.ndarray, K: np.ndarray):
    """robot-base 坐标系下的 3D 点 → (u, v, depth)；depth≤0 表示在相机后。"""
    p_cam = w2c @ np.append(p_world, 1.0)
    if p_cam[2] <= 0:
        return None
    uv = K @ p_cam[:3]
    return float(uv[0] / uv[2]), float(uv[1] / uv[2]), float(p_cam[2])


def _draw_axes(img: np.ndarray, K: np.ndarray, w2c: np.ndarray,
               origin: np.ndarray, R_obj: np.ndarray, scale: float = 0.05):
    """在 origin 处绘制 XYZ 坐标轴（红/绿/蓝）。w2c 为 world-to-cam 4×4。"""
    axes = R_obj * scale          # (3,3)，列为轴方向×scale
    pts = []
    for p in [origin,
              origin + axes[:, 0],
              origin + axes[:, 1],
              origin + axes[:, 2]]:
        res = _project(p, w2c, K)
        pts.append((int(res[0]), int(res[1])) if res else None)
    if pts[0] is None:
        return
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]   # X红 Y绿 Z蓝
    for i, col in enumerate(colors):
        if pts[i + 1] is not None:
            cv2.arrowedLine(img, pts[0], pts[i + 1], col, 2, tipLength=0.3)


def visualize_projection(
    video: np.ndarray,          # (T, H, W, 3) RGB
    joint_angles: np.ndarray,   # (T, 8)
    intrinsic: np.ndarray,      # (3, 3)
    extrinsic: np.ndarray,      # (4, 4) world-to-cam
    robot_tf,
    save_path: str,
    fps: int = 15,
    tag: str = "",
    axes_scale: float = 0.05,
):
    """
    用 FK + 内外参将 TCP 投影到每帧上，输出 MP4。
    extrinsic 为 world-to-cam (w2c)。
    """
    H, W = video.shape[1:3]
    out_path = os.path.join(save_path, f"vis_proj{tag}.mp4")
    ff = _ffmpeg_exe()
    cmd = (f'{ff} -loglevel error -y -f rawvideo -vcodec rawvideo '
           f'-s {W}x{H} -pix_fmt bgr24 -r {fps} -i pipe:0 '
           f'-vcodec libx264 -pix_fmt yuv420p "{out_path}"')
    proc = subprocess.Popen(shlex.split(cmd), stdin=subprocess.PIPE,
                            stderr=subprocess.DEVNULL)

    K   = intrinsic
    w2c = extrinsic

    for t in range(len(video)):
        frame = cv2.cvtColor(video[t], cv2.COLOR_RGB2BGR).copy()

        q   = joint_angles[t]
        hom = robot_tf.fkine(q[np.newaxis])  # (1, 4, 4) or (1, 2, 4, 4)
        if hom.ndim == 4:
            hom = hom[:, 0]                  # 双臂取左臂
        p_world = hom[0, :3, 3]
        R_world = hom[0, :3, :3]

        res = _project(p_world, w2c, K)
        if res is not None:
            u, v, depth = res
            u, v = int(round(u)), int(round(v))
            if 0 <= u < W and 0 <= v < H:
                cv2.circle(frame, (u, v), 6, (0, 255, 0), -1)
                cv2.putText(frame, "TCP", (u + 8, v + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1,
                            cv2.LINE_AA)
            _draw_axes(frame, K, w2c, p_world, R_world, scale=axes_scale)

        cv2.putText(frame,
            f"t={t}  tcp=[{p_world[0]:.2f},{p_world[1]:.2f},{p_world[2]:.2f}]",
            (5, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1,
            cv2.LINE_AA)

        proc.stdin.write(frame.tobytes())

    proc.stdin.close()
    proc.wait()
    print(f"  [VIS] → {out_path}")


# ── 数据路径工具 ───────────────────────────────────────────────────────────────
def find_video(dataset_dir: Path, cam: str, ep_idx: int, chunks_size: int
               ) -> Optional[Path]:
    chunk = ep_idx // chunks_size
    ep_str = f"episode_{ep_idx:06d}.mp4"
    for p in [
        dataset_dir / "videos" / f"chunk-{chunk:03d}" / f"observation.images.{cam}" / ep_str,
        dataset_dir / "videos" / f"observation.images.{cam}" / f"chunk-{chunk:03d}" / ep_str,
    ]:
        if p.exists():
            return p
    return None


# ── 单 episode 推理 ────────────────────────────────────────────────────────────

def infer_episode(
    dataset_dir: Path,
    ep_idx: int,
    chunks_size: int,
    cam: str,
    coarse_init: CoarseInit,
    refinement: Refinement,
    K: np.ndarray,
    rot_conv: str,
    output_dir: Path,
):
    chunk = ep_idx // chunks_size
    pq_path = (dataset_dir / "data"
               / f"chunk-{chunk:03d}" / f"episode_{ep_idx:06d}.parquet")
    if not pq_path.exists():
        print(f"[WARN] parquet 不存在: {pq_path}")
        return

    vid_path = find_video(dataset_dir, cam, ep_idx, chunks_size)
    if vid_path is None:
        print(f"[WARN] 找不到视频: cam={cam!r} ep={ep_idx}")
        return

    # ── 读取 parquet ──
    table = pq.read_table(pq_path)
    col = {name: table.column(name).to_pylist() for name in table.schema.names}

    states_raw = col.get(_STATE_KEY, [])
    if not states_raw:
        print(f"[WARN] ep{ep_idx} 无 {_STATE_KEY!r}，跳过")
        return

    joint_angles = np.array(states_raw, dtype=np.float64)   # (T, 8)

    ext_col = f"camera_extrinsics.{cam}"
    if ext_col not in col:
        print(f"[WARN] ep{ep_idx} 无 {ext_col!r}，init_w2c 置 None")
        init_w2c = None
    else:
        # 使用第 0 帧的 extrinsic 作为初始解
        init_w2c = droid_extrinsic_to_w2c(col[ext_col][0], rot_conv=rot_conv)

    # ── 读取视频 ──
    print(f"  [ffmpeg] 解码 {vid_path.name} ...")
    video = read_video_ffmpeg(str(vid_path))   # (T, H, W, 3) RGB
    T = min(len(video), len(joint_angles))
    video        = video[:T]
    joint_angles = joint_angles[:T]

    # ── 内参：优先使用传入的 K，否则按视频尺寸等比缩放默认值 ──
    if K is None:
        H, W = video.shape[1:3]
        sx = W / 320.0
        sy = H / 180.0
        K = np.array([[_DEFAULT_FX * sx, 0, _DEFAULT_CX * sx],
                      [0, _DEFAULT_FY * sy, _DEFAULT_CY * sy],
                      [0, 0, 1]], dtype=np.float64)
        print(f"  [K] 自动缩放内参（{W}×{H}）: fx={K[0,0]:.2f}")
    coarse_init._init_intrinsic(K)

    # ── CoarseInit ──
    save_path = str(output_dir / f"ep{ep_idx:06d}_{cam}")
    os.makedirs(save_path, exist_ok=True)

    print(f"  [CoarseInit] init_w2c={'yes' if init_w2c is not None else 'None'}")
    extrinsic, intrinsic = coarse_init.get_extrinsic(
        video=video,
        joint_angles=joint_angles,
        img_idx=0,
        save_path=save_path,
        init_w2c=init_w2c,
    )
    print(f"  [CoarseInit] extrinsic=\n{extrinsic}")

    np.save(os.path.join(save_path, "extrinsic_coarse.npy"), extrinsic)
    np.save(os.path.join(save_path, "intrinsic.npy"), intrinsic)

    # ── 可视化 coarse 结果 ──
    visualize_projection(
        video=video, joint_angles=joint_angles,
        intrinsic=intrinsic, extrinsic=extrinsic,
        robot_tf=coarse_init.robot_tf,
        save_path=save_path, tag="_coarse",
    )

    # # ── Refinement ──
    # print(f"  [Refinement] ...")
    # result, loss_dict = refinement.refine(
    #     video=video,
    #     joint_angles=joint_angles,
    #     intrinsic=intrinsic,
    #     extrinsic=extrinsic,
    #     base_path=save_path,
    # )
    # print(f"  [Refinement] loss={loss_dict}")
    # np.save(os.path.join(save_path, "extrinsic_refined.npy"), result)
    # # ── 可视化 refined 结果 ──
    # visualize_projection(
    #     video=video, joint_angles=joint_angles,
    #     intrinsic=intrinsic, extrinsic=result,
    #     robot_tf=coarse_init.robot_tf,
    #     save_path=save_path, tag="_refined",
    # )
    print(f"  → 结果保存至 {save_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="DROID CalibAll 推理")
    p.add_argument("--dataset_dir", default="data/droid_1.0.1")
    p.add_argument("--output_dir",  default="results/droid")
    p.add_argument("--episodes",    nargs="+", type=int, default=[0])
    p.add_argument("--cam",         default="exterior_1_left",
                   help="相机名（不含 observation.images. 前缀）")
    p.add_argument("--rot_conv",    default="xyz",
                   help="DROID extrinsic euler 约定（默认 xyz）")
    # 模型路径
    p.add_argument("--dinov2_ckpt",
                   default="ckpt/dinov2/dinov2_vitb14_pretrain.pth")
    p.add_argument("--dinov2_repo",   default="third_party/dinov2")
    p.add_argument("--dinov2_id",     default="dinov2_vitb14")
    p.add_argument("--tracker_repo",  default="third_party/co-tracker")
    p.add_argument("--tracker_id",    default="cotracker3_offline")
    p.add_argument("--tracker_ckpt",
                   default="ckpt/cotracker/scaled_offline.pth")
    p.add_argument("--sam3_bpe",
                   default="third_party/sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz")
    p.add_argument("--sam3_ckpt",     default="ckpt/sam3/sam3.pt")
    # 内参（可选，不填则自动按视频分辨率缩放）
    p.add_argument("--intrinsic", nargs=4, type=float,
                   metavar=("FX", "FY", "CX", "CY"), default=None)
    return p.parse_args()


def main():
    args = parse_args()

    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.is_absolute():
        dataset_dir = (_ROOT / dataset_dir).resolve()

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = (_ROOT / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    info = json.loads((dataset_dir / "meta" / "info.json").read_text())
    chunks_size = info.get("chunks_size", 1000)

    # ── 构建 config ──
    config = type("", (), {})()
    config.robot_type        = "franka_robotiq"   # DROID 使用 Franka + Robotiq
    config.ckpt_path         = args.dinov2_ckpt
    config.repo_dir          = args.dinov2_repo
    config.dino_id           = args.dinov2_id
    config.tracker_repo_dir  = args.tracker_repo
    config.tracker_id        = args.tracker_id
    config.tracker_ckpt_path = args.tracker_ckpt
    config.bpe_path          = args.sam3_bpe
    config.ckpt_path         = args.sam3_ckpt    # sam3 覆盖 dinov2（Refinement 用）

    # ── 初始化管道 ──
    print("[INFO] 初始化 CoarseInit ...")
    coarse_init = CoarseInit(config=config)
    coarse_init.to("cuda")

    print("[INFO] 初始化 Refinement ...")
    refinement = Refinement(config=config)

    # ── 内参矩阵（若命令行指定则固定，否则每 episode 按视频尺寸自动估算）──
    K = None
    if args.intrinsic is not None:
        fx, fy, cx, cy = args.intrinsic
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
        print(f"[INFO] 使用固定内参 K:\n{K}")

    # ── 逐 episode 推理 ──
    for ep_idx in tqdm(args.episodes, desc="Episodes"):
        print(f"\n[Episode {ep_idx}]")
        infer_episode(
            dataset_dir, ep_idx, chunks_size,
            args.cam, coarse_init, refinement,
            K, args.rot_conv, output_dir,
        )

    print(f"\n[DONE] 结果保存至 {output_dir}")


if __name__ == "__main__":
    main()
