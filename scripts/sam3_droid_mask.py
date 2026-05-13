#!/usr/bin/env python3
"""
sam3_droid_mask.py

读取 DROID (droid_1.0.1) 数据集视频，使用 SAM3 视频模式以文本提示（默认 "robotic arm"）
对整段视频做 mask 传播，输出带 mask 叠加的视频。

SAM3 视频模式流程：
  1. init_state(mp4_path)          —— 加载整段视频
  2. add_prompt(state, frame_idx=0, text_str=prompt) —— 在第 0 帧给定文本提示
  3. propagate_in_video(state)      —— 逐帧传播，yield (frame_idx, outputs)
  4. 用传播结果的 out_binary_masks 叠加到原始帧

用法：
    # 默认：episode 0，exterior_1_left + exterior_2_left
    python scripts/sam3_droid_mask.py --dataset_dir data/droid_1.0.1

    # 指定 episode、相机、输出目录
    python scripts/sam3_droid_mask.py \\
        --dataset_dir data/droid_1.0.1 \\
        --episodes 0 1 \\
        --cams exterior_1_left \\
        --output_dir /tmp/droid_sam3

    # 自定义 SAM3 模型路径
    python scripts/sam3_droid_mask.py \\
        --bpe_path /path/to/bpe_simple_vocab_16e6.txt.gz \\
        --ckpt_path /path/to/sam3.pt

    # 修改 prompt 或 mask 颜色（BGR 顺序）
    python scripts/sam3_droid_mask.py \\
        --prompt "robot arm" \\
        --mask_color 0 255 0 \\
        --mask_alpha 0.5
"""
from __future__ import annotations

import argparse
import json
import subprocess
import shlex
import sys
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
from tqdm import tqdm
import pyarrow.parquet as pq

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from src.caliball.utils.image import add_mask

# ── Monkey-patch SAM3 视频解码（AV1 编码，OpenCV 无法软解）─────────────────────

def _patch_sam3_video_loader() -> None:
    """
    替换 sam3.model.io_utils.load_video_frames_from_video_file_using_cv2，
    改用 ffmpeg 管道解码，支持 AV1 / H.265 等 OpenCV 无法处理的编码格式。
    调用须在 import sam3 之后、init_state 之前执行。
    """
    import sam3.model.io_utils as _io

    def _ffmpeg_loader(
        video_path, image_size,
        img_mean=(0.5, 0.5, 0.5), img_std=(0.5, 0.5, 0.5),
        offload_video_to_cpu=False,
    ):
        import torch
        from tqdm import tqdm as _tqdm

        ff = _ffmpeg_exe()

        # 用 cv2 只读元数据（不实际解码帧）
        cap = cv2.VideoCapture(video_path)
        orig_h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        orig_w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None
        cap.release()

        # ffmpeg 管道：解码为 RGB24 rawvideo
        cmd = (f'{ff} -loglevel error -i "{video_path}" '
               f'-f rawvideo -pix_fmt rgb24 pipe:1')
        proc = subprocess.Popen(
            shlex.split(cmd), stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
        )
        frame_bytes = orig_h * orig_w * 3
        frames = []
        pbar = _tqdm(desc="frame loading (ffmpeg patch)", total=n_frames)
        while True:
            raw = proc.stdout.read(frame_bytes)
            if len(raw) < frame_bytes:
                break
            frame = np.frombuffer(raw, np.uint8).reshape(orig_h, orig_w, 3).copy()
            frame = cv2.resize(frame, (image_size, image_size),
                               interpolation=cv2.INTER_CUBIC)
            frames.append(frame)
            pbar.update(1)
        proc.stdout.close()
        proc.wait()
        pbar.close()

        if not frames:
            raise RuntimeError(f"ffmpeg 无法读取任何帧: {video_path}")

        frames_np = np.stack(frames).astype(np.float32)          # (T, H, W, C)
        video_t   = torch.from_numpy(frames_np).permute(0, 3, 1, 2)  # (T, C, H, W)

        mean_t = torch.tensor(img_mean, dtype=torch.float16).view(1, 3, 1, 1)
        std_t  = torch.tensor(img_std,  dtype=torch.float16).view(1, 3, 1, 1)
        if not offload_video_to_cpu:
            video_t = video_t.cuda()
            mean_t  = mean_t.cuda()
            std_t   = std_t.cuda()
        video_t = (video_t.half() - mean_t) / std_t
        return video_t, orig_h, orig_w

    _io.load_video_frames_from_video_file_using_cv2 = _ffmpeg_loader


# ── 默认 SAM3 模型路径 ─────────────────────────────────────────────────────────
_DEFAULT_BPE  = ("/cpfs02/user/xiesicheng.xsc/CalibAll/third_party/sam3"
                 "/sam3/assets/bpe_simple_vocab_16e6.txt.gz")
_DEFAULT_CKPT = "/cpfs02/user/xiesicheng.xsc/CalibAll/ckpt/sam3/sam3.pt"

# ── 视频 I/O ──────────────────────────────────────────────────────────────────

def _ffmpeg_exe() -> str:
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return "ffmpeg"


class VideoReader:
    def __init__(self, path: str):
        ff = _ffmpeg_exe()
        cmd = f'{ff} -loglevel error -i "{path}" -f rawvideo -pix_fmt bgr24 pipe:1'
        self._proc = subprocess.Popen(
            shlex.split(cmd), stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
        )
        cap = cv2.VideoCapture(path)
        self.w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
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


# ── 工具函数 ──────────────────────────────────────────────────────────────────

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


def merge_masks(out_binary_masks: np.ndarray) -> np.ndarray:
    """
    (N, H, W) bool array → (H, W) int32 mask（所有目标 OR 合并）。
    N=0 时返回全零 mask。
    """
    if out_binary_masks.shape[0] == 0:
        return None
    return out_binary_masks.any(axis=0).astype(np.int32)


# ── 主处理 ────────────────────────────────────────────────────────────────────

def process_episode(
    dataset_dir: Path,
    ep_idx: int,
    chunks_size: int,
    cams: List[str],
    model,                   # Sam3VideoInferenceWithInstanceInteractivity
    prompt: str,
    mask_color: List[int],   # BGR 顺序
    mask_alpha: float,
    output_dir: Path,
    fps: int,
):
    chunk = ep_idx // chunks_size
    pq_path = (dataset_dir / "data"
               / f"chunk-{chunk:03d}" / f"episode_{ep_idx:06d}.parquet")
    if not pq_path.exists():
        print(f"[WARN] parquet 不存在: {pq_path}")
        return

    n_rows = len(pq.read_table(pq_path))

    ep_out = output_dir / f"episode_{ep_idx:06d}"
    ep_out.mkdir(parents=True, exist_ok=True)

    for cam in cams:
        vid_path = find_video(
            dataset_dir, f"observation.images.{cam}", ep_idx, chunks_size
        )
        if vid_path is None:
            print(f"[WARN] 找不到视频: cam={cam!r} ep={ep_idx}")
            continue

        print(f"  → ep{ep_idx} / {cam}  ({n_rows} 帧)")

        # ── 1. SAM3 视频推理（ffmpeg patch 已替换 cv2 解码器）────────────────
        print(f"     [SAM3] init_state ...")
        import torch
        inference_state = model.init_state(
            resource_path=str(vid_path),
            video_loader_type="cv2",   # 实际走 patched ffmpeg loader
        )
        num_frames = inference_state["num_frames"]

        print(f"     [SAM3] add_prompt frame=0, prompt={prompt!r}")
        model.add_prompt(
            inference_state=inference_state,
            frame_idx=0,
            text_str=prompt,
        )

        # 收集所有帧的 mask
        frame_masks: Dict[int, Optional[np.ndarray]] = {}
        print(f"     [SAM3] propagate_in_video ({num_frames} 帧) ...")
        for frame_idx, outputs in model.propagate_in_video(
            inference_state=inference_state,
            start_frame_idx=0,
            reverse=False,
        ):
            if outputs is not None and len(outputs["out_binary_masks"]) > 0:
                frame_masks[frame_idx] = merge_masks(outputs["out_binary_masks"])
            else:
                frame_masks[frame_idx] = None

        # ── 2. 读取原始帧并叠加 mask，写出视频 ──────────────────────────────
        out_mp4 = ep_out / f"ep{ep_idx:06d}_{cam}_sam3.mp4"
        reader = VideoReader(str(vid_path))
        writer: Optional[VideoWriter] = None
        no_mask_cnt = 0

        for t in tqdm(range(n_rows), desc=f"ep{ep_idx}/{cam} overlay", leave=False):
            ok, bgr = reader.read()
            if not ok:
                break
            if writer is None:
                writer = VideoWriter(str(out_mp4), fps, reader.w, reader.h)

            mask_hw = frame_masks.get(t, None)
            if mask_hw is not None:
                # mask 可能与原始帧尺寸不同（SAM3 内部 resize），对齐一下
                if mask_hw.shape != (reader.h, reader.w):
                    mask_hw = cv2.resize(
                        mask_hw.astype(np.uint8),
                        (reader.w, reader.h),
                        interpolation=cv2.INTER_NEAREST,
                    ).astype(np.int32)
                frame = add_mask(bgr, mask_hw, color=mask_color, alpha=mask_alpha)
            else:
                no_mask_cnt += 1
                frame = bgr

            writer.write(frame)

        reader.release()
        if writer:
            writer.release()

        if no_mask_cnt:
            print(f"     [WARN] {no_mask_cnt}/{n_rows} 帧无 mask")
        print(f"     输出: {out_mp4}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="DROID + SAM3 视频模式机械臂分割")
    p.add_argument("--dataset_dir", default="data/droid_1.0.1",
                   help="droid_1.0.1 数据集根目录")
    p.add_argument("--output_dir",  default="/tmp/droid_sam3",
                   help="输出视频目录")
    p.add_argument("--bpe_path",   default=_DEFAULT_BPE,
                   help="SAM3 BPE 词表路径")
    p.add_argument("--ckpt_path",  default=_DEFAULT_CKPT,
                   help="SAM3 checkpoint 路径")
    p.add_argument("--episodes",   nargs="+", type=int, default=[0],
                   help="要处理的 episode 编号（默认 0）")
    p.add_argument("--cams",       nargs="+",
                   default=["exterior_1_left", "exterior_2_left"],
                   help="相机名（不含 observation.images. 前缀）")
    p.add_argument("--prompt",     default="robotic arm",
                   help="SAM3 文本提示（默认 'robotic arm'）")
    p.add_argument("--mask_color", nargs=3, type=int,
                   default=[56, 179, 253],
                   metavar=("B", "G", "R"),
                   help="mask 叠加颜色，BGR 顺序（默认 56 179 253）")
    p.add_argument("--mask_alpha", type=float, default=0.6,
                   help="mask 不透明度 0-1（默认 0.6）")
    p.add_argument("--fps",        type=int,   default=15)
    return p.parse_args()


def main():
    args = parse_args()

    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.is_absolute():
        dataset_dir = _ROOT / dataset_dir
    dataset_dir = dataset_dir.resolve()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    info = json.loads((dataset_dir / "meta" / "info.json").read_text())
    chunks_size = info.get("chunks_size", 1000)

    print(f"[INFO] 加载 SAM3 视频模型 ...")
    print(f"       bpe : {args.bpe_path}")
    print(f"       ckpt: {args.ckpt_path}")
    from sam3.model_builder import build_sam3_video_model
    _patch_sam3_video_loader()   # 替换 cv2 解码器为 ffmpeg（支持 AV1）
    import torch
    model = build_sam3_video_model(
        bpe_path=args.bpe_path,
        checkpoint_path=args.ckpt_path,
        load_from_HF=False,
    ).eval()
    print(f"[INFO] SAM3 视频模型初始化完成，prompt={args.prompt!r}")

    for ep_idx in tqdm(args.episodes, desc="Episodes"):
        process_episode(
            dataset_dir, ep_idx, chunks_size,
            args.cams, model, args.prompt,
            args.mask_color, args.mask_alpha,
            output_dir, args.fps,
        )

    print(f"\n[DONE] 输出: {output_dir}")


if __name__ == "__main__":
    main()
