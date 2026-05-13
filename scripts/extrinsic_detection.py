"""
外参检测流程脚本（对应 extrinsic_detection_notebook.ipynb 前半段）：
分两趟运行：若导出目录里尚无完整 PNG，则本趟只解码视频、写入帧并校验后退出；帧已齐全时再跑浏览器选点、SAM3 与写 manual_label。

与 notebook 中 Matplotlib 交互等价的部分改用本地 HTTP 页面（见 caliball.utils.web_interaction）。

用法示例（在项目根目录）：
  PYTHONPATH=. python scripts/extrinsic_detection.py \\
    --task-path data/RoboMIND_lerobot_v2.1/.../pick_up_can \\
    --dataset-name robomind.ur_1rgb \\
    --camera-name observation.images.camera_top \\
    --robot-type ur5e

远程机器上可把 --host 0.0.0.0，本地 SSH 转发：
  ssh -L 8765:127.0.0.1:8765 -L 8766:127.0.0.1:8766 user@remote
  浏览器打开 http://127.0.0.1:8765 （tracking）与 :8766（SAM）。

封装启动（环境变量见脚本内注释）：
  export TASK_PATH=.../ur_1rgb/<task_name>
  bash scripts/run_extrinsic_detection.sh
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
_sam3_root = _PROJECT_ROOT / "third_party" / "sam3"
if _sam3_root.is_dir() and str(_sam3_root) not in sys.path:
    sys.path.insert(0, str(_sam3_root))

from src.caliball.dataset.lerobot_dataset import LeRobotDataset
from src.caliball.utils.web_interaction import pick_tracking_point_web, run_sam3_points_web


def ensure_dir(path: Path | str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_video_frames(
    video: np.ndarray,
    output_dir: Path,
    start_idx: int = 0,
    end_idx: Optional[int] = None,
) -> Path:
    output_dir = ensure_dir(output_dir)
    if end_idx is None:
        end_idx = len(video) - 1
    for idx in range(start_idx, end_idx + 1):
        Image.fromarray(video[idx]).save(output_dir / f"frame_{idx:06d}.png")
    print(f"保存帧完成: {output_dir} [{start_idx}, {end_idx}]")
    return output_dir


def exported_frames_complete(output_dir: Path, expected_count: int) -> bool:
    """导出目录是否已有与视频等长的非空 PNG（frame_000000.png …）。"""
    if expected_count <= 0:
        return False
    for idx in range(expected_count):
        p = output_dir / f"frame_{idx:06d}.png"
        if not p.is_file() or p.stat().st_size == 0:
            return False
    return True


def verify_exported_frames(output_dir: Path, expected_count: int) -> None:
    """确认已写入与视频长度一致的 PNG，否则中止。"""
    if expected_count <= 0:
        raise RuntimeError(f"视频长度为 0，无法校验导出帧: {output_dir}")
    missing = []
    for idx in range(expected_count):
        p = output_dir / f"frame_{idx:06d}.png"
        if not p.is_file() or p.stat().st_size == 0:
            missing.append(str(p))
    if missing:
        raise RuntimeError(
            f"帧导出不完整（期望 {expected_count} 张），缺 {len(missing)} 个，例如: {missing[:3]}"
        )


def overlay_mask(image_rgb: np.ndarray, mask: np.ndarray, color: Tuple[int, int, int] = (0, 255, 0), alpha: float = 0.45) -> np.ndarray:
    canvas = np.asarray(image_rgb).copy()
    overlay = canvas.copy()
    m = np.asarray(mask) > 0
    overlay[m] = color
    return cv2.addWeighted(overlay, alpha, canvas, 1 - alpha, 0)


def json_serialize(v: Any) -> Any:
    if isinstance(v, Path):
        return str(v)
    if isinstance(v, dict):
        return {k: json_serialize(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [json_serialize(x) for x in v]
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    return str(v)


def parse_args():
    p = argparse.ArgumentParser(description="外参检测：Web 交互选点 + SAM3 mask，写 manual_label")
    p.add_argument("--task-path", type=str, required=True, help="LeRobot 数据集根目录（本地路径）")
    p.add_argument("--task-name", type=str, default=None, help="默认 os.path.basename(task-path)")
    p.add_argument("--dataset-name", type=str, required=True)
    p.add_argument("--camera-name", type=str, required=True, help="如 observation.images.camera_top")
    p.add_argument("--robot-type", type=str, default="ur5e")
    p.add_argument("--episode-idx", type=int, default=0)
    p.add_argument("--strike", type=int, default=4)
    p.add_argument("--start-idx", type=int, default=0)
    p.add_argument("--end-idx", type=int, default=39)
    p.add_argument("--mask-frame-idx", type=int, default=35)
    p.add_argument(
        "--state-key",
        type=str,
        default="observation.states.joint_position",
        help="LeRobotDataset state_key",
    )
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument(
        "--frame-export-dir",
        type=str,
        default=str(_PROJECT_ROOT / "results" / "extrinsic_notebook" / "frames"),
    )
    p.add_argument(
        "--result-dir",
        type=str,
        default=str(_PROJECT_ROOT / "results" / "extrinsic_notebook"),
    )
    p.add_argument("--manual-label-dir", type=str, default=str(_PROJECT_ROOT / "manual_label"))
    p.add_argument("--sam-bpe-path", type=str, default="third_party/sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz")
    p.add_argument("--sam-ckpt-path", type=str, default="ckpt/sam3/sam3.pt")
    p.add_argument("--host", type=str, default="127.0.0.1", help="监听地址；远程可用 0.0.0.0 + SSH 转发")
    p.add_argument("--tracking-port", type=int, default=8765)
    p.add_argument("--sam-port", type=int, default=8766)
    p.add_argument("--no-browser", action="store_true", help="不自动打开浏览器")
    p.add_argument(
        "--tracking-x",
        type=float,
        default=None,
        help="若与 --tracking-y 同时给出则跳过网页选 tracking point",
    )
    p.add_argument("--tracking-y", type=float, default=None)
    p.add_argument(
        "--mask-npy",
        type=str,
        default=None,
        help="若指定则跳过 SAM 网页，直接加载该 .npy",
    )
    return p.parse_args()


def main():
    args = parse_args()
    task_path = args.task_path
    task_name = args.task_name or Path(task_path).name
    camera_key = args.camera_name

    frame_export_dir = Path(args.frame_export_dir)
    result_dir = Path(args.result_dir)
    manual_label_dir = ensure_dir(args.manual_label_dir)

    bpe = _PROJECT_ROOT / args.sam_bpe_path if not Path(args.sam_bpe_path).is_absolute() else Path(args.sam_bpe_path)
    sam_ckpt = _PROJECT_ROOT / args.sam_ckpt_path if not Path(args.sam_ckpt_path).is_absolute() else Path(args.sam_ckpt_path)

    CONFIG: Dict[str, Any] = {
        "task_path": task_path,
        "task_name": task_name,
        "dataset_name": args.dataset_name,
        "robot_type": args.robot_type,
        "episode_idx": args.episode_idx,
        "camera_name": camera_key,
        "strike": args.strike,
        "start_idx": args.start_idx,
        "end_idx": args.end_idx,
        "mask_frame_idx": args.mask_frame_idx,
        "tracking_point": None,
        "mask_save_path": None,
        "sam_prompt": "robotic arm",
        "device": args.device,
        "frame_export_dir": str(frame_export_dir),
        "result_dir": str(result_dir),
        "dino_ckpt_path": "ckpt/dinov2/dinov2_vitb14_pretrain.pth",
        "dino_repo_dir": "third_party/dinov2",
        "dino_id": "dinov2_vitb14",
        "tracker_repo_dir": "third_party/co-tracker",
        "tracker_id": "cotracker3_offline",
        "tracker_ckpt_path": "ckpt/cotracker/scaled_offline.pth",
        "sam_bpe_path": str(bpe),
        "sam_ckpt_path": str(sam_ckpt),
        "state_key": args.state_key,
    }

    dataset = LeRobotDataset(task_path, state_key=args.state_key)
    episode = dataset[args.episode_idx]
    video = episode["videos"][camera_key]
    joint_angles = episode["states"]
    actions = episode.get("actions")

    video = video[:: args.strike]
    joint_angles = joint_angles[:: args.strike]
    if actions is not None:
        actions = actions[:: args.strike]

    print("task_name   =", task_name)
    print("camera_key  =", camera_key)
    print("video shape =", None if video is None else video.shape)
    print("joint shape =", None if joint_angles is None else joint_angles.shape)

    episode_frame_dir = ensure_dir(
        frame_export_dir / task_name / f"ep_{args.episode_idx:06d}" / camera_key
    )
    n_frames = len(video)
    if n_frames == 0:
        raise SystemExit("视频长度为 0，无法导出帧，中止。")
    if exported_frames_complete(episode_frame_dir, n_frames):
        print(f"帧已存在（{n_frames} 张），跳过导出，继续后续流程: {episode_frame_dir}")
    else:
        save_video_frames(video, episode_frame_dir, start_idx=0, end_idx=n_frames - 1)
        verify_exported_frames(episode_frame_dir, n_frames)
        print(f"已写入并校验 {n_frames} 张帧 -> {episode_frame_dir}")
        print("本次仅导出图片；请再次运行同一命令以进行选点、SAM 与保存 manual_label。")
        sys.exit(0)

    start_idx = int(args.start_idx)
    end_idx = int(args.end_idx)
    if not (0 <= start_idx < len(video)):
        raise SystemExit(f"start_idx={start_idx} 越界，len(video)={len(video)}")
    if not (start_idx < end_idx < len(video)):
        raise SystemExit(f"需要 start_idx < end_idx < len(video)，当前 {start_idx=} {end_idx=} len={len(video)}")

    clip = video[start_idx : end_idx + 1]
    clip_joint = joint_angles[start_idx : end_idx + 1]
    print("clip shape =", clip.shape, "clip_joint =", clip_joint.shape)

    if args.tracking_x is not None and args.tracking_y is not None:
        CONFIG["tracking_point"] = [float(args.tracking_x), float(args.tracking_y)]
        print("tracking_point (CLI) =", CONFIG["tracking_point"])
    else:
        print(f"请在浏览器打开 tracking 页面（若未自动打开则访问 http://{args.host}:{args.tracking_port}/）")
        tx, ty = pick_tracking_point_web(
            clip[0],
            host=args.host,
            port=args.tracking_port,
            open_browser=not args.no_browser,
        )
        CONFIG["tracking_point"] = [float(tx), float(ty)]
        print("tracking_point (web) =", CONFIG["tracking_point"])

    mask_frame_idx = int(args.mask_frame_idx)
    if mask_frame_idx < start_idx or mask_frame_idx > end_idx:
        print(f"mask_frame_idx={mask_frame_idx} 不在 [{start_idx}, {end_idx}]，改为 start_idx")
        mask_frame_idx = start_idx
        CONFIG["mask_frame_idx"] = mask_frame_idx

    mask_output_dir = ensure_dir(
        result_dir / task_name / f"ep_{args.episode_idx:06d}" / "masks"
    )
    result_mask_path = mask_output_dir / f"{camera_key}_{mask_frame_idx:06d}.npy"
    result_overlay_path = mask_output_dir / f"{camera_key}_{mask_frame_idx:06d}_overlay.png"

    tgt_mask_img = video[mask_frame_idx]
    tgt_mask_pil = Image.fromarray(np.asarray(tgt_mask_img))

    if args.mask_npy:
        mask = np.load(args.mask_npy)
        if mask.dtype != np.uint8:
            mask = mask.astype(np.uint8)
        print("已加载 mask:", args.mask_npy, mask.shape)
    else:
        from sam3.model_builder import build_sam3_image_model  # type: ignore[import-not-found]
        from sam3.model.sam3_image_processor import Sam3Processor  # type: ignore[import-not-found]

        print("加载 SAM3 …")
        sam3_model = build_sam3_image_model(
            bpe_path=str(bpe),
            checkpoint_path=str(sam_ckpt),
            device=args.device,
            enable_inst_interactivity=True,
        )
        sam3_processor = Sam3Processor(sam3_model, device=args.device)
        sam3_state = sam3_processor.set_image(tgt_mask_pil)

        def predict_fn(pts: np.ndarray, lbs: np.ndarray) -> np.ndarray:
            masks, _scores, _logits = sam3_model.predict_inst(
                sam3_state,
                point_coords=pts,
                point_labels=lbs,
                multimask_output=False,
            )
            return masks[0].astype(np.uint8)

        print(f"SAM 交互页面: http://{args.host}:{args.sam_port}/ （左=前景 右=背景，保存/取消）")
        mask = run_sam3_points_web(
            np.asarray(tgt_mask_img),
            predict_fn,
            host=args.host,
            port=args.sam_port,
            open_browser=not args.no_browser,
        )
        if mask is None:
            raise SystemExit("已取消 SAM，未保存 mask")

    np.save(result_mask_path, mask.astype(np.uint8))
    Image.fromarray(overlay_mask(tgt_mask_img, mask)).save(result_overlay_path)
    print("结果 mask:", result_mask_path)
    print("结果 overlay:", result_overlay_path)

    dataset_name_fs = args.dataset_name.replace("/", ".")
    filename_prefix = f"{dataset_name_fs}.{task_name}.{camera_key}.{args.episode_idx}"
    manual_mask_path = manual_label_dir / f"{filename_prefix}.mask.npy"
    config_save_path = manual_label_dir / f"{filename_prefix}.config.json"
    mask_overlay_path = manual_label_dir / f"{filename_prefix}.mask_overlay.png"
    tracking_point_vis_path = manual_label_dir / f"{filename_prefix}.tracking_point_vis.png"

    np.save(manual_mask_path, mask.astype(np.uint8))
    print("manual_label mask:", manual_mask_path)

    overlay = overlay_mask(tgt_mask_img, mask)
    Image.fromarray(overlay).save(mask_overlay_path)

    frame0 = np.asarray(video[start_idx])
    if frame0.max() <= 1.0:
        frame0 = (frame0 * 255).astype(np.uint8)
    tp = CONFIG["tracking_point"]
    vis_img = Image.fromarray(frame0)
    draw = ImageDraw.Draw(vis_img)
    r = 7
    draw.ellipse([tp[0] - r, tp[1] - r, tp[0] + r, tp[1] + r], outline=(255, 0, 0), width=3, fill=(255, 0, 0))
    vis_img.save(tracking_point_vis_path)

    CONFIG["mask_save_path"] = str(manual_mask_path)
    with open(config_save_path, "w", encoding="utf-8") as f:
        json.dump(json_serialize(CONFIG), f, indent=2, ensure_ascii=False)
    print("config:", config_save_path)


if __name__ == "__main__":
    main()
