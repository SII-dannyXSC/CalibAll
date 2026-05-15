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
from src.caliball.utils.web_interaction import run_unified_web


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


def scan_datasets(data_root: str) -> list:
    """扫描 data_root 下的可用数据集，返回 datasets_info 列表。"""
    root = Path(data_root)
    if not root.is_dir():
        print(f"[scan] 数据根目录不存在: {root}")
        return []
    results = []
    # 遍历 data_root 下两级目录寻找 LeRobot 格式数据集
    # LeRobot 数据集特征：含 data/ 子目录或 meta/ 子目录
    for sub in sorted(root.iterdir()):
        if not sub.is_dir():
            continue
        # sub 可能是 dataset_name 层（如 robomind.ur_1rgb）
        # 再下一级是 task_name（如 pick_up_can）
        for task_dir in sorted(sub.iterdir()):
            if not task_dir.is_dir():
                continue
            # 检查是否有 LeRobot 数据集特征
            has_data = (task_dir / "data").is_dir() or (task_dir / "meta").is_dir()
            if not has_data:
                continue
            # 扫描 cameras
            cameras = []
            meta_dir = task_dir / "meta"
            if meta_dir.is_dir():
                info_json = meta_dir / "info.json"
                if info_json.is_file():
                    try:
                        import json as _json
                        info = _json.loads(info_json.read_text())
                        cameras = info.get("camera_keys", info.get("video_keys", []))
                    except Exception:
                        pass
            if not cameras:
                # 从 videos/ 目录推断
                videos_dir = task_dir / "videos"
                if videos_dir.is_dir():
                    for vf in videos_dir.iterdir():
                        if vf.suffix == ".mp4":
                            cam = vf.stem.rsplit("_episode_", 1)[0] if "_episode_" in vf.stem else vf.stem
                            if cam not in cameras:
                                cameras.append(cam)
                    cameras.sort()

            results.append({
                "task_path": str(task_dir),
                "task_name": task_dir.name,
                "dataset_name": sub.name,
                "display_name": f"{sub.name}/{task_dir.name}",
                "cameras": cameras,
            })
    print(f"[scan] 扫描到 {len(results)} 个数据集（{data_root}）")
    return results


def parse_args():
    p = argparse.ArgumentParser(description="外参检测：Web 交互选点 + SAM3 mask，写 manual_label")
    p.add_argument("--task-path", type=str, default=None, help="LeRobot 数据集根目录（--config 模式可省略）")
    p.add_argument("--task-name", type=str, default=None, help="默认 os.path.basename(task-path)")
    p.add_argument("--dataset-name", type=str, default=None)
    p.add_argument("--camera-name", type=str, default=None, help="如 observation.images.camera_top")
    p.add_argument("--data-root", type=str, default=None, help="数据根目录，自动扫描可用数据集（如 data/RoboMIND_lerobot_v2.1_sl/）")
    p.add_argument("--robot-type", type=str, default="ur5e")
    p.add_argument("--episode-idx", type=int, default=0)
    p.add_argument("--strike", type=int, default=4)
    p.add_argument("--start-idx", type=int, default=0)
    p.add_argument("--end-idx", type=int, default=39)
    p.add_argument("--mask-frame-idx", type=int, default=None, help="单个 mask 参考帧索引（向后兼容）")
    p.add_argument("--mask-frame-idxs", type=int, nargs="+", default=None, help="多个 mask 参考帧索引，如 --mask-frame-idxs 0 5 10")
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
    p.add_argument("--max-steps", type=int, default=10000, help="Refinement 最大迭代步数")
    p.add_argument("--config", type=str, default=None, help="从已有的 config.json 加载标注参数，跳过 web 交互直接运行 pipeline")
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

    # ── 从 JSON config 加载参数（如果提供） ──
    if args.config:
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        args.task_path = args.task_path or cfg["task_path"]
        args.task_name = args.task_name or cfg.get("task_name")
        args.dataset_name = cfg.get("dataset_name", args.dataset_name)
        args.camera_name = cfg.get("camera_name", args.camera_name)
        args.robot_type = cfg.get("robot_type", args.robot_type)
        args.episode_idx = cfg.get("episode_idx", args.episode_idx)
        args.strike = cfg.get("strike", args.strike)
        args.start_idx = cfg.get("start_idx", args.start_idx)
        args.end_idx = cfg.get("end_idx", args.end_idx)
        # 兼容旧 config 的 mask_frame_idx（单值）和新的 mask_frame_idxs（列表）
        if "mask_frame_idxs" in cfg:
            args.mask_frame_idxs = args.mask_frame_idxs or cfg["mask_frame_idxs"]
        elif "mask_frame_idx" in cfg:
            args.mask_frame_idx = cfg.get("mask_frame_idx", args.mask_frame_idx)
        if "tracking_point" in cfg and cfg["tracking_point"]:
            args.tracking_x = cfg["tracking_point"][0]
            args.tracking_y = cfg["tracking_point"][1]
        if "mask_save_path" in cfg:
            args.mask_npy = args.mask_npy or cfg["mask_save_path"]
        args.state_key = cfg.get("state_key", args.state_key)
        print(f"从 config 加载参数: {args.config}")

    # ── 数据集扫描（--data-root 模式） ──
    datasets_info = None
    if args.data_root:
        datasets_info = scan_datasets(args.data_root)
        # 如果未指定 task-path，使用第一个数据集
        if datasets_info and not args.task_path:
            ds0 = datasets_info[0]
            args.task_path = ds0["task_path"]
            args.dataset_name = args.dataset_name or ds0["dataset_name"]
            if not args.camera_name and ds0.get("cameras"):
                args.camera_name = ds0["cameras"][0]

    if not args.task_path or not args.dataset_name or not args.camera_name:
        raise SystemExit("需要 --task-path, --dataset-name, --camera-name（或使用 --config / --data-root）")

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
        "mask_frame_idxs": None,
        "tracking_point": None,
        "mask_save_paths": None,
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

    # ── 确定默认值 ──
    start_idx = int(args.start_idx)
    end_idx = int(args.end_idx)

    # 多帧 mask：优先 --mask-frame-idxs，其次 --mask-frame-idx，默认 [start_idx]
    if args.mask_frame_idxs is not None:
        mask_frame_idxs = list(args.mask_frame_idxs)
    elif args.mask_frame_idx is not None:
        mask_frame_idxs = [int(args.mask_frame_idx)]
    else:
        mask_frame_idxs = [start_idx]

    default_tracking = None
    if args.tracking_x is not None and args.tracking_y is not None:
        default_tracking = (float(args.tracking_x), float(args.tracking_y))

    default_masks = None
    if args.mask_npy:
        m = np.load(args.mask_npy)
        if m.dtype != np.uint8:
            m = m.astype(np.uint8)
        default_masks = [m]
        print("已加载默认 mask:", args.mask_npy)

    # ── 预加载所有 Pipeline 模型（标注期间模型已就绪） ──
    from omegaconf import OmegaConf
    from src.caliball.coarse_init import CoarseInit
    from src.caliball.refinement import Refinement

    model_config = OmegaConf.load(str(_PROJECT_ROOT / "src" / "caliball" / "config" / "models.yaml"))
    model_config.robot_type = args.robot_type

    print("加载 CoarseInit（DINOv2 + CoTracker）…")
    _coarse_init = CoarseInit(config=model_config)
    _coarse_init.to(args.device)

    print("加载 Refinement…")
    _refinement_init = Refinement(config=model_config)
    print("所有模型已加载")

    # ── 构造 Pipeline 函数（config 模式与 web 模式共享） ──
    def pipeline_fn(annotate_result, update_fn):
        """标注完成后运行 tracking → coarse → refine，通过 update_fn 报告进度。"""

        def _log_update(state):
            """同时更新 web 页面 + 终端 print。"""
            msg = state.get("message", "")
            stage = state.get("stage", "")
            print(f"[pipeline] [{stage}] {msg}")
            update_fn(state)

        s, e = annotate_result["start"], annotate_result["end"]
        mask_refs = annotate_result.get("mask_refs", [annotate_result.get("mask_ref", s)])
        tp = list(annotate_result["tracking_point"])
        masks = annotate_result.get("masks", [annotate_result.get("mask")])

        clip = video[s : e + 1]
        clip_joint = joint_angles[s : e + 1]
        mask_ids = [mr - s for mr in mask_refs]
        print(f"[pipeline] clip: [{s}, {e}], mask_refs={mask_refs}, mask_ids={mask_ids}, tracking={tp}")

        pipe_save = ensure_dir(result_dir / task_name / f"ep_{args.episode_idx:06d}" / "pipeline")
        print(f"[pipeline] save_path: {pipe_save}")

        # 检查机型是否变更
        selected_robot = annotate_result.get("robot_type", args.robot_type)
        if selected_robot != args.robot_type:
            _log_update({"stage": "tracking", "message": f"机型变更为 {selected_robot}，重新加载…"})
            cfg2 = OmegaConf.load(str(_PROJECT_ROOT / "src" / "caliball" / "config" / "models.yaml"))
            cfg2.robot_type = selected_robot
            coarse = CoarseInit(config=cfg2)
            coarse.to(args.device)
            refinement = Refinement(config=cfg2)
        else:
            coarse = _coarse_init
            refinement = _refinement_init

        # Stage 1: Tracking + Coarse
        _log_update({"stage": "tracking", "message": "正在追踪…"})
        extrinsic, K, details = coarse.get_extrinsic(
            video=clip, joint_angles=clip_joint,
            tracking_point=tp, img_idx=0,
            save_path=str(pipe_save), return_details=True,
        )
        print(f"[pipeline] coarse extrinsic:\n{extrinsic}")
        print(f"[pipeline] intrinsic:\n{K}")

        # Tracking 可视化
        pts_2d = details["points_2d"]
        vis = clip[0].copy()
        for pt in pts_2d:
            cv2.circle(vis, (int(pt[0]), int(pt[1])), 3, (0, 255, 0), -1)
        _log_update({"stage": "coarse", "message": "Coarse 外参估计完成", "image": vis})

        # 释放 CoarseInit 显存
        import torch, gc
        del coarse
        gc.collect()
        torch.cuda.empty_cache()
        print("[pipeline] 已释放 CoarseInit 显存")

        # Stage 2: Refinement
        _log_update({"stage": "refine", "message": "开始 Refinement…", "step": 0, "max_steps": args.max_steps})
        print("[pipeline] 调用 refinement.refine() ...")
        print(f"[pipeline] clip.shape={clip.shape}, clip_joint.shape={clip_joint.shape}, mask_ids={mask_ids}, n_masks={len(masks)}")

        def _refine_cb(step, max_steps, loss, overlay, overlays=None, mask_ids=None):
            state = {
                "stage": "refine",
                "message": f"step {step}/{max_steps}  loss={loss:.6f}",
                "step": step, "max_steps": max_steps,
                "image": overlay,
            }
            if overlays is not None:
                state["overlays"] = overlays
            if mask_ids is not None:
                state["mask_ids"] = mask_ids
            _log_update(state)

        output, loss_dict = refinement.refine(
            clip, clip_joint, K, extrinsic, str(pipe_save),
            mask=masks, mask_id=mask_ids, max_steps=args.max_steps,
            progress_callback=_refine_cb,
        )

        # 保存 pipeline 结果（refined extrinsic + intrinsic）
        refined_tsfm = output["tsfm"].detach().cpu().numpy()
        np.save(str(pipe_save / "extrinsic_coarse.npy"), extrinsic)
        np.save(str(pipe_save / "extrinsic_refined.npy"), refined_tsfm)
        np.save(str(pipe_save / "intrinsic.npy"), K)
        CONFIG.update(
            extrinsic_coarse_path=str(pipe_save / "extrinsic_coarse.npy"),
            extrinsic_refined_path=str(pipe_save / "extrinsic_refined.npy"),
            intrinsic_path=str(pipe_save / "intrinsic.npy"),
            pipeline_save_path=str(pipe_save),
        )

        # ── 生成标注可视化视频 ──
        _log_update({"stage": "done", "message": "生成可视化视频…"})
        import torch as _torch
        from src.caliball.pipeline.rendering_optimizer import RBSolver
        from src.caliball.utils.image import add_mask as _add_mask

        H, W = clip.shape[1:3]
        vis_solver = RBSolver(refinement.mesh_paths, H, W, refined_tsfm, device=args.device)
        vis_solver.to(args.device)
        vis_solver.eval()

        K_t = refinement._to_float_tensor(K).unsqueeze(0)

        anno_path = str(pipe_save / "anno_video.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        vw = cv2.VideoWriter(anno_path, fourcc, 15, (W, H))

        last_overlay_rgb = None
        for fi in range(len(clip)):
            link_poses_i = refinement._prepare_link_poses(clip_joint[fi : fi + 1])
            dps = {
                "global_step": 0,
                "mask": _torch.zeros((1, H, W), device=args.device),
                "link_poses": link_poses_i,
                "K": K_t,
            }
            with _torch.no_grad():
                out_v, _ = vis_solver.forward(dps)
            rmask = out_v["rendered_masks"][0].detach().cpu().numpy()
            rmask = (rmask > 0).astype(np.uint8)
            frame_bgr = clip[fi][:, :, ::-1].copy()
            overlay_bgr = _add_mask(frame_bgr, rmask, color=[0, 255, 0], alpha=0.5)
            vw.write(overlay_bgr)
            last_overlay_rgb = overlay_bgr[:, :, ::-1]
            if fi % 5 == 0:
                _log_update({"stage": "done", "message": f"可视化 {fi + 1}/{len(clip)}", "image": last_overlay_rgb})

        vw.release()
        print(f"[pipeline] 标注视频(mp4v): {anno_path}")

        # 转码为 H.264（浏览器兼容）
        import subprocess
        h264_path = str(pipe_save / "anno_video_h264.mp4")
        try:
            import imageio_ffmpeg
            ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        except ImportError:
            ffmpeg_exe = "ffmpeg"
        try:
            subprocess.run(
                [ffmpeg_exe, "-y", "-i", anno_path, "-c:v", "libx264",
                 "-preset", "fast", "-crf", "23", "-pix_fmt", "yuv420p", h264_path],
                check=True, capture_output=True,
            )
            anno_path = h264_path
            print(f"[pipeline] 已转码 H.264: {anno_path}")
        except Exception as e:
            print(f"[pipeline] ffmpeg 转码失败，使用原始视频: {e}")

        _log_update({"stage": "done", "message": "Pipeline 完成！视频已保存",
                      "image": last_overlay_rgb, "video_path": anno_path})

    if args.config:
        # ── Config 模式：跳过 SAM + web，直接运行 pipeline ──
        if default_tracking is None:
            raise SystemExit("config 中缺少 tracking_point")
        if default_masks is None:
            raise SystemExit("config 中缺少 mask（需要 --mask-npy 或 config 中的 mask_save_path）")

        annotate_result = {
            "start": start_idx, "end": end_idx,
            "mask_refs": mask_frame_idxs, "mask_ref": mask_frame_idxs[0],
            "tracking_point": default_tracking,
            "masks": default_masks, "mask": default_masks[0],
            "robot_type": args.robot_type,
        }
        CONFIG.update(
            start_idx=start_idx, end_idx=end_idx, mask_frame_idxs=mask_frame_idxs,
            tracking_point=list(default_tracking),
        )
        masks = default_masks

        def _noop_update(state):
            pass  # pipeline_fn 内部的 _log_update 已经 print

        print("=" * 60)
        print("Config 模式 — 跳过 web 交互，直接运行 pipeline")
        print("=" * 60)
        pipeline_fn(annotate_result, _noop_update)

    else:
        # ── Web 模式：加载 SAM + web 交互 + pipeline ──
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
        _sam3_state = [None]

        def _set_image(pil_img):
            _sam3_state[0] = sam3_processor.set_image(pil_img)

        def _predict(pts, lbs):
            masks, _, _ = sam3_model.predict_inst(
                _sam3_state[0], point_coords=pts, point_labels=lbs, multimask_output=False,
            )
            return masks[0].astype(np.uint8)

        _robot_types = sorted(p.stem for p in (_PROJECT_ROOT / "src" / "caliball" / "config" / "robot").glob("*.yaml"))

        # 构造 datasets_info load_fn（用于 web 动态切换数据集）
        if datasets_info:
            for ds_entry in datasets_info:
                def _make_load_fn(tp, dn):
                    def _load(camera):
                        ds = LeRobotDataset(tp, state_key=args.state_key)
                        ep = ds[args.episode_idx]
                        v = ep["videos"].get(camera)
                        if v is None:
                            return None
                        return v[:: args.strike]
                    return _load
                ds_entry["load_fn"] = _make_load_fn(ds_entry["task_path"], ds_entry["dataset_name"])

        print(f"Web 交互页面: http://{args.host}:{args.tracking_port}/")
        result = run_unified_web(
            video, _predict, _set_image,
            default_start=start_idx,
            default_end=end_idx,
            default_mask_refs=mask_frame_idxs,
            default_tracking=default_tracking,
            default_masks=default_masks,
            host=args.host,
            port=args.tracking_port,
            open_browser=not args.no_browser,
            pipeline_fn=pipeline_fn,
            robot_types=_robot_types,
            default_robot_type=args.robot_type,
            datasets_info=datasets_info,
        )

        start_idx = result["start"]
        end_idx = result["end"]
        mask_frame_idxs = result.get("mask_refs", [result.get("mask_ref", start_idx)])
        masks = result.get("masks", [result.get("mask")])
        CONFIG.update(
            start_idx=start_idx, end_idx=end_idx, mask_frame_idxs=mask_frame_idxs,
            tracking_point=list(result["tracking_point"]),
        )

    if not masks or all(m is None for m in masks):
        raise SystemExit("未获得 mask")

    clip = video[start_idx : end_idx + 1]
    clip_joint = joint_angles[start_idx : end_idx + 1]
    print("clip shape =", clip.shape, "clip_joint =", clip_joint.shape)

    # ── 保存多帧 mask 结果 ──
    mask_output_dir = ensure_dir(
        result_dir / task_name / f"ep_{args.episode_idx:06d}" / "masks"
    )
    for mi, (mfi, msk) in enumerate(zip(mask_frame_idxs, masks)):
        if msk is None:
            continue
        result_mask_path = mask_output_dir / f"{camera_key}_{mfi:06d}.npy"
        result_overlay_path = mask_output_dir / f"{camera_key}_{mfi:06d}_overlay.png"
        tgt_mask_img = video[mfi]
        np.save(result_mask_path, msk.astype(np.uint8))
        Image.fromarray(overlay_mask(tgt_mask_img, msk)).save(result_overlay_path)
        print(f"结果 mask[{mi}]: {result_mask_path}")
        print(f"结果 overlay[{mi}]: {result_overlay_path}")

    dataset_name_fs = args.dataset_name.replace("/", ".")
    filename_prefix = f"{dataset_name_fs}.{task_name}.{camera_key}.{args.episode_idx}"

    # 保存每帧 mask 到 manual_label
    mask_save_paths = []
    for mi, (mfi, msk) in enumerate(zip(mask_frame_idxs, masks)):
        if msk is None:
            continue
        suffix = f".mask_{mfi}" if len(mask_frame_idxs) > 1 else ".mask"
        manual_mask_path = manual_label_dir / f"{filename_prefix}{suffix}.npy"
        mask_overlay_path = manual_label_dir / f"{filename_prefix}{suffix}_overlay.png"
        np.save(manual_mask_path, msk.astype(np.uint8))
        Image.fromarray(overlay_mask(video[mfi], msk)).save(mask_overlay_path)
        mask_save_paths.append(str(manual_mask_path))
        print(f"manual_label mask[{mi}]: {manual_mask_path}")

    tracking_point_vis_path = manual_label_dir / f"{filename_prefix}.tracking_point_vis.png"
    config_save_path = manual_label_dir / f"{filename_prefix}.config.json"

    frame0 = np.asarray(video[start_idx])
    if frame0.max() <= 1.0:
        frame0 = (frame0 * 255).astype(np.uint8)
    tp = CONFIG["tracking_point"]
    vis_img = Image.fromarray(frame0)
    draw = ImageDraw.Draw(vis_img)
    r = 7
    draw.ellipse([tp[0] - r, tp[1] - r, tp[0] + r, tp[1] + r], outline=(255, 0, 0), width=3, fill=(255, 0, 0))
    vis_img.save(tracking_point_vis_path)

    CONFIG["mask_save_paths"] = mask_save_paths
    with open(config_save_path, "w", encoding="utf-8") as f:
        json.dump(json_serialize(CONFIG), f, indent=2, ensure_ascii=False)
    print("config:", config_save_path)


if __name__ == "__main__":
    main()
