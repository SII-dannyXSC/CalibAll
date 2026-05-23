"""
CalibAll Web — Extrinsic Calibration Web Interface
===================================================

Usage (no arguments needed, all config done in browser):
  PYTHONPATH=. python scripts/caliball_web.py

Optional server arguments:
  PYTHONPATH=. python scripts/caliball_web.py --host 0.0.0.0 --port 8765 --device cuda
"""

from __future__ import annotations

import argparse
import gc
import json
import queue
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_sam3_root = _PROJECT_ROOT / "third_party" / "sam3"
if _sam3_root.is_dir() and str(_sam3_root) not in sys.path:
    sys.path.insert(0, str(_sam3_root))

from caliball.web.app import create_app, run_app
from caliball.web.state import SharedState
from caliball.web.services.image_utils import image_to_data_url, image_to_thumb_url
from caliball.web.services.sam_service import SamService
from caliball.pipeline.extrinsic_pipeline import ExtrinsicPipeline
from caliball.pipeline.result_saver import save_masks, save_tracking_point_vis, save_config
from caliball.utils.image import ensure_dir, save_video_frames, exported_frames_complete, overlay_mask
from caliball.robots import build_robot
from caliball.web.services.dataset_builder import DatasetBuilder


def parse_args():
    p = argparse.ArgumentParser(description="CalibAll Web: extrinsic calibration (all config in browser)")
    p.add_argument("--host", type=str, default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--max-steps", type=int, default=10000)
    p.add_argument("--no-browser", action="store_true")
    p.add_argument("--result-dir", type=str,
                   default=str(_PROJECT_ROOT / "results" / "extrinsic_notebook"))
    p.add_argument("--manual-label-dir", type=str,
                   default=str(_PROJECT_ROOT / "manual_label"))
    p.add_argument("--sam-bpe-path", type=str,
                   default="third_party/sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz")
    p.add_argument("--sam-ckpt-path", type=str, default="ckpt/sam3/sam3.pt")
    p.add_argument("--config", type=str, default=None,
                   help="Run pipeline from saved config.json (no Web)")
    return p.parse_args()


def _overlay_vis(img, msk, pts, labs):
    """Overlay mask + points on frame for SAM visualization."""
    v = np.asarray(img).copy()
    if v.dtype != np.uint8:
        v = (v * 255).astype(np.uint8) if v.max() <= 1.0 else v.astype(np.uint8)
    b = v.copy()
    if msk is not None and msk.size:
        mm = msk > 0
        gg = np.zeros_like(v); gg[..., 1] = 255
        v = (b.astype(np.float32) * 0.55 + gg.astype(np.float32) * 0.45).astype(np.uint8)
        v = np.where(mm[..., None], v, b)
    for (px, py), lb in zip(pts, labs):
        c = (255, 0, 0) if lb == 1 else (0, 0, 255)
        cv2.circle(v, (int(px), int(py)), 6, c, -1)
        cv2.circle(v, (int(px), int(py)), 6, (255, 255, 255), 1)
    return image_to_data_url(v)


def _load_models(device, robot_type, sam_bpe_path, sam_ckpt_path, update_fn=None):
    """Load CoarseInit, Refinement, SAM3 models."""
    from omegaconf import OmegaConf
    from caliball.pipeline import CoarseInit, Refinement
    from caliball.algorithms.recognizer import Recognizer, build_feature_extractor
    from caliball.algorithms.tracker import build_tracker
    from caliball.algorithms.pose_estimator import solve_pnp
    from caliball.algorithms.mask_extractor import Sam3Extractor
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    def _up(msg, pct=0):
        if update_fn:
            update_fn(msg, pct)
        print(f"[loading] {msg}")

    model_config = OmegaConf.load(str(_PROJECT_ROOT / "src" / "caliball" / "config" / "models.yaml"))
    model_config.robot_type = robot_type

    _up("Loading CoarseInit (DINOv2 + CoTracker)...", 20)
    robot = build_robot(robot_type)
    recognizer = Recognizer(build_feature_extractor(model_config))
    tracker = build_tracker(model_config)
    coarse_init = CoarseInit(robot, recognizer, tracker, solve_pnp)
    coarse_init.to(device)

    _up("Loading Refinement...", 50)
    mask_extractor = Sam3Extractor(
        bpe_path=str(_PROJECT_ROOT / sam_bpe_path if not Path(sam_bpe_path).is_absolute() else sam_bpe_path),
        ckpt_path=str(_PROJECT_ROOT / sam_ckpt_path if not Path(sam_ckpt_path).is_absolute() else sam_ckpt_path),
    )
    refinement = Refinement(robot, mask_extractor, robot.MESH_PATHS, device=device)

    bpe = _PROJECT_ROOT / sam_bpe_path if not Path(sam_bpe_path).is_absolute() else Path(sam_bpe_path)
    ckpt = _PROJECT_ROOT / sam_ckpt_path if not Path(sam_ckpt_path).is_absolute() else Path(sam_ckpt_path)
    _up("Loading SAM3...", 75)
    sam3_model = build_sam3_image_model(
        bpe_path=str(bpe), checkpoint_path=str(ckpt),
        device=device, enable_inst_interactivity=True,
    )
    sam3_processor = Sam3Processor(sam3_model, device=device)

    _up("All models loaded", 100)
    return coarse_init, refinement, sam3_model, sam3_processor, model_config


def _run_from_config(args, result_dir, manual_label_dir):
    """Run pipeline directly from a saved config.json (no Web)."""
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    task_path = cfg["task_path"]
    camera_name = cfg["camera_name"]
    robot_type = cfg.get("robot_type", "ur5e")
    episode_idx = int(cfg.get("episode_idx", 0))
    strike = int(cfg.get("strike", 4))
    start_idx = int(cfg.get("start_idx", 0))
    end_idx = int(cfg.get("end_idx", 0))
    mask_frame_idxs = cfg.get("mask_frame_idxs", [start_idx])
    tracking_point = tuple(cfg["tracking_point"])
    task_name = cfg.get("task_name") or Path(task_path).name
    dataset_name = cfg.get("dataset_name") or Path(task_path).parent.name

    # Load masks
    mask_paths = cfg.get("mask_save_paths", [])
    if not mask_paths and cfg.get("mask_save_path"):
        mask_paths = [cfg["mask_save_path"]]
    if not mask_paths:
        raise SystemExit("config missing mask_save_paths")
    masks = [np.load(p).astype(np.uint8) for p in mask_paths]

    print(f"[config] task={task_path}, camera={camera_name}, robot={robot_type}")
    print(f"[config] tracking={tracking_point}, masks={len(masks)}")

    # Load dataset — new YAML-based or legacy state_key fallback
    yaml_filename = cfg.get("yaml_filename")
    if yaml_filename:
        overrides = cfg.get("overrides")
        dataset, ds_info = DatasetBuilder.build(
            yaml_filename, task_path,
            episode_idx=episode_idx,
            overrides=overrides,
        )
        episode = dataset[0]
        video = episode["videos"][camera_name]
        joint_angles = episode["states"]
        video = video[::strike]
        joint_angles = joint_angles[::strike]
    else:
        # Legacy fallback: old config.json with state_key
        from caliball.dataset.lerobot_dataset import LeRobotDataset
        state_key = cfg.get("state_key", "observation.state")
        ds = LeRobotDataset(task_path, state_keys=state_key, episodes=[episode_idx])
        episode = ds[0]
        video = episode["videos"][camera_name][::strike]
        joint_angles = episode["states"][::strike]

    # Load models (no SAM needed)
    from omegaconf import OmegaConf
    from caliball.pipeline import CoarseInit, Refinement
    from caliball.algorithms.recognizer import Recognizer, build_feature_extractor
    from caliball.algorithms.tracker import build_tracker
    from caliball.algorithms.pose_estimator import solve_pnp

    model_config = OmegaConf.load(str(_PROJECT_ROOT / "src" / "caliball" / "config" / "models.yaml"))
    model_config.robot_type = robot_type

    print("[loading] CoarseInit...")
    robot = build_robot(robot_type)
    recognizer = Recognizer(build_feature_extractor(model_config))
    tracker = build_tracker(model_config)
    coarse_init = CoarseInit(robot, recognizer, tracker, solve_pnp)
    coarse_init.to(args.device)
    print("[loading] Refinement...")
    refinement = Refinement(robot, None, robot.MESH_PATHS, device=args.device)

    pipeline = ExtrinsicPipeline(
        coarse_init=coarse_init, refinement=refinement,
        device=args.device, max_steps=args.max_steps,
    )

    # Run pipeline
    pipe_save = ensure_dir(result_dir / task_name / f"ep_{episode_idx:06d}" / "pipeline")
    clip = video[start_idx:end_idx + 1]
    clip_joint = joint_angles[start_idx:end_idx + 1]
    mask_ids = [mr - start_idx for mr in mask_frame_idxs]

    output = pipeline.run(clip, clip_joint, tracking_point, masks, mask_ids, pipe_save)
    print(f"Pipeline done: {output['save_dir']}")

    # Save
    dataset_name_fs = dataset_name.replace("/", ".")
    prefix = f"{dataset_name_fs}.{task_name}.{camera_name}.{episode_idx}"
    ep_result_dir = result_dir / task_name / f"ep_{episode_idx:06d}"

    save_masks(masks, mask_frame_idxs, video, camera_name,
               ep_result_dir, manual_label_dir, prefix, overlay_mask)
    save_tracking_point_vis(
        video[start_idx], tracking_point,
        manual_label_dir / f"{prefix}.tracking_point_vis.png",
    )
    save_config({
        "task_path": task_path, "task_name": task_name,
        "dataset_name": dataset_name, "robot_type": robot_type,
        "episode_idx": episode_idx, "camera_name": camera_name,
        "start_idx": start_idx, "end_idx": end_idx,
        "mask_frame_idxs": mask_frame_idxs,
        "tracking_point": list(tracking_point),
        **({"yaml_filename": yaml_filename, "overrides": cfg.get("overrides")} if yaml_filename else {"state_key": cfg.get("state_key", "observation.state")}),
        **{k: str(v) for k, v in output.items() if k != "loss_dict"},
    }, manual_label_dir / f"{prefix}.config.json")
    print("Done.")


def main():
    args = parse_args()
    result_dir = Path(args.result_dir)
    manual_label_dir = ensure_dir(args.manual_label_dir)

    # ── Config mode: no Web, directly run pipeline ──
    if args.config:
        _run_from_config(args, result_dir, manual_label_dir)
        return

    # ── Gather robot types from registry (exclude composite arm+gripper) ──
    from caliball.robots import list_robots as _list_robots
    from caliball.robots._registry import get_robot_cls as _get_cls
    from caliball.robots._composite import ArmGripperCompositeTF as _CompTF
    _robot_types = [
        rt for rt in _list_robots()
        if not issubclass(_get_cls(rt), _CompTF)
    ]

    # ── Setup Flask (config page first) ──
    shared_state = SharedState()
    cmd_q = queue.Queue()
    pipeline_stop = threading.Event()

    app = create_app(shared_state=shared_state)
    app.config["cmd_q"] = cmd_q
    app.config["pipeline_stop_event"] = pipeline_stop
    app.config["config_context"] = {
        "robot_types": _robot_types,
        "default_robot_type": "ur5e",
        "default_task_path": "",
        "default_dataset_name": "",
        "default_camera_name": "",
        "default_episode_idx": 0,
        "default_strike": 4,
    }

    # Start Flask — config page at /
    srv_thread = run_app(app, host=args.host, port=args.port, open_browser=not args.no_browser)

    # ── Outer loop: config → load → annotate → (reconfig → repeat | finish → exit) ──
    models_loaded = False
    coarse_init = refinement = sam3_model = sam3_processor = model_config = None
    pipeline = sam_svc = None
    _sam3_state = [None]

    def _update_loading(msg, pct=0):
        shared_state.set("loading_status", {"message": msg, "progress": pct, "done": False})

    while True:
        # ── Phase 0: Wait for config ──
        shared_state.set("config_done", False)
        shared_state.set("loading_status", {"message": "Preparing...", "progress": 0, "done": False})
        print("Waiting for dataset configuration in browser...")
        while not shared_state.get("config_done"):
            time.sleep(0.2)

        web_cfg = shared_state.get("config_result")
        task_path = web_cfg["task_path"]
        camera_name = web_cfg["camera_name"]
        robot_type = web_cfg.get("robot_type", "ur5e")
        episode_idx = int(web_cfg.get("episode_idx", 0))
        strike = int(web_cfg.get("strike", 4))
        dataset_name = web_cfg.get("dataset_name") or Path(task_path).parent.name
        task_name = Path(task_path).name
        yaml_filename = web_cfg.get("yaml_filename", "default.yaml")
        overrides = web_cfg.get("overrides")

        print(f"Config: task_path={task_path}, camera={camera_name}, yaml={yaml_filename}")

        # ── Load dataset via DatasetBuilder ──
        _update_loading("Loading dataset...", 5)
        try:
            dataset, ds_info = DatasetBuilder.build(
                yaml_filename, task_path,
                episode_idx=episode_idx,
                overrides=overrides,
            )
            episode = dataset[0]
            video = episode["videos"][camera_name]
            joint_angles = episode["states"]

            video = video[::strike]
            joint_angles = joint_angles[::strike]
            print(f"video shape = {video.shape}, joint shape = {joint_angles.shape}")
        except Exception as e:
            _update_loading(f"Dataset loading failed: {e}", 0)
            print(f"Dataset loading failed: {e}")
            continue

        n_frames = len(video)
        if n_frames == 0:
            _update_loading("Video length is 0", 0)
            continue

        # ── Load models (first time only) ──
        if not models_loaded:
            _update_loading("Loading models...", 10)
            coarse_init, refinement, sam3_model, sam3_processor, model_config = _load_models(
                args.device, robot_type, args.sam_bpe_path, args.sam_ckpt_path, _update_loading,
            )
            pipeline = ExtrinsicPipeline(
                coarse_init=coarse_init, refinement=refinement,
                device=args.device, max_steps=args.max_steps,
            )

            def _set_image(pil_img):
                _sam3_state[0] = sam3_processor.set_image(pil_img)

            def _predict(pts, lbs):
                masks_out, _, _ = sam3_model.predict_inst(
                    _sam3_state[0], point_coords=pts, point_labels=lbs, multimask_output=False,
                )
                return masks_out[0].astype(np.uint8)

            sam_svc = SamService(_predict, _set_image)
            # Initialize Recognizer with default EEF reference
            ref_img_path = _PROJECT_ROOT / "assets" / "test_img" / "source.png"
            if ref_img_path.exists():
                coarse_init._init_recognizer(Image.open(ref_img_path).convert("RGB"), (376, 131))
                print("[web] Recognizer initialized with default reference")
            models_loaded = True
        else:
            if robot_type != model_config.robot_type:
                _update_loading("Updating FK model...", 50)
                model_config.robot_type = robot_type
                new_robot = build_robot(robot_type)
                pipeline.update_robot(new_robot)
            pipeline.reset_intrinsic()

        # ── Prepare annotate page ──
        _update_loading("Generating thumbnails...", 90)
        print(f"[web] Generating {n_frames} thumbnails...")
        thumbs = [image_to_thumb_url(f) for f in video]

        start_idx, end_idx = 0, n_frames - 1
        mask_frame_idxs = [start_idx]
        first_mask_ref = mask_frame_idxs[0]
        initial_overlay = image_to_data_url(video[first_mask_ref])
        tracking_url = image_to_data_url(video[start_idx])
        _set_image(Image.fromarray(video[first_mask_ref]))
        sam_svc.clear()

        robot_html = ""
        if _robot_types:
            _opts = "".join(
                f"<option value='{rt}'{' selected' if rt == robot_type else ''}>{rt}</option>"
                for rt in _robot_types
            )
            robot_html = (
                "<div style='margin-bottom:12px'><label style='font-size:14px'>Robot: "
                f"<select id=rt style='padding:4px 8px;font-size:14px'>{_opts}</select>"
                "</label></div>"
            )

        shared_state.update_overlay(initial_overlay, "Left=foreground Right=background", 0)
        shared_state.clear_pipeline()
        app.config["frames"] = video
        app.config["annotate_context"] = {
            "thumbs": thumbs, "default_start": start_idx, "default_end": end_idx,
            "mask_refs": mask_frame_idxs,
            "tracking_x": "null", "tracking_y": "null",
            "has_default_tracking": False, "has_default_masks": False,
            "has_pipeline": True,
            "initial_overlay": initial_overlay, "tracking_url": tracking_url,
            "robot_html": robot_html, "dataset_html": "",
        }
        shared_state.set("loading_status", {"message": "Loading complete", "progress": 100, "done": True})
        print("[web] Annotate page ready")

        # ── Annotation + Pipeline loop ──
        session_done = False
        result_val = None
        pipeline_output = None
        reconfig_requested = False

        while not session_done:
            # Phase 1: Annotation
            while True:
                try:
                    path, raw = cmd_q.get(timeout=0.15)
                except queue.Empty:
                    continue
                d = json.loads(raw.decode("utf-8")) if raw and raw != b"{}" else {}

                if path == "/api/sam/set_image":
                    fi = int(d["idx"])
                    sam_svc.set_image(fi, video[fi])
                    fs = sam_svc.get_frame_state(fi)
                    shared_state.update_overlay(_overlay_vis(video[fi], fs["mask"], fs["points"], fs["labels"]), f"Switched to frame {fi}", 0)
                elif path == "/api/sam/switch_frame":
                    fi = int(d["fi"])
                    sam_svc.switch_frame(fi, video[fi])
                    fs = sam_svc.get_frame_state(fi)
                    shared_state.update_overlay(_overlay_vis(video[fi], fs["mask"], fs["points"], fs["labels"]), f"Frame {fi}: {len(fs['points'])} pts", len(fs["points"]))
                elif path == "/api/sam/click":
                    fi = int(d.get("fi", sam_svc.active_frame))
                    sam_svc.add_point(fi, float(d["x"]), float(d["y"]), int(d.get("label", 1)), video[fi])
                    fs = sam_svc.get_frame_state(fi)
                    shared_state.update_overlay(_overlay_vis(video[fi], fs["mask"], fs["points"], fs["labels"]), f"Frame {fi}: {len(fs['points'])} pts", len(fs["points"]))
                elif path == "/api/sam/undo":
                    fi = int(d.get("fi", sam_svc.active_frame))
                    sam_svc.undo(fi)
                    fs = sam_svc.get_frame_state(fi)
                    shared_state.update_overlay(_overlay_vis(video[fi], fs["mask"], fs["points"], fs["labels"]), f"Frame {fi}: {len(fs['points'])} pts after undo", len(fs["points"]))
                elif path == "/api/auto_detect":
                    fi = int(d.get("fi", 0))
                    resp_q = app.config.get("auto_detect_resp_q")
                    try:
                        img_pil = Image.fromarray(video[fi])
                        state = sam3_processor.set_image(img_pil)
                        out = sam3_processor.set_text_prompt(state=state, prompt="robotic arm")
                        auto_masks = out["masks"]
                        if len(auto_masks) > 0:
                            mask = auto_masks[0]
                            if hasattr(mask, 'cpu'):
                                mask = mask.cpu().numpy()
                            mask = np.asarray(mask, dtype=np.uint8)
                            if mask.ndim == 3:
                                mask = mask.squeeze()
                            fs = sam_svc.get_frame_state(fi)
                            fs["points"] = []
                            fs["labels"] = []
                            fs["mask"] = mask
                            overlay_url = _overlay_vis(video[fi], mask, [], [])
                            shared_state.update_overlay(overlay_url, f"SAM3 auto-detect done frame {fi}", len(fs["points"]))
                            if resp_q:
                                resp_q.put({"ok": True, "overlay": overlay_url})
                        else:
                            if resp_q:
                                resp_q.put({"ok": False, "error": "No robotic arm detected"})
                    except Exception as e:
                        if resp_q:
                            resp_q.put({"ok": False, "error": str(e)})
                elif path == "/api/auto_tracking":
                    resp_q = app.config.get("auto_tracking_resp_q")
                    try:
                        target_fi = int(d.get("target_fi", 0))
                        target_pil = Image.fromarray(video[target_fi])
                        tx, ty = coarse_init.recognizer.get_uv(target_img_pil=target_pil)
                        if resp_q:
                            resp_q.put({"ok": True, "tracking_x": float(tx), "tracking_y": float(ty)})
                    except Exception as e:
                        if resp_q:
                            resp_q.put({"ok": False, "error": str(e)})
                elif path == "/api/done":
                    mask_refs = [int(x) for x in d.get("maskRefs", d.get("maskRef", []))]
                    if isinstance(mask_refs, int):
                        mask_refs = [mask_refs]
                    result_val = {
                        "start": int(d["start"]), "end": int(d["end"]),
                        "mask_refs": mask_refs,
                        "tracking_point": (float(d["trackingX"]), float(d["trackingY"])),
                        "masks": sam_svc.get_masks_for_refs(mask_refs),
                        "robot_type": d.get("robotType", robot_type),
                    }
                    break
                elif path == "/api/finish":
                    session_done = True; break
                elif path == "/api/reconfig":
                    session_done = True; reconfig_requested = True; break

            if session_done:
                break

            # Phase 2: Pipeline
            if result_val is not None:
                gc.collect()
                s, e = result_val["start"], result_val["end"]
                mask_refs = result_val["mask_refs"]
                # Extend clip range to cover mask ref frames (may be outside start-end)
                clip_start = min(s, *mask_refs)
                clip_end = max(e, *mask_refs)
                clip, clip_joint = video[clip_start:clip_end+1], joint_angles[clip_start:clip_end+1]
                tp, masks = result_val["tracking_point"], result_val["masks"]
                mask_ids = [mr - clip_start for mr in mask_refs]

                sel_robot = result_val.get("robot_type", robot_type)
                if sel_robot != robot_type:
                    robot_type = sel_robot
                    model_config.robot_type = sel_robot
                    new_robot = build_robot(sel_robot)
                    pipeline.update_robot(new_robot)

                pipe_save = ensure_dir(result_dir / task_name / f"ep_{episode_idx:06d}" / "pipeline")

                def _progress(stage, message, **kw):
                    update = {"stage": stage, "message": message}
                    if "image" in kw and isinstance(kw["image"], np.ndarray):
                        update["image"] = image_to_data_url(kw["image"])
                    if "overlays" in kw and isinstance(kw["overlays"], list):
                        update["overlays"] = [image_to_data_url(ov) if isinstance(ov, np.ndarray) else ov for ov in kw["overlays"]]
                    for k in ("step", "max_steps", "video_path", "tracking_video_path", "mask_ids", "warning"):
                        if k in kw: update[k] = kw[k]
                    shared_state.update_pipeline(**update)
                    print(f"[pipeline] [{stage}] {message}")

                try:
                    arm_index = int(result_val.get("armIndex", result_val.get("arm_index", 0)))
                    output = pipeline.run(clip, clip_joint, tp, masks, mask_ids, pipe_save,
                                          progress_fn=_progress, stop_check=lambda: pipeline_stop.is_set(),
                                          arm_index=arm_index, full_video=video, full_joint_angles=joint_angles,
                                          tracking_img_idx=s - clip_start)
                    pipeline_output = output
                    def _fmt(m): return np.array2string(np.array(m), precision=6, suppress_small=True)

                    # Generate calibration yaml immediately so download works
                    import yaml as _yaml
                    K = np.array(output["intrinsic"])
                    T = np.array(output["extrinsic_refined"])
                    calib_data = {"dataset": dataset_name, "cameras": {camera_name: {"intrinsic": K.tolist(), "extrinsic": T.tolist()}}}
                    calib_filename = dataset_name.replace("/", "_").replace(".", "_") + ".yaml"
                    calib_path = pipe_save / calib_filename
                    calib_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(calib_path, "w") as f:
                        _yaml.dump(calib_data, f, default_flow_style=None, allow_unicode=True)
                    print(f"Calibration yaml saved: {calib_path}")

                    shared_state.update_pipeline(
                        stage="done", message="Pipeline complete!",
                        video_path=output.get("anno_video_path"),
                        tracking_video_path=output.get("tracking_video_path"),
                        intrinsic_path=str(pipe_save / "intrinsic.npy"),
                        extrinsic_coarse_path=str(pipe_save / "extrinsic_coarse.npy"),
                        extrinsic_refined_path=str(pipe_save / "extrinsic_refined.npy"),
                        calibration_path=str(calib_path),
                        intrinsic_str=_fmt(output["intrinsic"]),
                        extrinsic_coarse_str=_fmt(output["extrinsic_coarse"]),
                        extrinsic_refined_str=_fmt(output["extrinsic_refined"]),
                    )
                except Exception as exc:
                    import traceback; traceback.print_exc()
                    shared_state.update_pipeline(stage="done", message=f"Pipeline error: {exc}")

            # Phase 3: Wait for restart/finish/reconfig
            while True:
                try:
                    path, raw = cmd_q.get(timeout=0.15)
                except queue.Empty:
                    continue
                if path == "/api/finish":
                    session_done = True; break
                elif path == "/api/reconfig":
                    session_done = True; reconfig_requested = True; break
                elif path == "/api/restart":
                    break

            if not session_done:
                sam_svc.clear()
                _set_image(Image.fromarray(video[first_mask_ref]))
                shared_state.reset(initial_overlay)
                shared_state.clear_pipeline()
                pipeline_stop.clear()
                print("[web] Re-annotate — page reset")

        # ── Save results ──
        if result_val is not None and pipeline_output is not None:
            ep_result_dir = result_dir / task_name / f"ep_{episode_idx:06d}"
            dataset_name_fs = dataset_name.replace("/", ".")
            prefix = f"{dataset_name_fs}.{task_name}.{camera_name}.{episode_idx}"

            mask_save_paths = save_masks(result_val["masks"], result_val["mask_refs"], video, camera_name, ep_result_dir, manual_label_dir, prefix, overlay_mask)
            save_tracking_point_vis(video[result_val["start"]], result_val["tracking_point"], manual_label_dir / f"{prefix}.tracking_point_vis.png")
            save_config({
                "task_path": task_path, "task_name": task_name,
                "dataset_name": dataset_name, "robot_type": robot_type,
                "episode_idx": episode_idx, "camera_name": camera_name,
                "strike": strike,
                "yaml_filename": yaml_filename,
                "overrides": overrides,
                "start_idx": result_val["start"], "end_idx": result_val["end"],
                "mask_frame_idxs": result_val["mask_refs"],
                "tracking_point": list(result_val["tracking_point"]),
                "mask_save_paths": mask_save_paths,
            }, manual_label_dir / f"{prefix}.config.json")
            print(f"Config saved: manual_label/{prefix}.config.json")

        # ── Reconfig or exit ──
        if reconfig_requested:
            print("[web] Returning to config page...")
            pipeline_stop.clear()
            while not cmd_q.empty():
                try: cmd_q.get_nowait()
                except queue.Empty: break
            continue
        else:
            break

    print("Done.")


if __name__ == "__main__":
    main()
