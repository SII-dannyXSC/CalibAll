#!/usr/bin/env python3
"""Transfer UMI pose into the ARX base frame.

Pipeline:
  UMI eef pose in UMI base
    -> UMI TCP pose in camera frame, using UMI camera extrinsic
    -> same pose interpreted as ARX TCP in camera frame
    -> ARX TCP pose in ARX base, using inverse ARX camera extrinsic
    -> optional ARX EEF pose in ARX base, using inverse closed-gripper TCP offset

The output pose format is:
  [x, y, z, r1, r2, r3, r4, r5, r6, gripper]
where r1..r6 are the first two rotation-matrix columns.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
from omegaconf import OmegaConf

from caliball.dataset.lerobot_dataset import LeRobotDataset
from caliball.robots import build_robot


def _load_extrinsic(calib_path: str | Path, camera_name: str) -> np.ndarray:
    cfg = OmegaConf.load(calib_path)
    if camera_name not in cfg.cameras:
        raise KeyError(
            f"Camera {camera_name!r} not found in {calib_path}. "
            f"Available: {list(cfg.cameras.keys())}"
        )
    return np.array(
        OmegaConf.to_container(cfg.cameras[camera_name].extrinsic),
        dtype=np.float64,
    )


def _side_slice(side: str) -> slice:
    if side == "left":
        return slice(0, 10)
    if side == "right":
        return slice(10, 20)
    raise ValueError(f"side must be left or right, got {side!r}")


def _matrix_to_rot6d(T: np.ndarray) -> np.ndarray:
    R = np.asarray(T[:3, :3], dtype=np.float64)
    return np.concatenate([R[:, 0], R[:, 1]])


def _matrix_to_pose10(T: np.ndarray, gripper: float) -> np.ndarray:
    return np.concatenate(
        [np.asarray(T[:3, 3], dtype=np.float64), _matrix_to_rot6d(T), [float(gripper)]]
    )


def _arx_eef_to_tcp(arx_robot) -> np.ndarray:
    if not hasattr(arx_robot, "TCP_IN_LINK6_HOM"):
        raise AttributeError(
            "ARX robot must expose TCP_IN_LINK6_HOM. "
            "This script expects arx_x5 with closed-gripper TCP offset."
        )
    T = np.eye(4, dtype=np.float64)
    T[:3, 3] = np.asarray(arx_robot.TCP_IN_LINK6_HOM[:3], dtype=np.float64)
    return T


def _iter_rows(
    states: np.ndarray,
    *,
    start_frame: int,
    limit: int | None,
) -> Iterable[tuple[int, np.ndarray]]:
    end = len(states) if limit is None else min(len(states), start_frame + limit)
    for frame_idx in range(start_frame, end):
        yield frame_idx, states[frame_idx]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Transfer UMI eef/TCP pose to ARX base-frame eef pose."
    )
    p.add_argument("--umi-dataset", default="/home/xiesicheng/data/umi/test4_lerobot2_1")
    p.add_argument("--umi-state-key", default="observation.state.eef_pose")
    p.add_argument("--umi-calib", default="src/caliball/config/calibration/testing_umi.yaml")
    p.add_argument("--umi-camera", default="observation.image.third_view")
    p.add_argument("--arx-calib", default="src/caliball/config/calibration/testing_arx.yaml")
    p.add_argument("--arx-camera", default="observation.image.right_wrist_view")
    p.add_argument("--episode", type=int, default=0)
    p.add_argument("--side", choices=["left", "right"], default="left")
    p.add_argument(
        "--source-frame",
        choices=["tcp", "eef"],
        default="tcp",
        help="UMI frame to transfer into the camera frame.",
    )
    p.add_argument(
        "--target-frame",
        choices=["eef", "tcp", "raw"],
        default="eef",
        help=(
            "Output ARX frame in ARX base. "
            "eef subtracts ARX closed TCP offset; tcp/raw keep the transformed pose."
        ),
    )
    p.add_argument("--start-frame", type=int, default=0)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--output-dir", default="results/umi_to_arx_pose")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    T_cam_umi_base = _load_extrinsic(args.umi_calib, args.umi_camera)
    T_cam_arx_base = _load_extrinsic(args.arx_calib, args.arx_camera)
    T_arx_base_cam = np.linalg.inv(T_cam_arx_base)

    umi_robot = build_robot("umi")
    arx_robot = build_robot("arx_x5")
    T_arx_eef_tcp = _arx_eef_to_tcp(arx_robot)
    T_arx_tcp_eef = np.linalg.inv(T_arx_eef_tcp)

    ds = LeRobotDataset(
        args.umi_dataset,
        state_keys=args.umi_state_key,
        episodes=[args.episode],
        decode_video_keys=[],
    )
    episode = ds[0]
    states = np.asarray(episode["states"], dtype=np.float64)
    if states.ndim != 2 or states.shape[1] < 20:
        raise ValueError(
            f"Expected UMI state shape (T, >=20) from {args.umi_state_key}, "
            f"got {states.shape}"
        )

    block_slice = _side_slice(args.side)
    pose10_rows: list[np.ndarray] = []
    jsonl_path = output_dir / "frames.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for frame_idx, raw_state in _iter_rows(
            states, start_frame=args.start_frame, limit=args.limit
        ):
            umi_block = raw_state[block_slice]
            gripper = float(umi_block[-1])

            if args.source_frame == "tcp":
                T_umi_base_src = umi_robot.fkine_tcp(umi_block)[0]
            else:
                T_umi_base_src = umi_robot.fkine_flange(umi_block)[0]

            T_cam_src = T_cam_umi_base @ T_umi_base_src
            T_arx_base_tcp = T_arx_base_cam @ T_cam_src

            if args.target_frame == "eef":
                T_arx_base_out = T_arx_base_tcp @ T_arx_tcp_eef
            elif args.target_frame in ("tcp", "raw"):
                T_arx_base_out = T_arx_base_tcp
            else:
                raise ValueError(f"Unsupported target frame: {args.target_frame}")

            pose10 = _matrix_to_pose10(T_arx_base_out, gripper)
            pose10_rows.append(pose10)

            record = {
                "frame_index": int(frame_idx),
                "side": args.side,
                "source_frame": args.source_frame,
                "target_frame": args.target_frame,
                "umi_pose10": umi_block.tolist(),
                "umi_base_source_mat": T_umi_base_src.tolist(),
                "camera_source_mat": T_cam_src.tolist(),
                "arx_base_tcp_mat": T_arx_base_tcp.tolist(),
                "arx_base_output_mat": T_arx_base_out.tolist(),
                "arx_pose10": pose10.tolist(),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    pose10_arr = np.stack(pose10_rows, axis=0) if pose10_rows else np.zeros((0, 10))
    np.save(output_dir / "arx_pose10.npy", pose10_arr)

    summary = {
        "umi_dataset": args.umi_dataset,
        "umi_state_key": args.umi_state_key,
        "umi_calib": args.umi_calib,
        "umi_camera": args.umi_camera,
        "arx_calib": args.arx_calib,
        "arx_camera": args.arx_camera,
        "episode": args.episode,
        "side": args.side,
        "source_frame": args.source_frame,
        "target_frame": args.target_frame,
        "start_frame": args.start_frame,
        "num_frames": int(len(pose10_arr)),
        "output_pose_format": "[x, y, z, rot6d_col0, rot6d_col1, gripper]",
        "arx_eef_to_tcp_closed": T_arx_eef_tcp.tolist(),
        "files": {
            "pose10_npy": str(output_dir / "arx_pose10.npy"),
            "frames_jsonl": str(jsonl_path),
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"[OK] frames: {len(pose10_arr)}")
    print(f"[OK] pose10: {output_dir / 'arx_pose10.npy'}")
    print(f"[OK] jsonl:  {jsonl_path}")
    if len(pose10_arr):
        print(f"[INFO] first arx_pose10: {pose10_arr[0].tolist()}")


if __name__ == "__main__":
    main()
