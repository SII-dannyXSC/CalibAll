"""
check_robot_axes.py

导出 robot mesh 点云 + link 坐标轴 PLY，用于检查 URDF 坐标系方向。

示例：
    PYTHONPATH=src python scripts/check_robot_axes.py \
        --config src/caliball/config/demo_umi.yaml \
        --output results/umi_axes_open.ply \
        --joints 0.08

    # 查看原始 URDF link frame，不使用 UmiTF 的 RENDER_ALIGN_T
    PYTHONPATH=src python scripts/check_robot_axes.py \
        --config src/caliball/config/demo_umi.yaml \
        --output results/umi_axes_open_raw.ply \
        --joints 0.08 \
        --raw-urdf
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent

from caliball.config import compose_job_config_from_path, instantiate_tf
from caliball.robots._composite import ArmGripperCompositeTF
from caliball.robots._dual_arm import DualArmTF
from caliball.utils.mesh_loader import _get_mesh_paths


AXIS_COLORS = np.array(
    [
        [255, 40, 40, 255],   # X red
        [40, 220, 40, 255],   # Y green
        [40, 120, 255, 255],  # Z blue
    ],
    dtype=np.uint8,
)
MESH_COLOR = np.array([180, 180, 180, 255], dtype=np.uint8)
ORIGIN_COLOR = np.array([255, 255, 255, 255], dtype=np.uint8)
WORLD_ORIGIN_COLOR = np.array([255, 220, 40, 255], dtype=np.uint8)


def default_state(tf) -> np.ndarray:
    """根据 TF 类型构造默认关节向量（零位 + 夹爪闭合）。"""
    if isinstance(tf, ArmGripperCompositeTF):
        n = tf.arm_joint_num + 1
        state = np.zeros(n, dtype=np.float64)
        state[-1] = float(tf.gripper_closed_q)
        return state
    if isinstance(tf, DualArmTF):
        return np.zeros(tf.n_left_joints * 2, dtype=np.float64)
    for n in range(1, 20):
        try:
            tf.fkine_all(np.zeros(n, dtype=np.float64))
            return np.zeros(n, dtype=np.float64)
        except Exception:
            continue
    raise TypeError(f"无法推断 {type(tf).__name__} 的关节数，请手动指定 --joints")


def parse_args():
    p = argparse.ArgumentParser(description="导出 robot mesh 点云 + link 坐标轴 PLY")
    p.add_argument("--config", default="src/caliball/config/demo_umi.yaml", help="任务 YAML 路径")
    p.add_argument("--output", default="results/check_robot_axes.ply", help="输出 PLY 路径")
    p.add_argument("--joints", type=float, nargs="+", default=None, help="关节/state 列表")
    p.add_argument("--axis-length", type=float, default=0.06, help="坐标轴长度，单位米")
    p.add_argument("--axis-points", type=int, default=80, help="每根轴采样点数")
    p.add_argument("--origin-axis-length", type=float, default=0.12, help="全局原点坐标轴长度，单位米")
    p.add_argument("--points-per-link", type=int, default=4000, help="每个 mesh 采样点数")
    p.add_argument("--no-origin-axes", action="store_true", help="不额外绘制 [0,0,0] 全局坐标轴")
    p.add_argument(
        "--raw-urdf",
        action="store_true",
        help="显示原始 URDF link frame，不使用 UmiTF 等 wrapper 的渲染对齐",
    )
    return p.parse_args()


def flatten_rendered_link_poses(tf, state: np.ndarray) -> np.ndarray:
    tf_all = tf.fkine_all(state)
    if tf_all.ndim == 3:
        return tf_all
    if getattr(tf, "n_arms", 1) > 1:
        return np.concatenate([tf_all[i] for i in range(tf_all.shape[0])], axis=0)
    return tf_all[0]


def raw_urdf_link_poses(tf, state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """返回 (link_frames, visual_frames)，仅适用于有 ``robot`` 和 ``name_list`` 的 RTB wrapper。"""
    if not hasattr(tf, "robot") or not hasattr(tf, "name_list"):
        raise TypeError("--raw-urdf requires TF with robot and name_list attributes")

    q_full = tf._expand_q(state) if hasattr(tf, "_expand_q") else np.asarray(state, dtype=np.float64)
    link_names = list(tf.name_list)
    link_frames = np.array([tf.robot.fkine(q_full, end=name).A for name in link_names])
    visual_frames = link_frames.copy()

    mesh_adjust = getattr(tf, "mesh_adjust", {})
    for i, name in enumerate(link_names):
        if name in mesh_adjust:
            visual_frames[i] = visual_frames[i] @ mesh_adjust[name]
    return link_frames, visual_frames


def axis_points(
    frames: np.ndarray,
    axis_length: float,
    axis_points_per_axis: int,
    origin_color: np.ndarray = ORIGIN_COLOR,
) -> tuple[np.ndarray, np.ndarray]:
    points = []
    colors = []
    ts = np.linspace(0.0, axis_length, axis_points_per_axis)

    for T in frames:
        origin = T[:3, 3]
        points.append(origin[np.newaxis])
        colors.append(origin_color[np.newaxis])
        for axis_idx in range(3):
            direction = T[:3, axis_idx]
            pts = origin[np.newaxis] + ts[:, np.newaxis] * direction[np.newaxis]
            points.append(pts)
            colors.append(np.repeat(AXIS_COLORS[axis_idx][np.newaxis], len(ts), axis=0))

    return np.concatenate(points, axis=0), np.concatenate(colors, axis=0)


def main():
    import trimesh

    args = parse_args()
    cfg = compose_job_config_from_path(args.config, project_root=_REPO_ROOT)
    tf = instantiate_tf(cfg)
    mesh_paths = _get_mesh_paths(tf if hasattr(cfg, "robot_type") else cfg.robot)

    state = np.array(args.joints, dtype=np.float64) if args.joints is not None else default_state(tf)
    print(f"[INFO] config: {args.config}")
    print(f"  tf: {cfg.get('robot_type') or type(tf).__name__}")
    print(f"  joints ({len(state)}): {state.tolist()}")
    print(f"  raw_urdf: {args.raw_urdf}")

    if args.raw_urdf:
        axis_frames, mesh_frames = raw_urdf_link_poses(tf, state)
    else:
        mesh_frames = flatten_rendered_link_poses(tf, state)
        axis_frames = mesh_frames

    all_points = []
    all_colors = []
    for link_idx, mesh_path in enumerate(mesh_paths):
        mp = Path(mesh_path)
        mesh_full = mp if mp.is_absolute() else _REPO_ROOT / mp
        mesh = trimesh.load(str(mesh_full), force="mesh")
        points = mesh.sample(args.points_per_link)
        points_hom = np.hstack([points, np.ones((points.shape[0], 1))])
        points_tf = (mesh_frames[link_idx] @ points_hom.T).T[..., :3]
        all_points.append(points_tf)
        all_colors.append(np.repeat(MESH_COLOR[np.newaxis], len(points_tf), axis=0))
        print(f"  [{link_idx}] {mp.name}")

    pts_axis, colors_axis = axis_points(axis_frames, args.axis_length, args.axis_points)
    all_points.append(pts_axis)
    all_colors.append(colors_axis)

    out_points = np.concatenate(all_points, axis=0)
    out_colors = np.concatenate(all_colors, axis=0)

    if not args.no_origin_axes:
        world_axis_pts, world_axis_colors = axis_points(
            np.eye(4, dtype=np.float64)[np.newaxis],
            args.origin_axis_length,
            max(args.axis_points * 2, 1),
            WORLD_ORIGIN_COLOR,
        )
        out_points = np.concatenate([out_points, world_axis_pts], axis=0)
        out_colors = np.concatenate([out_colors, world_axis_colors], axis=0)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    trimesh.points.PointCloud(out_points, colors=out_colors).export(str(out_path))
    print(f"[OK] {len(out_points)} colored points -> {out_path}")
    print("  link axes: X=red, Y=green, Z=blue, link origins=white")
    if not args.no_origin_axes:
        print("  world origin axes: X=red, Y=green, Z=blue, origin=yellow")


if __name__ == "__main__":
    main()
