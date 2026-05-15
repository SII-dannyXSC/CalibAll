"""
check_robot_urdf.py

读取任务 YAML，实例化 TF 模型，用零关节角做 FK 并导出 mesh 点云 PLY。
用于验证 URDF/mesh 配置是否正确。

示例：
    PYTHONPATH=. python scripts/check_robot_urdf.py \\
        --config src/caliball/config/berkeley_autolab_ur5.yaml

    PYTHONPATH=. python scripts/check_robot_urdf.py \\
        --config src/caliball/config/rdt_aloha.yaml \\
        --joints 0 0 0 0 0 0 0 0.04 0 0 0 0 0 0 0 0.04
"""
import argparse
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.caliball.config import compose_job_config_from_path, instantiate_tf
from src.caliball.robot.arm_gripper_composite import ArmGripperCompositeTF
from src.caliball.robot.dual_arm.dual_arm_base import DualArmTF
from src.caliball.utils.mesh_loader import _get_mesh_paths


def default_state(tf) -> np.ndarray:
    """根据 TF 类型构造默认关节向量（零位 + 夹爪闭合）。"""
    if isinstance(tf, ArmGripperCompositeTF):
        n = tf.arm_joint_num + 1
        state = np.zeros(n, dtype=np.float64)
        state[-1] = float(tf.gripper_closed_q)
        return state
    if isinstance(tf, DualArmTF):
        # 左右各用 n_left_joints 个零（假设对称）
        n = tf.n_left_joints * 2
        return np.zeros(n, dtype=np.float64)
    # 通用兜底：从 fkine_all 试探
    for n in range(6, 20):
        try:
            tf.fkine_all(np.zeros(n, dtype=np.float64))
            return np.zeros(n, dtype=np.float64)
        except Exception:
            continue
    raise TypeError(f"无法推断 {type(tf).__name__} 的关节数，请手动指定 --joints")


def parse_args():
    p = argparse.ArgumentParser(description="检查机器人 URDF/mesh 配置，导出点云 PLY")
    p.add_argument("--config", default="src/caliball/config/berkeley_autolab_ur5.yaml",
                   help="任务 YAML 路径")
    p.add_argument("--output", default="/tmp/check_robot.ply",
                   help="输出 PLY 路径 (默认 /tmp/check_robot.ply)")
    p.add_argument("--joints", type=float, nargs="+", default=None,
                   help="关节角度列表（弧度）")
    return p.parse_args()


def main():
    import trimesh

    args = parse_args()
    cfg = compose_job_config_from_path(args.config, project_root=_REPO_ROOT)

    print(f"[INFO] config: {args.config}")
    print(f"  tf: {cfg.tf.get('_target_')}")

    tf = instantiate_tf(cfg)
    mesh_paths = _get_mesh_paths(cfg.robot)

    if args.joints is not None:
        state = np.array(args.joints, dtype=np.float64)
    else:
        state = default_state(tf)
    print(f"  joints ({len(state)}): {state.tolist()}")

    tf_all = tf.fkine_all(state)  # (n_arms, n_links, 4, 4)
    if tf_all.ndim == 3:
        tf_list = tf_all  # (n_links, 4, 4) 单臂无 arm 维
    elif tf.n_arms > 1:
        tf_list = np.concatenate([tf_all[i] for i in range(tf.n_arms)], axis=0)
    else:
        tf_list = tf_all[0]

    all_points = []
    for link_idx, mesh_path in enumerate(mesh_paths):
        mp = Path(mesh_path)
        mesh_full = mp if mp.is_absolute() else _REPO_ROOT / mp
        mesh = trimesh.load(str(mesh_full), force="mesh")
        points = mesh.sample(10000)
        points_hom = np.hstack([points, np.ones((points.shape[0], 1))])
        points_tf = (tf_list[link_idx] @ points_hom.T).T[..., :3]
        all_points.append(points_tf)
        print(f"  [{link_idx}] {mp.name}")

    all_points = np.concatenate(all_points, axis=0)
    out_path = Path(args.output)
    trimesh.points.PointCloud(all_points).export(str(out_path))
    print(f"[OK] {len(all_points)} points → {out_path}")


if __name__ == "__main__":
    main()
