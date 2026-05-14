"""
用 Hydra compose 读取任务 YAML（含 defaults + 插值），实例化 ``tf`` 并导出点云 PLY。

示例：
    PYTHONPATH=. python scripts/check_robot_from_config.py \\
        --config src/caliball/config/berkeley_autolab_ur5.yaml
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import trimesh

from src.caliball.robot.arm_gripper_composite import ArmGripperCompositeTF

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.caliball.config import compose_job_config_from_path, instantiate_tf


def default_state_for_tf(tf) -> np.ndarray:
    """根据 TF 类型构造一行关节向量（臂 + 单维夹爪标量等）。"""
    if isinstance(tf, ArmGripperCompositeTF):
        n = tf.arm_joint_num + 1
        state = np.zeros((1, n), dtype=np.float64)
        state[0, -1] = float(tf.gripper_closed_q) if n > tf.arm_joint_num else 0.0
        return state
    raise TypeError(f"未支持的 tf 类型: {type(tf)!r}，请扩展 default_state_for_tf")


def parse_args():
    p = argparse.ArgumentParser(description="Hydra 读取任务 config 并检查机器人 mesh 点云")
    p.add_argument(
        "--config",
        type=str,
        default="src/caliball/config/berkeley_autolab_ur5.yaml",
        help="任务 YAML 路径（相对项目根或绝对路径）",
    )
    p.add_argument(
        "--output",
        type=str,
        default="test_from_config.ply",
        help="输出点云 PLY 路径",
    )
    p.add_argument(
        "--joints",
        type=float,
        nargs="+",
        default=None,
        help="关节角度列表（弧度），如 --joints 0 0.5 -0.3 0 0 0 0 0.04",
    )
    return p.parse_args()


def main():
    args = parse_args()
    cfg = compose_job_config_from_path(args.config, project_root=_REPO_ROOT)

    print("[Hydra] 已加载:", args.config)
    print("  dataset_name:", cfg.get("dataset_name"))
    print("  tf._target_:", cfg.tf.get("_target_"))

    tf = instantiate_tf(cfg)
    robot_config = cfg.robot
    mesh_paths = robot_config.mesh_paths

    if args.joints is not None:
        state = np.array(args.joints, dtype=np.float64).reshape(1, -1)
        print(f"  joints: {args.joints}")
    else:
        state = default_state_for_tf(tf)
        print(f"  joints: default (zeros + gripper closed)")
    tf_list = tf.fkine_all(state)[0]

    all_points = []
    for link_idx, mesh_path in enumerate(mesh_paths):
        mp = Path(mesh_path)
        mesh_full = mp if mp.is_absolute() else _REPO_ROOT / mp
        print(link_idx, mesh_full)
        mesh = trimesh.load(str(mesh_full), force="mesh")
        points = mesh.sample(10000)
        points_hom = np.hstack([points, np.ones((points.shape[0], 1))])
        point_tg = (tf_list[link_idx] @ points_hom.T).T[..., :3]
        all_points.append(point_tg)

    all_points = np.concatenate(all_points, axis=0)
    cloud = trimesh.points.PointCloud(all_points)
    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = _REPO_ROOT / out_path
    cloud.export(str(out_path))
    print("[OK] 点云已写入", out_path)


if __name__ == "__main__":
    main()
