import sys
from pathlib import Path

import numpy as np
import trimesh

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.caliball.robot import build_robot
from src.caliball.config import build_robot_config

# 选择机器人类型:
# "franka" | "franka_panda_hand" | "franka_robotiq" | "ur5e" | "ur5e_robotiq" |
# "aloha" | "aloha_cobot_magic" | "arx5_robotwin" | "xarm7_with_gripper"
robot_type = "ur5e"

config = type("", (), {})()
config.robot_type = robot_type
robot_config = build_robot_config(config)
tf = build_robot(config, robot_config)

# 关节向量维度需与对应 TF 一致
if robot_type == "franka":
    state = np.zeros((1, 8))
    state = np.hstack([state, np.ones((state.shape[0], 1))])
elif robot_type == "franka_panda_hand":
    state = np.zeros((1, 8))
    state[0, :7] = [0.0, 0.0, 0.0, -1.5, 0.0, 1.5, 0.0]
    # state[0, 7] = 0.02
    state[0, 7] = 0.00
elif robot_type == "franka_robotiq":
    state = np.array([[0.0, 0.0, 0.0, -1.5, 0.0, 1.5, 0.0, 0.8]])
elif robot_type == "xarm7_with_gripper":
    state = np.zeros((1, 8))
    state[0, 7] = 0.0
    # state[0, 7] = 0.0
elif robot_type == "ur5e":
    # state = np.zeros((1, 6))
    state = np.array([[ 2.199691  , -1.6051203 , -1.5302501 , -1.514033  ,  1.5815353 , 1.3117591 ,  0.01960784]])
elif robot_type == "ur5e_robotiq":
    state = np.zeros((1, 7))
    state[0, 6] = 0.0
elif robot_type in ("aloha", "aloha_cobot_magic", "arx5_robotwin"):
    state = np.zeros((1, 14))
    state[0, 6] = 0.1
    state[0, 7:14] = [0.2871, 2.0472, 1.8923, -1.1164, -0.0345, 0.0563, 0.1]
else:
    raise ValueError(f"Unknown robot_type: {robot_type}")

tf_list = tf.fkine_all(state)[0]
mesh_paths = robot_config.mesh_paths

all_points = []
for link_idx, mesh_path in enumerate(mesh_paths):
    print(link_idx, mesh_path)
    mesh = trimesh.load(mesh_path, force = 'mesh')
    points = mesh.sample(10000)
    
    # points_hom = np.hstack([points, np.ones((points.shape[0], 1))])
    points_hom = np.hstack([points, np.ones((points.shape[0], 1))])
    point_tg = (tf_list[link_idx] @ points_hom.T).T
    point_tg = point_tg[...,:3]
    all_points.append(point_tg)
    print(mesh_path)

all_points = np.array(all_points)
# cloud = trimesh.points.PointCloud(all_points.reshape(-1,3)).export("test.ply")
all_points = all_points.reshape(-1,3)
cloud = trimesh.points.PointCloud(all_points)
cloud.export("./test.ply")
