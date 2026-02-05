from caliball.robot.franka import FrankaTF
import numpy as np
from omegaconf import OmegaConf
import trimesh

tf = FrankaTF()

state = np.zeros((1,8))
# state = np.array([[ -0.10277671,  0.19634955,  0.08743691,  0.0076699 ,  1.23332059,
#     -0.02607767, 0, 0]])
# state = np.array([[ 0.        , -0.3       ,  0.        , -2.2       ,  0.        ,
#     2.        ,  0.78539816,  0.        ]])
# import pdb; pdb.set_trace()
state = np.hstack([state, np.ones((state.shape[0], 1))])
tf_list = tf.fkine_all(state)[0]

mesh_cfg = OmegaConf.load("caliball/config/robot/franka.yaml")
mesh_paths = mesh_cfg.mesh_paths

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
