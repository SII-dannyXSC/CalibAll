import numpy as np
import cv2

from caliball.pipeline.point_tracker import Tracker

def solve_pnp(points_3d, points_2d, camera_matrix, method=cv2.SOLVEPNP_ITERATIVE):
    dist_coeffs = np.zeros(shape=[8, 1], dtype='float64')

    assert points_3d.shape[0] == points_2d.shape[0], 'points 3D and points 2D must have same number of vertices'
    if method==cv2.SOLVEPNP_EPNP:
        points_3d=np.expand_dims(points_3d, 0)
        points_2d=np.expand_dims(points_2d, 0)

    points_2d = np.ascontiguousarray(points_2d.astype(np.float64))
    points_3d = np.ascontiguousarray(points_3d.astype(np.float64))
    camera_matrix = camera_matrix.astype(np.float64)
    _, R_exp, t = cv2.solvePnP(points_3d,
                            points_2d,
                            camera_matrix,
                            dist_coeffs,
                            flags=method)
    R, _ = cv2.Rodrigues(R_exp)
    # 点从世界坐标系变化到相机坐标系的旋转矩阵，描述相机怎么移动到原点
    w2c = np.eye(4)
    w2c[:3, :3] = R
    w2c[:3, 3:] = t
    return w2c
        



