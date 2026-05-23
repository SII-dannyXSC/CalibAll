# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Merged from caliball.utils.utils_3d and caliball.utils.pytorch3d_se3

import cv2
import loguru
import numpy as np
import pytorch3d.transforms
import torch
from multipledispatch import dispatch
from scipy.spatial.transform import Rotation
from typing import Tuple
from pytorch3d.transforms.so3 import hat, so3_log_map
from trimesh.caching import TrackedArray


# ---------------------------------------------------------------------------
# Inlined from caliball.utils.pn_utils (only the ``to_array`` helper)
# ---------------------------------------------------------------------------

def to_array(x, dtype=float):
    if isinstance(x, np.ndarray):
        return x.astype(dtype)
    elif isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy().astype(dtype)
    elif isinstance(x, list):
        return [to_array(a) for a in x]
    elif isinstance(x, dict):
        return {k: to_array(v) for k, v in x.items()}
    elif isinstance(x, TrackedArray):
        return np.array(x)
    else:
        return x


# ===========================================================================
# Functions originally from caliball.utils.pytorch3d_se3
# ===========================================================================

def _so3_exp_map(
    log_rot: torch.Tensor, eps: float = 0.0001
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    A helper function that computes the so3 exponential map and,
    apart from the rotation matrix, also returns intermediate variables
    that can be re-used in other functions.
    """
    _, dim = log_rot.shape
    if dim != 3:
        raise ValueError("Input tensor shape has to be Nx3.")

    nrms = (log_rot * log_rot).sum(1)
    # phis ... rotation angles
    rot_angles = torch.clamp(nrms, eps).sqrt()
    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
    rot_angles_inv = 1.0 / rot_angles
    fac1 = rot_angles_inv * rot_angles.sin()
    fac2 = rot_angles_inv * rot_angles_inv * (1.0 - rot_angles.cos())
    skews = hat(log_rot)
    skews_square = torch.bmm(skews, skews)

    R = (
        fac1[:, None, None] * skews
        # pyre-fixme[16]: `float` has no attribute `__getitem__`.
        + fac2[:, None, None] * skews_square
        + torch.eye(3, dtype=log_rot.dtype, device=log_rot.device)[None]
    )

    return R, rot_angles, skews, skews_square


def _se3_V_matrix(
    log_rotation: torch.Tensor,
    log_rotation_hat: torch.Tensor,
    log_rotation_hat_square: torch.Tensor,
    rotation_angles: torch.Tensor,
    eps: float = 1e-4,
) -> torch.Tensor:
    """
    A helper function that computes the "V" matrix from [1], Sec 9.4.2.
    [1] https://jinyongjeong.github.io/Download/SE3/jlblanco2010geometry3d_techrep.pdf
    """

    V = (
        torch.eye(3, dtype=log_rotation.dtype, device=log_rotation.device)[None]
        + log_rotation_hat
        # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and `int`.
        * ((1 - torch.cos(rotation_angles)) / (rotation_angles**2))[:, None, None]
        + (
            log_rotation_hat_square
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            * ((rotation_angles - torch.sin(rotation_angles)) / (rotation_angles**3))[
                :, None, None
            ]
        )
    )

    return V


def _get_se3_V_input(log_rotation: torch.Tensor, eps: float = 1e-4):
    """
    A helper function that computes the input variables to the `_se3_V_matrix`
    function.
    """
    # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and `int`.
    nrms = (log_rotation**2).sum(-1)
    rotation_angles = torch.clamp(nrms, eps).sqrt()
    log_rotation_hat = hat(log_rotation)
    log_rotation_hat_square = torch.bmm(log_rotation_hat, log_rotation_hat)
    return log_rotation, log_rotation_hat, log_rotation_hat_square, rotation_angles


def _p3d_se3_exp_map(log_transform: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """
    Convert a batch of logarithmic representations of SE(3) matrices `log_transform`
    to a batch of 4x4 SE(3) matrices using the exponential map.
    See e.g. [1], Sec 9.4.2. for more detailed description.

    A SE(3) matrix has the following form:
        ```
        [ R 0 ]
        [ T 1 ] ,
        ```
    where `R` is a 3x3 rotation matrix and `T` is a 3-D translation vector.
    SE(3) matrices are commonly used to represent rigid motions or camera extrinsics.

    In the SE(3) logarithmic representation SE(3) matrices are
    represented as 6-dimensional vectors `[log_translation | log_rotation]`,
    i.e. a concatenation of two 3D vectors `log_translation` and `log_rotation`.

    The conversion from the 6D representation to a 4x4 SE(3) matrix `transform`
    is done as follows:
        ```
        transform = exp( [ hat(log_rotation) 0 ]
                         [   log_translation 1 ] ) ,
        ```
    where `exp` is the matrix exponential and `hat` is the Hat operator [2].

    Note that for any `log_transform` with `0 <= ||log_rotation|| < 2pi`
    (i.e. the rotation angle is between 0 and 2pi), the following identity holds:
    ```
    se3_log_map(se3_exponential_map(log_transform)) == log_transform
    ```

    The conversion has a singularity around `||log(transform)|| = 0`
    which is handled by clamping controlled with the `eps` argument.

    Args:
        log_transform: Batch of vectors of shape `(minibatch, 6)`.
        eps: A threshold for clipping the squared norm of the rotation logarithm
            to avoid unstable gradients in the singular case.

    Returns:
        Batch of transformation matrices of shape `(minibatch, 4, 4)`.

    Raises:
        ValueError if `log_transform` is of incorrect shape.

    [1] https://jinyongjeong.github.io/Download/SE3/jlblanco2010geometry3d_techrep.pdf
    [2] https://en.wikipedia.org/wiki/Hat_operator
    """

    if log_transform.ndim != 2 or log_transform.shape[1] != 6:
        raise ValueError("Expected input to be of shape (N, 6).")

    N, _ = log_transform.shape

    log_translation = log_transform[..., :3]
    log_rotation = log_transform[..., 3:]

    # rotation is an exponential map of log_rotation
    (
        R,
        rotation_angles,
        log_rotation_hat,
        log_rotation_hat_square,
    ) = _so3_exp_map(log_rotation, eps=eps)

    # translation is V @ T
    V = _se3_V_matrix(
        log_rotation,
        log_rotation_hat,
        log_rotation_hat_square,
        rotation_angles,
        eps=eps,
    )
    T = torch.bmm(V, log_translation[:, :, None])[:, :, 0]

    transform = torch.zeros(
        N, 4, 4, dtype=log_transform.dtype, device=log_transform.device
    )

    transform[:, :3, :3] = R
    transform[:, :3, 3] = T
    transform[:, 3, 3] = 1.0

    return transform.permute(0, 2, 1)


def _p3d_se3_log_map(
    transform: torch.Tensor, eps: float = 1e-4, cos_bound: float = 1e-4
) -> torch.Tensor:
    """
    Convert a batch of 4x4 transformation matrices `transform`
    to a batch of 6-dimensional SE(3) logarithms of the SE(3) matrices.
    See e.g. [1], Sec 9.4.2. for more detailed description.

    A SE(3) matrix has the following form:
        ```
        [ R 0 ]
        [ T 1 ] ,
        ```
    where `R` is an orthonormal 3x3 rotation matrix and `T` is a 3-D translation vector.
    SE(3) matrices are commonly used to represent rigid motions or camera extrinsics.

    In the SE(3) logarithmic representation SE(3) matrices are
    represented as 6-dimensional vectors `[log_translation | log_rotation]`,
    i.e. a concatenation of two 3D vectors `log_translation` and `log_rotation`.

    The conversion from the 4x4 SE(3) matrix `transform` to the
    6D representation `log_transform = [log_translation | log_rotation]`
    is done as follows:
        ```
        log_transform = log(transform)
        log_translation = log_transform[3, :3]
        log_rotation = inv_hat(log_transform[:3, :3])
        ```
    where `log` is the matrix logarithm
    and `inv_hat` is the inverse of the Hat operator [2].

    Note that for any valid 4x4 `transform` matrix, the following identity holds:
    ```
    se3_exp_map(se3_log_map(transform)) == transform
    ```

    The conversion has a singularity around `(transform=I)` which is handled
    by clamping controlled with the `eps` and `cos_bound` arguments.

    Args:
        transform: batch of SE(3) matrices of shape `(minibatch, 4, 4)`.
        eps: A threshold for clipping the squared norm of the rotation logarithm
            to avoid division by zero in the singular case.
        cos_bound: Clamps the cosine of the rotation angle to
            [-1 + cos_bound, 3 - cos_bound] to avoid non-finite outputs.
            The non-finite outputs can be caused by passing small rotation angles
            to the `acos` function in `so3_rotation_angle` of `so3_log_map`.

    Returns:
        Batch of logarithms of input SE(3) matrices
        of shape `(minibatch, 6)`.

    Raises:
        ValueError if `transform` is of incorrect shape.
        ValueError if `R` has an unexpected trace.

    [1] https://jinyongjeong.github.io/Download/SE3/jlblanco2010geometry3d_techrep.pdf
    [2] https://en.wikipedia.org/wiki/Hat_operator
    """

    if transform.ndim != 3:
        raise ValueError("Input tensor shape has to be (N, 4, 4).")

    N, dim1, dim2 = transform.shape
    if dim1 != 4 or dim2 != 4:
        raise ValueError("Input tensor shape has to be (N, 4, 4).")

    if not torch.allclose(transform[:, :3, 3], torch.zeros_like(transform[:, :3, 3])):
        raise ValueError("All elements of `transform[:, :3, 3]` should be 0.")

    # log_rot is just so3_log_map of the upper left 3x3 block
    R = transform[:, :3, :3].permute(0, 2, 1)
    log_rotation = so3_log_map(R, eps=eps, cos_bound=cos_bound)

    # log_translation is V^-1 @ T
    T = transform[:, 3, :3]
    V = _se3_V_matrix(*_get_se3_V_input(log_rotation), eps=eps)
    log_translation = torch.linalg.solve(V, T[:, :, None])[:, :, 0]

    return torch.cat((log_translation, log_rotation), dim=1)


# ===========================================================================
# Functions originally from caliball.utils.utils_3d
# ===========================================================================

def pose_to_hom_mat(pose, rot_type="euler_xyz"):
    """将位姿 [x,y,z,r1,r2,r3] 转为 4x4 齐次矩阵。

    ``axis_angle`` / ``axis_angle_residual``：后三维为旋转向量（axis-angle，与 scipy ``Rotation.from_rotvec`` 一致），
    常用于 OXE 等数据集中「末端轴角残差」与平移残差并列的 6 维表示。
    """
    hom_mat = np.eye(4)
    hom_mat[:3, 3] = pose[:3]
    if rot_type in ("axis_angle", "axis_angle_residual", "axis-angle", "axis-angle-residual"):
        rot = Rotation.from_rotvec(pose[3:6]).as_matrix()
    elif rot_type == "euler_xyz":
        rot = Rotation.from_euler("xyz", pose[3:6]).as_matrix()
    elif rot_type == "euler_zyx":
        rot = Rotation.from_euler("zyx", pose[3:6]).as_matrix()
    elif rot_type == "quaternion":
        rot = Rotation.from_quat(pose[3:7]).as_matrix()
    elif rot_type == "rotation_matrix":
        pose_arr = np.array(pose)
        rot = pose_arr[:3, :3] if pose_arr.shape == (4, 4) else pose_arr[3:12].reshape(3, 3)
    else:
        raise ValueError(f"Invalid rotation type: {rot_type}")
    hom_mat[:3, :3] = rot
    return hom_mat


def hom_mat_to_pose(hom_mat, rot_type="euler_xyz"):
    """将 4x4 齐次矩阵转为位姿 [x,y,z,r1,r2,r3]（轴角时为旋转向量）"""
    hom_mat = np.array(hom_mat)
    if rot_type in ("axis_angle", "axis_angle_residual", "axis-angle", "axis-angle-residual"):
        pos = hom_mat[:3, 3]
        rot = Rotation.from_matrix(hom_mat[:3, :3]).as_rotvec()
        return np.concatenate((pos, rot))
    elif rot_type == "euler_xyz":
        pos = hom_mat[:3, 3]
        rot = Rotation.from_matrix(hom_mat[:3, :3]).as_euler("xyz")
        return np.concatenate((pos, rot))
    elif rot_type == "euler_zyx":
        pos = hom_mat[:3, 3]
        rot = Rotation.from_matrix(hom_mat[:3, :3]).as_euler("zyx")
        return np.concatenate((pos, rot))
    elif rot_type == "quaternion":
        pos = hom_mat[:3, 3]
        rot = Rotation.from_matrix(hom_mat[:3, :3]).as_quat()
        return np.concatenate((pos, rot))
    elif rot_type == "rotation_matrix":
        return hom_mat
    else:
        raise ValueError(f"Invalid rotation type: {rot_type}")


def diff_rot_mat(rot_end, rot_start):
    """rot_end @ inv(rot_start)，用于计算旋转差分"""
    rot_end = np.array(rot_end)
    rot_start = np.array(rot_start)
    if rot_end.shape != (3, 3) or rot_start.shape != (3, 3):
        raise ValueError(f"Invalid rotation matrix shape")
    return rot_end @ np.linalg.inv(rot_start)


@dispatch(np.ndarray, np.ndarray)
def transform_points(pts, pose):
    """
    :param pts: Nx3 P_b
    :param pose: 4x4 Ta_a2b
    :return: Nx3 P_a
    """
    pts = cart_to_hom(pts)
    pts = pts @ pose.T
    pts = hom_to_cart(pts)
    return pts


def cart_to_hom(pts):
    """
    :param pts: (N, 3 or 2)
    :return pts_hom: (N, 4 or 3)
    """
    if isinstance(pts, np.ndarray):
        pts_hom = np.concatenate((pts, np.ones([*pts.shape[:-1], 1], dtype=np.float32)), -1)
    else:
        ones = torch.ones([*pts.shape[:-1], 1], dtype=torch.float32, device=pts.device)
        pts_hom = torch.cat((pts, ones), dim=-1)
    return pts_hom


@dispatch(torch.Tensor, torch.Tensor)
def transform_points(pts, pose):
    pts = cart_to_hom(pts)
    pts = pts @ pose.transpose(-1, -2)
    pts = hom_to_cart(pts)
    return pts


def hom_to_cart(pts):
    return pts[..., :-1] / pts[..., -1:]


def rotx_np(a):
    """
    :param a: np.ndarray of (N, 1) or (N), or float, or int
              angle
    :return: np.ndarray of (N, 3, 3)
             rotation matrix
    """
    if isinstance(a, (int, float)):
        a = np.array([a])
    a = a.astype(float).reshape((-1, 1))
    ones = np.ones_like(a)
    zeros = np.zeros_like(a)
    c = np.cos(a)
    s = np.sin(a)
    rot = np.stack([ones, zeros, zeros,
                    zeros, c, -s,
                    zeros, s, c])
    return rot.reshape((-1, 3, 3))


def roty_np(a):
    """
    :param a: np.ndarray of (N, 1) or (N), or float, or int
              angle
    :return: np.ndarray of (N, 3, 3)
             rotation matrix
    """
    if isinstance(a, (int, float)):
        a = np.array([a])
    a = a.astype(float).reshape((-1, 1))
    ones = np.ones_like(a)
    zeros = np.zeros_like(a)
    c = np.cos(a)
    s = np.sin(a)
    rot = np.stack([c, zeros, s,
                    zeros, ones, zeros,
                    -s, zeros, c])
    return rot.reshape((-1, 3, 3))


def roty_torch(a):
    """
    :param a: np.ndarray of (N, 1) or (N), or float, or int
              angle
    :return: np.ndarray of (N, 3, 3)
             rotation matrix
    """
    if isinstance(a, (int, float)):
        a = torch.tensor([a])
    # if a.shape[-1] != 1:
    #     a = a[..., None]
    a = a.float()
    ones = torch.ones_like(a)
    zeros = torch.zeros_like(a)
    c = torch.cos(a)
    s = torch.sin(a)
    rot = torch.stack([c, zeros, s,
                       zeros, ones, zeros,
                       -s, zeros, c], dim=-1)
    return rot.reshape(*rot.shape[:-1], 3, 3)


def rotz_np(a):
    """
    :param a: np.ndarray of (N, 1) or (N), or float, or int
              angle
    :return: np.ndarray of (N, 3, 3)
             rotation matrix
    """
    if isinstance(a, (int, float)):
        a = np.array([a])
    a = a.astype(float).reshape((-1, 1))
    ones = np.ones_like(a)
    zeros = np.zeros_like(a)
    c = np.cos(a)
    s = np.sin(a)
    rot = np.stack([c, -s, zeros,
                    s, c, zeros,
                    zeros, zeros, ones])
    return rot.reshape((-1, 3, 3))


def rotz_torch(a):
    if isinstance(a, (int, float)):
        a = torch.tensor([a])
    if a.shape[-1] != 1:
        a = a[..., None]
    a = a.float()
    ones = torch.ones_like(a)
    zeros = torch.zeros_like(a)
    c = torch.cos(a)
    s = torch.sin(a)
    rot = torch.stack([c, -s, zeros,
                       s, c, zeros,
                       zeros, zeros, ones], dim=-1)
    return rot.reshape(*rot.shape[:-1], 3, 3)


def rotx(t):
    """
    Rotation along the x-axis.
    :param t: tensor of (N, 1) or (N), or float, or int
              angle
    :return: tensor of (N, 3, 3)
             rotation matrix
    """
    if isinstance(t, (int, float)):
        t = torch.tensor([t])
    if t.shape[-1] != 1:
        t = t[..., None]
    t = t.type(torch.float)
    ones = torch.ones_like(t)
    zeros = torch.zeros_like(t)
    c = torch.cos(t)
    s = torch.sin(t)
    rot = torch.stack([ones, zeros, zeros,
                       zeros, c, -s,
                       zeros, s, c], dim=-1)
    return rot.reshape(*rot.shape[:-1], 3, 3)


def matrix_3x4_to_4x4(a):
    if len(a.shape) == 2:
        assert a.shape == (3, 4)
    else:
        assert len(a.shape) == 3
        assert a.shape[1:] == (3, 4)
    if isinstance(a, np.ndarray):
        if a.ndim == 2:
            ones = np.array([[0, 0, 0, 1]])
            return np.vstack((a, ones))
        else:
            ones = np.array([[0, 0, 0, 1]])[None].repeat(a.shape[0], axis=0)
            return np.concatenate((a, ones), axis=1)
    else:
        ones = torch.tensor([[0, 0, 0, 1]]).float().to(device=a.device)
        if a.ndim == 3:
            ones = ones[None].repeat(a.shape[0], 1, 1)
            ret = torch.cat((a, ones), dim=1)
        else:
            ret = torch.cat((a, ones), dim=0)
        return ret


def img_to_rect(fu, fv, cu, cv, u, v, depth_rect):
    """
    :param u: (N)
    :param v: (N)
    :param depth_rect: (N)
    :return: pts_rect:(N, 3)
    """
    # check_type(u)
    # check_type(v)

    if isinstance(depth_rect, np.ndarray):
        x = ((u - cu) * depth_rect) / fu
        y = ((v - cv) * depth_rect) / fv
        pts_rect = np.concatenate((x.reshape(-1, 1), y.reshape(-1, 1), depth_rect.reshape(-1, 1)), axis=1)
    else:
        x = ((u.float() - cu) * depth_rect) / fu
        y = ((v.float() - cv) * depth_rect) / fv
        pts_rect = torch.cat((x.reshape(-1, 1), y.reshape(-1, 1), depth_rect.reshape(-1, 1)), dim=1)
    # x = ((u - cu) * depth_rect) / fu
    # y = ((v - cv) * depth_rect) / fv
    # pts_rect = np.concatenate((x.reshape(-1, 1), y.reshape(-1, 1), depth_rect.reshape(-1, 1)), axis=1)
    return pts_rect


def depth_to_rect(fu, fv, cu, cv, depth_map, ray_mode=False, select_coords=None):
    """

    :param fu:
    :param fv:
    :param cu:
    :param cv:
    :param depth_map:
    :param ray_mode: whether values in depth_map are Z or norm
    :return:
    """
    if len(depth_map.shape) == 2:
        if isinstance(depth_map, np.ndarray):
            x_range = np.arange(0, depth_map.shape[1])
            y_range = np.arange(0, depth_map.shape[0])
            x_idxs, y_idxs = np.meshgrid(x_range, y_range)
        else:
            x_range = torch.arange(0, depth_map.shape[1]).to(device=depth_map.device)
            y_range = torch.arange(0, depth_map.shape[0]).to(device=depth_map.device)
            y_idxs, x_idxs = torch.meshgrid(y_range, x_range, indexing='ij')
        x_idxs, y_idxs = x_idxs.reshape(-1), y_idxs.reshape(-1)
        depth = depth_map[y_idxs, x_idxs]
    else:
        x_idxs = select_coords[:, 1].float()
        y_idxs = select_coords[:, 0].float()
        depth = depth_map
    if ray_mode is True:
        if isinstance(depth, torch.Tensor):
            depth = depth / (((x_idxs.float() - cu.float()) / fu.float()) ** 2 + (
                    (y_idxs.float() - cv.float()) / fv.float()) ** 2 + 1) ** 0.5
        else:
            depth = depth / (((x_idxs - cu) / fu) ** 2 + (
                    (y_idxs - cv) / fv) ** 2 + 1) ** 0.5
    pts_rect = img_to_rect(fu, fv, cu, cv, x_idxs, y_idxs, depth)
    return pts_rect


def create_center_radius(center=np.array([0, 0, 0]), dist=5., angle_z=30, nrad=180, start=0., endpoint=True,
                         end=2 * np.pi):
    RTs = []
    center = np.array(center).reshape(3, 1)
    thetas = np.linspace(start, end, nrad, endpoint=endpoint)
    angle_z = np.deg2rad(angle_z)
    radius = dist * np.cos(angle_z)
    height = dist * np.sin(angle_z)
    for theta in thetas:
        st = np.sin(theta)
        ct = np.cos(theta)
        center_ = np.array([radius * ct, radius * st, height]).reshape(3, 1)
        center_[0] += center[0, 0]
        center_[1] += center[1, 0]
        R = np.array([
            [-st, ct, 0],
            [0, 0, -1],
            [-ct, -st, 0]
        ])
        Rotx = cv2.Rodrigues(angle_z * np.array([1., 0., 0.]))[0]
        R = Rotx @ R
        T = - R @ center_
        center_ = - R.T @ T
        RT = np.hstack([R, T])
        RTs.append(RT)
    return np.stack(RTs)


def open3d_plane_segment_api(pts, distance_threshold, ransac_n=3, num_iterations=1000):
    import open3d as o3d

    pts = to_array(pts)
    pcd0 = o3d.geometry.PointCloud()
    pcd0.points = o3d.utility.Vector3dVector(pts)
    plane_model, inliers = pcd0.segment_plane(distance_threshold,
                                              ransac_n=ransac_n,
                                              num_iterations=num_iterations)
    return plane_model, inliers


def point_plane_distance_api(pts, plane_model):
    a, b, c, d = plane_model.tolist()
    if isinstance(pts, torch.Tensor):
        dists = (a * pts[:, 0] + b * pts[:, 1] + c * pts[:, 2] + d).abs() / ((a * a + b * b + c * c) ** 0.5)
    else:
        dists = np.abs(a * pts[:, 0] + b * pts[:, 1] + c * pts[:, 2] + d) / ((a * a + b * b + c * c) ** 0.5)
    return dists


def se3_exp_map(log_transform: torch.Tensor, eps: float = 1e-4):
    return _p3d_se3_exp_map(log_transform, eps)


def se3_log_map(transform: torch.Tensor, eps: float = 1e-4, cos_bound: float = 1e-4, backend=None,
                test_acc=True):
    if backend is None:
        loguru.logger.warning("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        loguru.logger.warning("!!!!se3_log_map backend is None!!!!")
        loguru.logger.warning("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        backend = 'pytorch3d'
    if backend == 'pytorch3d':
        dof6 = pytorch3d.transforms.se3.se3_log_map(transform, eps, cos_bound)
    elif backend == 'opencv':
        # from pytorch3d.common.compat import solve
        log_rotation = []
        for tsfm in transform:
            cv2_rot = -cv2.Rodrigues(to_array(tsfm[:3, :3]))[0]
            log_rotation.append(torch.from_numpy(cv2_rot.reshape(-1)).to(transform.device).float())
        log_rotation = torch.stack(log_rotation, dim=0)
        T = transform[:, 3, :3]
        V = _se3_V_matrix(*_get_se3_V_input(log_rotation), eps=eps)
        log_translation = torch.linalg.solve(V, T[:, :, None])[:, :, 0]
        dof6 = torch.cat((log_translation, log_rotation), dim=1)
    else:
        raise NotImplementedError()
    if test_acc:
        err = (se3_exp_map(dof6) - transform).abs().max()
        if err > 0.1:
            raise RuntimeError()
    return dof6


def Rt_to_pose(R, t=np.zeros(3)):
    pose = np.eye(4)
    pose[:3, :3] = R
    pose[:3, 3] = t
    return pose


def calc_pts_diameter2(pts):
    """
    Calculates diameter of a set of points (i.e. the maximum distance between
    any two points in the set). Faster but requires more memory than
    calc_pts_diameter.
    :param pts: nx3 ndarray with 3D points.
    :return: Diameter.
    """
    from scipy.spatial import distance
    dists = distance.cdist(pts, pts, 'euclidean')
    diameter = np.max(dists)
    return diameter


def calc_pose_from_lookat(phis, thetas, size, radius=1.2):
    """
    :param phis: [B], north 0, south pi
    :param thetas: [B]
    :param size: int
    return Tw_w2c
    """
    import torch
    def normalize(vectors):
        return vectors / (torch.norm(vectors, dim=-1, keepdim=True) + 1e-10)

    device = torch.device('cpu')
    thetas = torch.FloatTensor(thetas).to(device)
    phis = torch.FloatTensor(phis).to(device)

    centers = torch.stack([
        radius * torch.sin(thetas) * torch.sin(phis),
        -radius * torch.cos(thetas) * torch.sin(phis),
        radius * torch.cos(phis),
    ], dim=-1)  # [B, 3]

    # lookat
    forward_vector = normalize(centers)
    up_vector = torch.FloatTensor([0, 0, 1]).to(device).unsqueeze(0).repeat(size, 1)
    right_vector = normalize(torch.cross(up_vector, forward_vector, dim=-1))
    if right_vector.pow(2).sum() < 0.01:
        right_vector = torch.FloatTensor([0, 1, 0]).to(device).unsqueeze(0).repeat(size, 1)
    up_vector = normalize(torch.cross(forward_vector, right_vector, dim=-1))

    poses = torch.eye(4, dtype=torch.float, device=device).unsqueeze(0).repeat(size, 1, 1)
    poses[:, :3, :3] = torch.stack((right_vector, up_vector, forward_vector), dim=-1)
    poses[:, :3, 3] = centers
    blender2opencv = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]], dtype=np.float32)
    for i in range(len(poses)):
        poses[i] = poses[i] @ blender2opencv
    return poses
