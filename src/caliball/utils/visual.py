import numpy as np
import trimesh


def sample_robot_points(tf_list, mesh_paths, num_points_per_link=10000):
    """
    Args:
        tf_list: list or np.ndarray, shape (num_links, 4, 4)
        mesh_paths: list[str], mesh path for each link
        num_points_per_link: int, number of sampled points per link

    Returns:
        all_points: np.ndarray, shape (N, 3)
    """

    all_points = []

    for link_idx, mesh_path in enumerate(mesh_paths):
        # load mesh
        mesh = trimesh.load(mesh_path, force='mesh')

        # sample points in local frame
        points = mesh.sample(num_points_per_link)

        # convert to homogeneous
        ones = np.ones((points.shape[0], 1))
        points_hom = np.hstack([points, ones])

        # transform to global frame
        point_global = (tf_list[link_idx] @ points_hom.T).T[:, :3]

        all_points.append(point_global)

    all_points = np.concatenate(all_points, axis=0)

    return all_points
