from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
import trimesh

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent


def _get_mesh_paths(robot_config) -> list:
    """从 robot_config 获取 mesh_paths 列表。

    优先使用顶层 mesh_paths；不存在时自动拼接 arm.mesh_paths + gripper.mesh_paths。
    """
    top = getattr(robot_config, "mesh_paths", None)
    if top is not None:
        return list(top)

    arm = getattr(robot_config, "arm", None)
    gripper = getattr(robot_config, "gripper", None)
    arm_paths = list(arm.mesh_paths) if arm is not None and getattr(arm, "mesh_paths", None) is not None else []
    grip_paths = list(gripper.mesh_paths) if gripper is not None and getattr(gripper, "mesh_paths", None) is not None else []

    if not arm_paths and not grip_paths:
        raise ValueError(
            "robot_config 既无顶层 mesh_paths，也无 arm.mesh_paths / gripper.mesh_paths"
        )
    return arm_paths + grip_paths


def load_meshes(robot_config, device: str, project_root: Path | None = None) -> tuple:
    """加载 robot mesh，返回 (vertices_list, faces_list)，均为 CUDA Tensor。

    mesh_paths 来源（按优先级）：
    1. robot_config.mesh_paths（顶层列表）
    2. robot_config.arm.mesh_paths + robot_config.gripper.mesh_paths（自动拼接）
    """
    root = project_root if project_root is not None else _REPO_ROOT
    vertices_list, faces_list = [], []
    for mesh_path in _get_mesh_paths(robot_config):
        full_path = os.path.join(str(root), mesh_path) if not os.path.isabs(mesh_path) else mesh_path
        mesh = trimesh.load(full_path, force="mesh")
        vertices_list.append(torch.from_numpy(np.array(mesh.vertices)).float().to(device=device))
        faces_list.append(torch.from_numpy(np.array(mesh.faces)).int().to(device=device))
    return vertices_list, faces_list
