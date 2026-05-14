"""自动扫描 calibration/ 目录下的 YAML 标定文件，提供统一查询接口。

新增数据集只需在 calibration/ 下添加 YAML 文件，无需修改本文件。

YAML 格式：
    dataset: robomind/franka_3rgb
    aliases: [rdt]            # 可选，dataset 的别名
    tasks:                    # 可选，省略 = 该 dataset 的默认标定
      - task_name_1
      - task_name_2
    cameras:
      observation.images.camera_top:
        intrinsic: [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
        extrinsic: [[r00, ...], ...]
      default:                # 'default' 作为任意 camera_name 的 fallback
        intrinsic: [...]
        extrinsic: [...]
"""

from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

CALIB_DIR = Path(__file__).resolve().parent / "calibration"

_REGISTRY = None


def _load_all():
    registry = []
    for f in sorted(CALIB_DIR.glob("*.yaml")):
        cfg = OmegaConf.load(f)
        registry.append(cfg)
    return registry


def _get_registry():
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = _load_all()
    return _REGISTRY


def _match_dataset(cfg, dataset_name):
    if cfg.dataset == dataset_name:
        return True
    if "aliases" in cfg and cfg.aliases is not None:
        return dataset_name in cfg.aliases
    return False


def _resolve_camera(cfg, camera_name):
    cameras = cfg.cameras
    if camera_name in cameras:
        return cameras[camera_name]
    if "default" in cameras:
        return cameras["default"]
    return None


def get_intrinsic_extrinsic(dataset_name, task_name, camera_name, episode=None):
    """查找标定矩阵。

    匹配优先级:
      1. dataset + task 在 tasks 列表中 + camera
      2. dataset 匹配且无 tasks 限制（fallback）+ camera
    """
    registry = _get_registry()

    # Pass 1: 精确匹配 dataset + task
    for cfg in registry:
        if not _match_dataset(cfg, dataset_name):
            continue
        if "tasks" in cfg and cfg.tasks is not None:
            if task_name in cfg.tasks:
                cam = _resolve_camera(cfg, camera_name)
                if cam is not None:
                    return (
                        np.array(OmegaConf.to_container(cam.intrinsic), dtype=np.float64),
                        np.array(OmegaConf.to_container(cam.extrinsic), dtype=np.float64),
                    )

    # Pass 2: fallback — dataset 匹配 + 无 tasks 限制
    for cfg in registry:
        if not _match_dataset(cfg, dataset_name):
            continue
        if "tasks" not in cfg or cfg.tasks is None:
            cam = _resolve_camera(cfg, camera_name)
            if cam is not None:
                return (
                    np.array(OmegaConf.to_container(cam.intrinsic), dtype=np.float64),
                    np.array(OmegaConf.to_container(cam.extrinsic), dtype=np.float64),
                )

    raise ValueError(
        f"No calibration found for dataset={dataset_name!r}, "
        f"task={task_name!r}, camera={camera_name!r}"
    )


if __name__ == "__main__":
    I, E = get_intrinsic_extrinsic(
        "robomind/franka_3rgb", "put_the_red_apple_in_the_bowl",
        "observation.images.camera_top", 0,
    )
    print("Franka scene1 top:", I, E)

    I, E = get_intrinsic_extrinsic("berkeley_autolab_ur5", None, "default", 0)
    print("Berkeley UR5:", I, E)

    I, E = get_intrinsic_extrinsic("rdt", None, "default", 0)
    print("RDT (alias):", I, E)
