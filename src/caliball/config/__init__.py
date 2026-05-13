import os
from pathlib import Path
from typing import Union

from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

CUR_DIR = Path(__file__).resolve().parent


def _load_composite_config(arm_yaml: str, gripper_yaml: str) -> DictConfig:
    """加载 arm + gripper 子 yaml 并合并为 {arm: ..., gripper: ...}。
    用于 franka_robotiq / ur5e_robotiq / franka_panda_hand 等使用 Hydra defaults 的配置。
    """
    robot_dir = os.path.join(CUR_DIR, "robot")
    arm_cfg     = OmegaConf.load(os.path.join(robot_dir, arm_yaml))
    gripper_cfg = OmegaConf.load(os.path.join(robot_dir, gripper_yaml))
    return OmegaConf.create({"arm": arm_cfg, "gripper": gripper_cfg})


def build_robot_config(config):
    robot_type = config.robot_type
    if robot_type == "franka":
        return OmegaConf.load(os.path.join(CUR_DIR, "robot/franka.yaml"))
    elif robot_type == "ur5e":
        return OmegaConf.load(os.path.join(CUR_DIR, "robot/ur5e.yaml"))
    elif robot_type in ("aloha", "aloha_cobot_magic"):
        return OmegaConf.load(os.path.join(CUR_DIR, "robot/aloha.yaml"))
    elif robot_type == "arx5_robotwin":
        return OmegaConf.load(os.path.join(CUR_DIR, "robot/arx5_robotwin.yaml"))
    elif robot_type == "franka_robotiq":
        return _load_composite_config("arm/franka_panda.yaml", "gripper/robotiq.yaml")
    elif robot_type == "ur5e_robotiq":
        return _load_composite_config("arm/ur5e.yaml", "gripper/robotiq.yaml")
    elif robot_type == "franka_panda_hand":
        return _load_composite_config("arm/fr3.yaml", "gripper/panda_hand.yaml")
    elif robot_type == "xarm7_with_gripper":
        return OmegaConf.load(os.path.join(CUR_DIR, "robot/xarm7_with_gripper.yaml"))


def compose_job_config(
    config_name: str,
    *,
    config_dir: Union[str, Path, None] = None,
) -> DictConfig:
    """
    在 ``config_dir`` 下用 Hydra ``compose`` 加载任务 YAML（含 ``defaults`` 合并与插值）。

    ``config_name`` 不含 ``.yaml`` 后缀，例如 ``berkeley_autolab_ur5``。
    默认 ``config_dir`` 为本包目录 ``caliball/config``（其下须有 ``robot/*.yaml`` 等）。
    """
    d = Path(config_dir) if config_dir is not None else CUR_DIR
    with initialize_config_dir(version_base=None, config_dir=str(d.resolve())):
        return compose(config_name=config_name)


def compose_job_config_from_path(
    config_path: Union[str, Path],
    *,
    project_root: Union[str, Path, None] = None,
) -> DictConfig:
    """从 YAML 路径推断 ``config_dir`` 与文件名（stem），再 ``compose_job_config``。"""
    p = Path(config_path)
    if not p.is_absolute():
        base = Path(project_root).resolve() if project_root is not None else Path.cwd()
        p = (base / p).resolve()
    else:
        p = p.resolve()
    return compose_job_config(config_name=p.stem, config_dir=p.parent)


def instantiate_tf(cfg: DictConfig):
    """对任务配置中的 ``tf`` 节点做 ``hydra.utils.instantiate``。"""
    return instantiate(cfg.tf)


def instantiate_dataset(cfg: DictConfig):
    """对任务配置中的 ``dataset`` 节点做 instantiate（若存在）。"""
    if "dataset" not in cfg or cfg.dataset is None:
        return None
    return instantiate(cfg.dataset)
