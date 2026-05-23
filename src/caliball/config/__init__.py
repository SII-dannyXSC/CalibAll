from pathlib import Path
from typing import Union

from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

CUR_DIR = Path(__file__).resolve().parent


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
    """Construct robot TF from task config.

    If *cfg* carries a ``robot_type`` attribute, the new registry-based
    ``caliball.robots.build_robot`` is used.  Otherwise falls back to
    ``hydra.utils.instantiate(cfg.tf)`` for backward compatibility.
    """
    if hasattr(cfg, 'robot_type'):
        from caliball.robots import build_robot
        tf_kwargs = {}
        if "tf" in cfg and cfg.tf is not None:
            tf_kwargs = {k: v for k, v in cfg.tf.items() if k != "_target_"}
        return build_robot(cfg.robot_type, **tf_kwargs)
    # Fallback to hydra instantiate for backward compat
    return instantiate(cfg.tf)


def instantiate_dataset(cfg: DictConfig):
    """对任务配置中的 ``dataset`` 节点做 instantiate（若存在）。"""
    if "dataset" not in cfg or cfg.dataset is None:
        return None
    return instantiate(cfg.dataset)
