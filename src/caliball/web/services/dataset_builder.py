"""YAML-driven dataset construction service.

Parses dataset YAML configs from ``config/web/``, determines whether
the frontend should show state configuration controls, and constructs
the dataset with appropriate StateProcessor.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml


# config/web/ YAML 目录
_CONFIG_WEB_DIR = Path(__file__).resolve().parent.parent.parent / "config" / "web"


class DatasetBuilder:
    """YAML 驱动的 dataset 构造服务。"""

    _CONFIGURABLE_PROCESSORS = {
        "caliball.dataset.state_processors.SliceProcessor",
        "caliball.dataset.state_processors.DualArmSliceProcessor",
        "caliball.dataset.state_processors.StateProcessor",
    }
    _CONFIGURABLE_DATASETS = {
        "caliball.dataset.lerobot_dataset.LeRobotDataset",
    }

    @staticmethod
    def list_configs() -> list[dict]:
        """扫描 config/web/*.yaml，返回配置列表。

        Returns:
            [{"name": "默认（单臂）", "filename": "default.yaml"}, ...]
            按文件名排序，default.yaml 排在最前面。
        """
        configs = []
        if not _CONFIG_WEB_DIR.is_dir():
            return configs
        for p in sorted(_CONFIG_WEB_DIR.glob("*.yaml")):
            try:
                with open(p) as f:
                    data = yaml.safe_load(f)
                configs.append({
                    "name": data.get("name", p.stem),
                    "filename": p.name,
                })
            except Exception:
                continue

        def _sort_key(c):
            fn = c["filename"]
            if fn == "default.yaml":
                return (0, fn)
            if fn == "default_dual.yaml":
                return (1, fn)
            return (2, fn)

        configs.sort(key=_sort_key)
        return configs

    @staticmethod
    def parse_config(yaml_filename: str) -> dict:
        """解析 YAML，返回前端需要的信息。

        自动判断 configurable：dataset._target_ 和 state_processor._target_
        都在已知列表中，且存在未指定的参数 → configurable = True。

        Returns:
            {
                "name": str,
                "configurable": bool,
                "processor_type": str,
                "state_keys": list | None,
                "processor_params": dict,
            }
        """
        yaml_path = _CONFIG_WEB_DIR / yaml_filename
        with open(yaml_path) as f:
            data = yaml.safe_load(f)

        ds_cfg = data.get("dataset", {})
        ds_target = ds_cfg.get("_target_", "")
        sp_cfg = ds_cfg.get("state_processor", {})
        sp_target = sp_cfg.get("_target_", "")

        processor_type = sp_target.rsplit(".", 1)[-1] if sp_target else ""

        state_keys = ds_cfg.get("state_keys")
        if isinstance(state_keys, str):
            state_keys = [state_keys]

        processor_params = {k: v for k, v in sp_cfg.items() if k != "_target_"}

        is_known_ds = ds_target in DatasetBuilder._CONFIGURABLE_DATASETS
        is_known_sp = sp_target in DatasetBuilder._CONFIGURABLE_PROCESSORS
        has_missing = state_keys is None or not processor_params
        configurable = is_known_ds and is_known_sp and has_missing

        return {
            "name": data.get("name", yaml_filename),
            "configurable": configurable,
            "processor_type": processor_type,
            "state_keys": state_keys,
            "processor_params": processor_params,
        }

    @staticmethod
    def build(
        yaml_filename: str,
        task_path: str,
        episode_idx: int,
        overrides: Optional[dict] = None,
    ) -> tuple:
        """构造 dataset。

        Args:
            yaml_filename: web YAML 配置文件名
            task_path: 数据集本地路径
            episode_idx: episode 索引
            overrides: 前端传来的覆盖参数
                单臂: {"state_key": "obs.state", "start": 0, "stop": 6}
                双臂: {"state_key": "obs.left", "start": 0, "stop": 6,
                       "state_key_2": "obs.right", "start_2": 0, "stop_2": 6}

        Returns:
            (dataset, info) where info = {
                "yaml_filename": str,
                "state_keys": list[str],
                "processor_type": str,
                "overrides": dict | None,
            }
        """
        from hydra.utils import instantiate
        from omegaconf import OmegaConf

        from caliball.dataset.lerobot_dataset import LeRobotDataset
        from caliball.dataset.state_processors import StateProcessor

        yaml_path = _CONFIG_WEB_DIR / yaml_filename
        with open(yaml_path) as f:
            data = yaml.safe_load(f)

        ds_cfg = data.get("dataset", {})
        sp_cfg = dict(ds_cfg.get("state_processor", {}))
        ds_target = ds_cfg.get("_target_", "")

        state_keys = ds_cfg.get("state_keys")
        if isinstance(state_keys, str):
            state_keys = [state_keys]

        if overrides:
            if overrides.get("state_key"):
                state_keys = [overrides["state_key"]]
                if overrides.get("state_key_2"):
                    state_keys.append(overrides["state_key_2"])
            for key in ("start", "stop", "step", "offset",
                        "start_2", "stop_2", "step_2", "offset_2"):
                if key in overrides and overrides[key] is not None:
                    sp_cfg[key] = overrides[key]

        if not state_keys:
            state_keys = ["observation.state"]

        # Instantiate state_processor via Hydra
        sp_target = sp_cfg.get("_target_", "")
        if sp_target:
            processor = instantiate(OmegaConf.create(sp_cfg))
        else:
            processor = StateProcessor()

        processor_type = sp_target.rsplit(".", 1)[-1] if sp_target else "StateProcessor"

        # Construct dataset
        if ds_target in DatasetBuilder._CONFIGURABLE_DATASETS:
            dataset = LeRobotDataset(
                repo_id=task_path,
                episodes=[episode_idx],
                state_keys=state_keys,
                state_processor=processor,
            )
        else:
            # 自定义 dataset：透传 YAML 中 dataset 下的所有参数，用 Hydra instantiate
            custom_cfg = {k: v for k, v in ds_cfg.items()}
            custom_cfg.setdefault("repo_id", task_path)
            # custom_cfg.setdefault("episodes", [episode_idx])
            dataset = instantiate(OmegaConf.create(custom_cfg))

        info = {
            "yaml_filename": yaml_filename,
            "state_keys": state_keys,
            "processor_type": processor_type,
            "overrides": overrides,
        }
        return dataset, info
