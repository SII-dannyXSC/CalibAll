import os
from omegaconf import OmegaConf
from pathlib import Path

CUR_DIR = Path(__file__).resolve().parent

def build_robot_config(config):
    robot_type = config.robot_type
    if robot_type == "franka":
        return OmegaConf.load(os.path.join(CUR_DIR, "robot/franka.yaml"))
    elif robot_type == "ur5e":
        return OmegaConf.load(os.path.join(CUR_DIR, "robot/ur5e.yaml"))
        