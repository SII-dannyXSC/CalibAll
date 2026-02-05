import os
from omegaconf import OmegaConf
from pathlib import Path

CUR_DIR = Path(__file__).resolve().parent

def build_robot_config():
    return OmegaConf.load(os.path.join("robot/franka.yaml"))