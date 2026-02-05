from omegaconf import OmegaConf

def build_robot_config():
    return OmegaConf.load("caliball/config/robot/franka.yaml")