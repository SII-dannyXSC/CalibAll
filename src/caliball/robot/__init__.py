from .franka import FrankaTF
from .ur5e import Ur5eTF
# from caliball.robot.

def build_robot(config, robot_config = None):
    robot_type = config.robot_type
    if robot_type=="franka":
        return FrankaTF()
    elif robot_type=="ur5e":
        return Ur5eTF(robot_config.names, robot_config.eef_name)