from .arm.franka import FrankaTF
from .arm.ur5e import Ur5eTF
from .arm.fr3 import Fr3ArmTF
from .gripper.robotiq import RobotiqTF
from .gripper.fr3_hand import Fr3PandaGripperTF
from .composite.ur5e_robotiq import Ur5eRobotiqTF
from .composite.franka_robotiq import FrankaRobotiqTF
from .composite.franka_hand import FrankaPandaHandTF
from .composite.xarm import XArm7WithGripperTF
from .dual_arm.dual_arm_base import DualArmTF
from .dual_arm.aloha import AlohaArmTF, AlohaCobotMagicTF
from .dual_arm.arx5 import Arx5ArmTF, Arx5RobotwinTF
from .arm_gripper_composite import ArmGripperCompositeTF


def build_robot(config, robot_config=None):
    robot_type = config.robot_type
    if robot_type == "franka":
        return FrankaTF()
    elif robot_type == "franka_panda_hand":
        return FrankaPandaHandTF(
            list(robot_config.arm.names),
            robot_config.arm.eef_name,
            list(robot_config.gripper.names),
            robot_config.gripper.eef_name,
        )
    elif robot_type == "franka_robotiq":
        return FrankaRobotiqTF(
            list(robot_config.names),
            robot_config.gripper.eef_name,
        )
    elif robot_type == "xarm7_with_gripper":
        return XArm7WithGripperTF(
            list(robot_config.names),
            robot_config.gripper.eef_name,
        )
    elif robot_type == "ur5e":
        return Ur5eTF(robot_config.names, robot_config.eef_name)
    elif robot_type == "ur5e_robotiq":
        return Ur5eRobotiqTF(
            list(robot_config.arm.names),
            robot_config.arm.eef_name,
            list(robot_config.gripper.names),
            robot_config.gripper.eef_name,
        )
    elif robot_type == "aloha_cobot_magic":
        return AlohaCobotMagicTF(robot_config.names, robot_config.eef_name)
    elif robot_type == "arx5_robotwin":
        return Arx5RobotwinTF(robot_config.names, robot_config.eef_name)
    raise ValueError(f"Unknown robot_type: {robot_type}")
