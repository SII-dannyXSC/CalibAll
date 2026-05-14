"""
ARX5 双臂机器人类，基于 arx5_description_isaac.urdf 提取的 front-left 和 front-right 两个机械臂。
来源: Agilex_Cobot_Magic_robotwin
"""

from roboticstoolbox.robot.Robot import Robot


class Arx5Robotwin(Robot):
    """ARX5 双臂机器人（仅包含 front-left 和 front-right 两个机械臂）"""

    def __init__(self):
        links, name, urdf_string, urdf_filepath = self.URDF_read(
            file_path="urdf/arx5_dual_arm.urdf",
            tld="./third_party/urdf/arx5",
        )
        super().__init__(
            links,
            name=name,
            manufacturer="arx5_robotwin",
            urdf_string=urdf_string,
            urdf_filepath=urdf_filepath,
        )


if __name__ == "__main__":
    robot = Arx5Robotwin()
    import pdb

    pdb.set_trace()
    input()
