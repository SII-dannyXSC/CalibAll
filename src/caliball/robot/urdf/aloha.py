"""
ALOHA 双臂机器人类，基于 aloha_new.urdf 提取的 front-left 和 front-right 两个机械臂。
"""

from roboticstoolbox.robot.Robot import Robot


class AlohaCobotMagic(Robot):
    """ALOHA 双臂机器人（仅包含 front-left 和 front-right 两个机械臂）"""

    def __init__(self):
        links, name, urdf_string, urdf_filepath = self.URDF_read(
            file_path="aloha_new_description/urdf/aloha_dual_arm.urdf",
            tld="./third_party/urdf/aloha",
        )
        super().__init__(
            links,
            name=name,
            manufacturer="aloha",
            urdf_string=urdf_string,
            urdf_filepath=urdf_filepath,
        )


if __name__ == "__main__":
    robot = AlohaCobotMagic()
    import pdb

    pdb.set_trace()
    input()
