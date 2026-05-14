from roboticstoolbox.robot.Robot import Robot


class Fr3Arm(Robot):
    """FR3 机械臂（无手爪），fr3.urdf，7 个旋转关节。"""

    def __init__(self):
        links, name, urdf_string, urdf_filepath = self.URDF_read(
            file_path="robots/fr3/fr3.urdf",
            tld="./third_party/urdf/franka_description",
        )
        super().__init__(
            links,
            name=name,
            manufacturer="Franka",
            urdf_string=urdf_string,
            urdf_filepath=urdf_filepath,
        )
