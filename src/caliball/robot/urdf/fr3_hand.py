from roboticstoolbox.robot.Robot import Robot


class Fr3Hand(Robot):
    """fr3_hand.urdf（7 臂 + 2 指）。由 Fr3PandaGripperTF 取子链，在 fr3_link8 法兰系下做 FK。"""

    def __init__(self):
        links, name, urdf_string, urdf_filepath = self.URDF_read(
            file_path="robots/fr3/fr3_hand.urdf",
            tld="./third_party/urdf_files_dataset/urdf_files/oems/xacro_generated/franka_emika/franka_description",
        )
        super().__init__(
            links,
            name=name,
            manufacturer="Franka",
            urdf_string=urdf_string,
            urdf_filepath=urdf_filepath,
        )
