from roboticstoolbox.robot.Robot import Robot


class Robotiq85(Robot):
    def __init__(self):
        links, name, urdf_string, urdf_filepath = self.URDF_read(
            file_path="robotiq_2f_85_gripper_visualization/urdf/robotiq_arg2f_85_model.urdf",
            tld="./third_party/urdf_files_dataset/urdf_files/ros-industrial/xacro_generated/robotiq",
        )
        super().__init__(
            links,
            name=name,
            manufacturer="Robotiq",
            urdf_string=urdf_string,
            urdf_filepath=urdf_filepath,
        )


if __name__ == "__main__":
    robot = Robotiq85()
    import pdb

    pdb.set_trace()
