import numpy as np

from roboticstoolbox.robot.Robot import Robot

class Ur5e(Robot):
    def __init__(self):
        
        links, name, urdf_string, urdf_filepath = self.URDF_read(file_path="ur_description/urdf/ur5e.urdf",
                       tld="./assets/urdf_files_dataset/urdf_files/ros-industrial/xacro_generated/universal_robots")
        super().__init__(
            links,
            name=name,
            manufacturer="ur5e",
            urdf_string=urdf_string,
            urdf_filepath=urdf_filepath,
        )

if __name__ == "__main__":
    robot = Ur5e()
    import pdb;pdb.set_trace()
    input()