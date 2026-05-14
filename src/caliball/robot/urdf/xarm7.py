from roboticstoolbox.robot.Robot import Robot


class XArm7(Robot):
    """xArm7 + 夹爪（third_party/urdf/xarm7/xarm7.urdf，源自 mujoco_menagerie ufactory_xarm7）。"""

    def __init__(self):
        links, name, urdf_string, urdf_filepath = self.URDF_read(
            file_path="xarm7.urdf",
            tld="./third_party/urdf/xarm7",
        )
        super().__init__(
            links,
            name=name,
            manufacturer="UFACTORY",
            urdf_string=urdf_string,
            urdf_filepath=urdf_filepath,
        )

if __name__ == "__main__":
    import numpy as np

    robot = XArm7()
    import pdb; pdb.set_trace()
    print("n", robot.n)
    q = np.zeros(robot.n)
    print("fkine link7", robot.fkine(q, end="link7").A.shape)
    print("fkine link_tcp", robot.fkine(q, end="link_tcp").A.shape)