import numpy as np
from scipy.spatial.transform import Rotation as R

from src.caliball.robot.rtb import RoboticsToolBoxTF
from src.caliball.robot.urdf.ur5e import Ur5e


class Ur5eTF(RoboticsToolBoxTF):
    """UR5e 臂（纯臂）。q: (6,)，fkine_gripper = fkine_eef。"""

    def __init__(self, name_list, eef_name):
        super().__init__(name_list, eef_name)
        self._robot = Ur5e()

        self.mesh_adjust = {name: np.eye(4) for name in self.name_list}
        for link in self.robot.links:
            if link.name not in self.mesh_adjust or len(link.geometry) == 0:
                continue
            geo = link.geometry[0]
            t = geo._wT[:3, 3]
            q = geo._wq
            scale = geo.scale

            trans_mat = np.eye(4)
            trans_mat[:3, 3] = t
            trans_mat[:3, :3] = R.from_quat(q).as_matrix()

            scale_mat = np.eye(4)
            scale_mat[:3, :3] = np.diag(scale)
            self.mesh_adjust[link.name] = trans_mat @ scale_mat

    @property
    def robot(self):
        return self._robot

    def fkine_all(self, q: np.ndarray) -> np.ndarray:
        """所有 link 变换矩阵（含 mesh 对齐），shape (1, n_links, 4, 4)。"""
        result = super().fkine_all(q)  # (1, n_links, 4, 4)
        for idx, name in enumerate(self.name_list):
            result[0, idx] = result[0, idx] @ self.mesh_adjust[name]
        return result
