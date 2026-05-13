import numpy as np
from scipy.spatial.transform import Rotation as R

from src.caliball.robot.base import BaseTF
from src.caliball.robot.urdf.fr3_hand import Fr3Hand


class Fr3PandaGripperTF(BaseTF):
    """FR3 原厂手爪子链（fr3_hand.urdf）。位姿在 **fr3_link8（法兰）坐标系** 下。

    q: (1,) 单标量开合值 [0, FINGER_MAX]，内部复制到两指 prismatic。
    fkine_eef: eef link 在法兰系下的位姿。
    fkine_gripper: 与 fkine_eef 相同（指尖即 eef）。
    """

    ARM_DOF = 7
    FLANGE_LINK = "fr3_link8"
    FINGER_CLOSED = 0.04

    def __init__(self, name_list, eef_name):
        self.name_list = list(name_list)
        self.eef_name = eef_name
        self._robot = Fr3Hand()

        self.mesh_adjust = {name: np.eye(4) for name in self.name_list}
        for link in self._robot.links:
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

        q0 = np.zeros(9)
        T_fl = self._robot.fkine(q0, end=self.FLANGE_LINK).A
        self._T_fl_inv_zero = np.linalg.inv(T_fl)

    def _q_full(self, g: float) -> np.ndarray:
        return np.concatenate([np.zeros(self.ARM_DOF), [g, g]])

    def fkine_eef(self, q: np.ndarray) -> np.ndarray:
        """eef link 在法兰系下位姿，shape (1, 4, 4)。"""
        q = np.asarray(q, dtype=np.float64)
        g = float(q[0])
        qf = self._q_full(g)
        T_w = self._robot.fkine(qf, end=self.eef_name).A
        return (self._T_fl_inv_zero @ T_w)[np.newaxis]  # (1, 4, 4)

    def _fkine_gripper_raw(self, q: np.ndarray) -> np.ndarray:
        """与 fkine_eef 相同，shape (1, 4, 4)。"""
        return self.fkine_eef(q)

    def fkine_all(self, q: np.ndarray) -> np.ndarray:
        """各 gripper link 在法兰系下，shape (1, n_links, 4, 4)。"""
        q = np.asarray(q, dtype=np.float64)
        g = float(q[0])
        qf = self._q_full(g)
        row = []
        for name in self.name_list:
            T_w = self._robot.fkine(qf, end=name).A
            T_rel = self._T_fl_inv_zero @ T_w
            row.append(T_rel @ self.mesh_adjust[name])
        return np.array(row)[np.newaxis]  # (1, n_links, 4, 4)
