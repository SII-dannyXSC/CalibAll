import numpy as np
from scipy.spatial.transform import Rotation as R

from src.caliball.robot.rtb import RoboticsToolBoxTF
from src.caliball.robot.urdf.robotiq_85 import Robotiq85


class RobotiqTF(RoboticsToolBoxTF):
    """Robotiq 2F-85 夹爪子链。q: (1,) 标量开合值 [0, 0.8]。

    fkine_eef: eef link 原点位姿（法兰系下）。
    fkine_gripper: 含指尖偏移的 TCP 位姿（法兰系下）。
    """

    MIMIC_MULTIPLIERS = np.array([1.0, -1.0, 1.0, 1.0, -1.0, 1.0])
    TIP_OFFSET = np.array([0.0, -0.0139, 0.0445, 1.0])

    def __init__(self, name_list, eef_name):
        super().__init__(name_list, eef_name)
        self._robot = Robotiq85()

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

    def _expand_q(self, q: np.ndarray) -> np.ndarray:
        """q: (1,) 标量 -> (6,) 展开后的 URDF 关节角。"""
        return float(q[0]) * self.MIMIC_MULTIPLIERS

    def fkine_eef(self, q: np.ndarray) -> np.ndarray:
        """eef link 位姿（无指尖偏移），shape (1, 4, 4)。"""
        q = np.asarray(q, dtype=np.float64)
        q6 = self._expand_q(q)
        eef_pose = self.robot.fkine(q6, end=self.eef_name).A
        return eef_pose[np.newaxis]  # (1, 4, 4)

    def _fkine_gripper_raw(self, q: np.ndarray) -> np.ndarray:
        """含指尖偏移的 TCP 位姿，shape (1, 4, 4)。"""
        q = np.asarray(q, dtype=np.float64)
        T_eef = self.fkine_eef(q)  # (1, 4, 4)
        T_tip = T_eef.copy()
        T_tip[0, :3, 3] = (T_eef[0] @ self.TIP_OFFSET)[:3]
        return T_tip  # (1, 4, 4)

    def fkine_all(self, q: np.ndarray) -> np.ndarray:
        """所有 link 变换矩阵（含 mesh 对齐），shape (1, n_links, 4, 4)。"""
        q = np.asarray(q, dtype=np.float64)
        q6 = self._expand_q(q)
        all_tf = [self.robot.fkine(q6, end=name).A for name in self.name_list]
        result = np.array(all_tf)[np.newaxis]  # (1, n_links, 4, 4)
        for idx, name in enumerate(self.name_list):
            result[0, idx] = result[0, idx] @ self.mesh_adjust[name]
        return result
