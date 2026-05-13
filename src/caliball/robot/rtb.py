from abc import ABC, abstractmethod

import numpy as np

from src.caliball.robot.base import BaseTF


class RoboticsToolBoxTF(BaseTF, ABC):
    """基于 roboticstoolbox 的臂类基类。纯臂：fkine_gripper = fkine_eef。"""

    def __init__(self, name_list, eef_name):
        self.name_list = list(name_list)
        self.eef_name = eef_name

    @property
    @abstractmethod
    def robot(self):
        pass

    def fkine_eef(self, q: np.ndarray) -> np.ndarray:
        """臂末端位姿，shape (1, 4, 4)。"""
        q = np.asarray(q, dtype=np.float64)
        eef_pose = self.robot.fkine(q, end=self.eef_name).A
        return eef_pose[np.newaxis]  # (1, 4, 4)

    def _fkine_gripper_raw(self, q: np.ndarray) -> np.ndarray:
        """纯臂无夹爪，与 fkine_eef 相同，shape (1, 4, 4)。"""
        return self.fkine_eef(q)

    def fkine_all(self, q: np.ndarray) -> np.ndarray:
        """所有 link 变换矩阵，shape (1, n_links, 4, 4)。"""
        q = np.asarray(q, dtype=np.float64)
        all_tf = [self.robot.fkine(q, end=name).A for name in self.name_list]
        return np.array(all_tf)[np.newaxis]  # (1, n_links, 4, 4)
