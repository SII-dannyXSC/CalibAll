import numpy as np
import roboticstoolbox as rtb

from src.caliball.robot.base import BaseTF


class FrankaTF(BaseTF):
    """Franka Panda 臂（纯臂）。q: (7,)，fkine_gripper = fkine_eef。"""

    JOINT_NUM = 7

    def __init__(self):
        self._robot = rtb.models.Panda()

    def fkine_eef(self, q: np.ndarray) -> np.ndarray:
        """panda_link8 位姿，shape (1, 4, 4)。"""
        q = np.asarray(q, dtype=np.float64)[: self.JOINT_NUM]
        all_tf = self._robot.fkine_all(q)
        eef_pose = all_tf[9].A
        return eef_pose[np.newaxis]  # (1, 4, 4)

    def _fkine_gripper_raw(self, q: np.ndarray) -> np.ndarray:
        """纯臂，与 fkine_eef 相同，shape (1, 4, 4)。"""
        return self.fkine_eef(q)

    def fkine_all(self, q: np.ndarray) -> np.ndarray:
        """臂所有 link（link1-link8 + panda_hand），shape (1, 9, 4, 4)。"""
        q = np.asarray(q, dtype=np.float64)[: self.JOINT_NUM]
        all_tf = self._robot.fkine_all(q)
        link8 = all_tf[9]
        arm_links = all_tf[1:9]
        tf_hand = link8 @ self._robot.grippers[0].links[0].A()
        tfs = np.array([item.A for item in arm_links] + [tf_hand.A])
        return tfs[np.newaxis]  # (1, 9, 4, 4)
