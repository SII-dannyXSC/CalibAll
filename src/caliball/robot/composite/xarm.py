import numpy as np
from scipy.spatial.transform import Rotation as R

from src.caliball.robot.arm_gripper_composite import ArmGripperCompositeTF
from src.caliball.robot.base import BaseTF
from src.caliball.robot.rtb import RoboticsToolBoxTF
from src.caliball.robot.urdf.xarm7 import XArm7


class XArm7ArmTF(RoboticsToolBoxTF):
    """xArm7 臂链（link_base…link7），仅前 7 维关节角；其余关节在 FK 中置 0。"""

    def __init__(self, name_list, eef_name: str):
        super().__init__(list(name_list), eef_name)
        self._robot = XArm7()

    @property
    def robot(self):
        return self._robot

    def _pad(self, q_arm: np.ndarray) -> np.ndarray:
        q_arm = np.asarray(q_arm, dtype=np.float64)
        q = np.zeros(self._robot.n, dtype=np.float64)
        q[:7] = q_arm[:7]
        return q

    def fkine_eef(self, q: np.ndarray) -> np.ndarray:
        """臂末端 link7 位姿，shape (1, 4, 4)。"""
        return self.robot.fkine(self._pad(q), end=self.eef_name).A[np.newaxis]

    def _fkine_gripper_raw(self, q: np.ndarray) -> np.ndarray:
        return self.fkine_eef(q)

    def fkine_all(self, q: np.ndarray) -> np.ndarray:
        """所有 link 变换矩阵，shape (1, n_links, 4, 4)。"""
        q_pad = self._pad(q)
        tfs = [self.robot.fkine(q_pad, end=name).A for name in self.name_list]
        return np.array(tfs)[np.newaxis]  # (1, n_links, 4, 4)


class XArm7MenagerieGripperTF(BaseTF):
    """xArm7 menagerie 一体夹爪子链；位姿在 **link7（法兰）坐标系** 下。

    q: (1,) 标量绑定左右 finger 转动关节（0 开，0.85 闭，弧度）。
    fkine_eef / fkine_gripper: eef link 在法兰系下位姿。
    """

    FLANGE_LINK = "link7"

    @staticmethod
    def _jindex_for_child_link(robot, child_link_name: str) -> int:
        for lk in robot.links:
            if lk is None or lk.name != child_link_name:
                continue
            return int(lk.jindex)
        raise ValueError(f"xarm7 URDF: no link {child_link_name!r} with jindex")

    def __init__(self, name_list, eef_name: str):
        self.name_list = list(name_list)
        self.eef_name = eef_name
        self._robot = XArm7()
        self._idx_left_finger = self._jindex_for_child_link(self._robot, "left_finger")
        self._idx_right_finger = self._jindex_for_child_link(self._robot, "right_finger")

        self.mesh_adjust = {name: np.eye(4) for name in self.name_list}
        for link in self._robot.links:
            if link is None or link.name not in self.mesh_adjust or len(link.geometry) == 0:
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

        q0 = np.zeros(self._robot.n)
        T_fl = self._robot.fkine(q0, end=self.FLANGE_LINK).A
        self._T_fl_inv_zero = np.linalg.inv(T_fl)

    def _q_full(self, g: float) -> np.ndarray:
        q = np.zeros(self._robot.n, dtype=np.float64)
        q[self._idx_left_finger] = g
        q[self._idx_right_finger] = g
        return q

    def fkine_eef(self, q: np.ndarray) -> np.ndarray:
        """eef link 在法兰系下位姿，shape (1, 4, 4)。"""
        q = np.asarray(q, dtype=np.float64)
        qf = self._q_full(float(q[0]))
        T_w = self._robot.fkine(qf, end=self.eef_name).A
        return (self._T_fl_inv_zero @ T_w)[np.newaxis]  # (1, 4, 4)

    def _fkine_gripper_raw(self, q: np.ndarray) -> np.ndarray:
        return self.fkine_eef(q)

    def fkine_all(self, q: np.ndarray) -> np.ndarray:
        """各 gripper link 在法兰系下（含 mesh 对齐），shape (1, n_links, 4, 4)。"""
        q = np.asarray(q, dtype=np.float64)
        qf = self._q_full(float(q[0]))
        row = []
        for name in self.name_list:
            T_w = self._robot.fkine(qf, end=name).A
            T_rel = self._T_fl_inv_zero @ T_w
            row.append(T_rel @ self.mesh_adjust[name])
        return np.array(row)[np.newaxis]  # (1, n_links, 4, 4)


class XArm7WithGripperTF(ArmGripperCompositeTF):
    """xArm7（menagerie URDF）+ 原厂夹爪。q: (8,) = 7 臂关节 + 1 夹爪行程。

    fkine_eef: link7 法兰位姿。
    fkine_gripper: 左右指尖位置中点（当前开合值），姿态取 xarm_gripper_base_link 在法兰系下的旋转。
    fkine_all: 臂 8 link + 夹爪 link（与 xarm7_with_gripper.yaml mesh_paths 对齐）。
    """

    GRIPPER_CLOSED = 0.85
    ARM_NAME_PREFIX_LEN = 8
    EE_TIP_LEFT_HOM = np.array([0.01323607, -0.0240032, 0.06080743, 1.0], dtype=np.float64)
    EE_TIP_RIGHT_HOM = np.array([-0.01323607, 0.0240032, 0.06080743, 1.0], dtype=np.float64)
    EE_ORIENTATION_FLANGE_LINK = "xarm_gripper_base_link"

    def __init__(
        self,
        full_chain_names,
        gripper_eef_name: str,
        *,
        gripper_mount_yaw_deg: float = 0.0,
        grasp_point_rotation_align=None,
    ):
        super().__init__(
            arm_joint_num=7,
            gripper_closed_q=float(self.GRIPPER_CLOSED),
            gripper_mount_yaw_deg=gripper_mount_yaw_deg,
        )
        names = list(full_chain_names)
        self.arm = XArm7ArmTF(names[: self.ARM_NAME_PREFIX_LEN], "link7")
        self.gripper = XArm7MenagerieGripperTF(
            names[self.ARM_NAME_PREFIX_LEN :],
            gripper_eef_name,
        )
        self.grasp_point_R_align = self._build_grasp_point_R_align(grasp_point_rotation_align)

    def _mount_T_for_tcp(self, q_arm: np.ndarray) -> np.ndarray:
        return self.arm.fkine_eef(q_arm)  # (1, 4, 4)

    def _mount_T_for_gripper_meshes(
        self, arm_tfs: np.ndarray, q_arm: np.ndarray
    ) -> np.ndarray:
        return arm_tfs[:, -1]  # (1, 4, 4)

    def _fkine_gripper_raw(self, q: np.ndarray) -> np.ndarray:
        """左右指尖中点位置 + 夹爪座姿态，shape (1, 4, 4)。"""
        q = np.atleast_1d(np.asarray(q, dtype=np.float64))
        q_arm = q[: self.arm_joint_num]
        q_grip = q[self.arm_joint_num]

        arm_tfs = self.arm.fkine_all(q_arm)  # (1, n_arm_links, 4, 4)
        T_mount = self._mount_with_gripper_bias(
            self._mount_T_for_gripper_meshes(arm_tfs, q_arm)
        )  # (1, 4, 4)

        g = self.gripper
        r = g._robot
        inv_fl = g._T_fl_inv_zero
        qf = g._q_full(float(q_grip))

        T_Lw = r.fkine(qf, end="left_finger").A
        T_Rw = r.fkine(qf, end="right_finger").A
        T_Bw = r.fkine(qf, end=self.EE_ORIENTATION_FLANGE_LINK).A

        T_rel_L = inv_fl @ T_Lw
        T_rel_R = inv_fl @ T_Rw
        T_rel_B = inv_fl @ T_Bw

        p_l = (T_rel_L @ self.EE_TIP_LEFT_HOM)[:3]
        p_r = (T_rel_R @ self.EE_TIP_RIGHT_HOM)[:3]
        p_mid = 0.5 * (p_l + p_r)

        Tee = np.eye(4, dtype=np.float64)
        Tee[:3, :3] = T_rel_B[:3, :3]
        Tee[:3, 3] = p_mid

        return (T_mount[0] @ Tee)[np.newaxis]  # (1, 4, 4)
