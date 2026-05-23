"""xArm7 + factory gripper composite (menagerie URDF)."""

from __future__ import annotations

import numpy as np

from caliball.robots._base import BaseTF
from caliball.robots._composite import ArmGripperCompositeTF
from caliball.robots._registry import register_robot
from caliball.robots.xarm7 import XArm7URDF, XArm7ArmTF


# ---------------------------------------------------------------------------
# Gripper sub-chain TF
# ---------------------------------------------------------------------------

class _XArm7MenagerieGripperTF(BaseTF):
    """xArm7 menagerie gripper sub-chain; poses in link7 (flange) frame.

    q: (1,) scalar binding left/right finger revolute joints (0 open, 0.85 closed, rad).
    """

    FLANGE_LINK = "link7"
    GRIPPER_NAMES = ["xarm_gripper_base_link"]
    GRIPPER_TCP_NAME = "link_tcp"

    @staticmethod
    def _jindex_for_child_link(robot, child_link_name: str) -> int:
        for lk in robot.links:
            if lk is None or lk.name != child_link_name:
                continue
            return int(lk.jindex)
        raise ValueError(f"xarm7 URDF: no link {child_link_name!r} with jindex")

    def __init__(self, name_list=None, tcp_name=None, **kwargs):
        self.name_list = list(name_list or self.GRIPPER_NAMES)
        self.tcp_name = tcp_name or self.GRIPPER_TCP_NAME
        self._robot = XArm7URDF()
        self._idx_left_finger = self._jindex_for_child_link(self._robot, "left_finger")
        self._idx_right_finger = self._jindex_for_child_link(self._robot, "right_finger")

        self.mesh_adjust = BaseTF.build_mesh_adjust(self._robot, self.name_list)

        q0 = np.zeros(self._robot.n)
        T_fl = self._robot.fkine(q0, end=self.FLANGE_LINK).A
        self._T_fl_inv_zero = np.linalg.inv(T_fl)

    def _q_full(self, g: float) -> np.ndarray:
        q = np.zeros(self._robot.n, dtype=np.float64)
        q[self._idx_left_finger] = g
        q[self._idx_right_finger] = g
        return q

    def fkine_flange(self, q: np.ndarray) -> np.ndarray:
        q = np.asarray(q, dtype=np.float64)
        qf = self._q_full(float(q[0]))
        T_w = self._robot.fkine(qf, end=self.tcp_name).A
        return (self._T_fl_inv_zero @ T_w)[np.newaxis]

    def _fkine_tcp_raw(self, q: np.ndarray) -> np.ndarray:
        return self.fkine_flange(q)

    def fkine_all(self, q: np.ndarray) -> np.ndarray:
        q = np.asarray(q, dtype=np.float64)
        qf = self._q_full(float(q[0]))
        row = []
        for name in self.name_list:
            T_w = self._robot.fkine(qf, end=name).A
            T_rel = self._T_fl_inv_zero @ T_w
            row.append(T_rel @ self.mesh_adjust[name])
        return np.array(row)[np.newaxis]


# ---------------------------------------------------------------------------
# Composite: xArm7 + gripper (with custom fingertip midpoint FK)
# ---------------------------------------------------------------------------

@register_robot("xarm7_xarmgripper")
class XArm7WithGripperTF(ArmGripperCompositeTF):
    """xArm7 (menagerie URDF) + factory gripper. q: (8,) = 7 arm + 1 gripper.

    fkine_flange: link7 flange pose.
    fkine_tcp: left/right fingertip midpoint position + xarm_gripper_base_link orientation.
    fkine_all: arm 8 links + gripper links.
    """

    GRIPPER_CLOSED = 0.85
    ARM_NAME_PREFIX_LEN = 8
    EE_TIP_LEFT_HOM = np.array([0.01323607, -0.0240032, 0.06080743, 1.0], dtype=np.float64)
    EE_TIP_RIGHT_HOM = np.array([-0.01323607, 0.0240032, 0.06080743, 1.0], dtype=np.float64)
    EE_ORIENTATION_FLANGE_LINK = "xarm_gripper_base_link"

    FULL_CHAIN_NAMES = XArm7ArmTF.ARM_NAMES + _XArm7MenagerieGripperTF.GRIPPER_NAMES
    GRIPPER_TCP_NAME = _XArm7MenagerieGripperTF.GRIPPER_TCP_NAME

    LINK_NAMES = [
        "link_base", "link1", "link2", "link3",
        "link4", "link5", "link6", "link7",
        "xarm_gripper_base_link",
    ]
    TCP_NAME = "link_tcp"
    MESH_PATHS = [
        "third_party/urdf/xarm7/assets/link_base.stl",
        "third_party/urdf/xarm7/assets/link1.stl",
        "third_party/urdf/xarm7/assets/link2.stl",
        "third_party/urdf/xarm7/assets/link3.stl",
        "third_party/urdf/xarm7/assets/link4.stl",
        "third_party/urdf/xarm7/assets/link5.stl",
        "third_party/urdf/xarm7/assets/link6.stl",
        "third_party/urdf/xarm7/assets/end_tool.stl",
        "third_party/urdf/xarm7/assets/base_link.stl",
    ]

    def __init__(self, full_chain_names=None, gripper_tcp_name=None, *,
                 gripper_mount_yaw_deg=0.0, grasp_point_rotation_align=None, **kwargs):
        super().__init__(
            arm_joint_num=7,
            gripper_closed_q=float(self.GRIPPER_CLOSED),
            gripper_mount_yaw_deg=gripper_mount_yaw_deg,
        )
        names = list(full_chain_names or self.FULL_CHAIN_NAMES)
        gripper_tcp_name = gripper_tcp_name or self.GRIPPER_TCP_NAME
        self.arm = XArm7ArmTF(names[:self.ARM_NAME_PREFIX_LEN], "link7")
        self.gripper = _XArm7MenagerieGripperTF(names[self.ARM_NAME_PREFIX_LEN:], gripper_tcp_name)
        self.grasp_point_R_align = self._build_grasp_point_R_align(grasp_point_rotation_align)

    def _mount_T_for_tcp(self, q_arm):
        return self.arm.fkine_flange(q_arm)

    def _mount_T_for_gripper_meshes(self, arm_tfs, q_arm):
        return arm_tfs[:, -1]

    def _fkine_tcp_raw(self, q):
        """Left/right fingertip midpoint + gripper base orientation, shape (1, 4, 4)."""
        q = np.atleast_1d(np.asarray(q, dtype=np.float64))
        q_arm = q[:self.arm_joint_num]
        q_grip = q[self.arm_joint_num]

        arm_tfs = self.arm.fkine_all(q_arm)
        T_mount = self._mount_with_gripper_bias(self._mount_T_for_gripper_meshes(arm_tfs, q_arm))

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

        return (T_mount[0] @ Tee)[np.newaxis]
