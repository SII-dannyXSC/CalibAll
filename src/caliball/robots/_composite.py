"""ArmGripperCompositeTF -- base class for arm + gripper composites."""

from __future__ import annotations

import numpy as np

from caliball.robots._base import BaseTF


class ArmGripperCompositeTF(BaseTF):
    """Arm TF + end-effector sub-chain TF (gripper / hand) composite base.

    Subclass ``__init__`` must call::

        super().__init__(arm_joint_num=..., gripper_closed_q=..., gripper_mount_yaw_deg=...)

    then set ``self.arm`` and ``self.gripper``.

    Subclasses may override:
      - _mount_T_for_tcp(q_arm)             -> (1, 4, 4)   used by fkine_tcp
      - _mount_T_for_gripper_meshes(arm_tfs, q_arm) -> (1, 4, 4)   used by fkine_all

    Default implementations use ``self.arm.fkine_flange(q_arm)``.

    Class-level attributes (override in subclass):
      - ARM_JOINT_NUM: int           -- number of arm joints
      - GRIPPER_CLOSED_Q: float      -- closed gripper joint value
      - MOUNT_LINK: str              -- mount link name (for FK to mount point)
      - MOUNT_YAW_DEG: float         -- yaw offset at mount (degrees)
    """

    ARM_JOINT_NUM: int = 7
    GRIPPER_CLOSED_Q: float = 0.0
    MOUNT_LINK: str = ""
    MOUNT_YAW_DEG: float = 0.0

    def __init__(
        self,
        arm_joint_num: int,
        gripper_closed_q: float,
        gripper_mount_yaw_deg: float = 0.0,
        **kwargs,
    ):
        self.arm_joint_num = int(arm_joint_num)
        self.gripper_closed_q = float(gripper_closed_q)
        self.gripper_mount_yaw_deg = float(gripper_mount_yaw_deg)

    @staticmethod
    def _Rz_4x4(rad: float) -> np.ndarray:
        c, s = np.cos(rad), np.sin(rad)
        T = np.eye(4, dtype=np.float64)
        T[0, 0] = c
        T[0, 1] = -s
        T[1, 0] = s
        T[1, 1] = c
        return T

    def _gripper_mount_bias_T(self) -> np.ndarray:
        """(4,4) fixed yaw bias at mount; identity when 0 deg."""
        return self._Rz_4x4(float(np.deg2rad(self.gripper_mount_yaw_deg)))

    def _mount_with_gripper_bias(self, T_mount: np.ndarray) -> np.ndarray:
        """T_mount: (1, 4, 4) -> (1, 4, 4), insert Z-axis mount bias."""
        Tb = self._gripper_mount_bias_T()
        return np.asarray(T_mount, dtype=np.float64) @ Tb

    # ------------------------------------------------------------------
    # Default mount-point methods (can be overridden by subclasses)
    # ------------------------------------------------------------------

    def _mount_T_for_tcp(self, q_arm: np.ndarray) -> np.ndarray:
        """Default: use arm FK to get EEF pose. Override for custom mount.
        q_arm: (arm_joints,) -> (1, 4, 4)
        """
        return self.arm.fkine_flange(q_arm)  # (1, 4, 4)

    def _mount_T_for_gripper_meshes(
        self, arm_tfs: np.ndarray, q_arm: np.ndarray
    ) -> np.ndarray:
        """Default: same as _mount_T_for_tcp. Override for custom mount.
        arm_tfs: (1, n_arm_links, 4, 4), q_arm: (arm_joints,) -> (1, 4, 4)
        """
        return self._mount_T_for_tcp(q_arm)

    # ------------------------------------------------------------------
    # FK interface
    # ------------------------------------------------------------------

    def fkine_flange(self, q: np.ndarray) -> np.ndarray:
        """Arm end-effector TCP (no gripper offset), shape (1, 4, 4)."""
        q = np.asarray(q, dtype=np.float64)
        return self.arm.fkine_flange(q[: self.arm_joint_num])

    def _fkine_tcp_raw(self, q: np.ndarray) -> np.ndarray:
        """Closed-gripper fingertip pose, shape (1, 4, 4)."""
        q = np.asarray(q, dtype=np.float64)
        q_arm = q[: self.arm_joint_num]
        T_mount = self._mount_with_gripper_bias(
            self._mount_T_for_tcp(q_arm)
        )  # (1, 4, 4)
        T_tcp = self.gripper.fkine_tcp(
            np.array([self.gripper_closed_q])
        )  # (1, 4, 4)
        return T_mount @ T_tcp  # (1, 4, 4)

    def fkine_all(self, q: np.ndarray) -> np.ndarray:
        """Arm + gripper all links, shape (1, n_arm+n_grip, 4, 4)."""
        q = np.asarray(q, dtype=np.float64)
        q_arm = q[: self.arm_joint_num]
        q_grip = q[self.arm_joint_num : self.arm_joint_num + 1]  # (1,)
        if q_grip.size == 0:
            q_grip = np.array([0.0])  # default gripper open
        arm_tfs = self.arm.fkine_all(q_arm)  # (1, n_arm_links, 4, 4)
        T_mount = self._mount_with_gripper_bias(
            self._mount_T_for_gripper_meshes(arm_tfs, q_arm)
        )  # (1, 4, 4)
        grip_loc = self.gripper.fkine_all(q_grip)  # (1, n_grip_links, 4, 4)
        grip_w = T_mount[:, np.newaxis] @ grip_loc  # (1, n_grip_links, 4, 4)
        return np.concatenate([arm_tfs, grip_w], axis=1)  # (1, n_total, 4, 4)
