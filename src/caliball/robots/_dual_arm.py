"""DualArmTF -- generic dual-arm composite base class."""

from __future__ import annotations

import numpy as np

from caliball.robots._base import BaseTF


class DualArmTF(BaseTF):
    """Generic dual-arm composite.  Holds left and right single-arm TFs.

    q: (n_left_joints + n_right_joints,), first half left, second half right.
    fkine_flange:     (2, 4, 4)
    fkine_tcp: (2, 4, 4)
    fkine_all:     (2, n_links, 4, 4)  -- left and right must have same n_links
    """

    n_arms = 2

    def __init__(self, left: BaseTF, right: BaseTF, n_left_joints: int, **kwargs):
        self.left = left
        self.right = right
        self.n_left_joints = n_left_joints

    def fkine_flange(self, q: np.ndarray) -> np.ndarray:
        q = np.asarray(q, dtype=np.float64)
        ql, qr = q[: self.n_left_joints], q[self.n_left_joints :]
        return np.stack(
            [self.left.fkine_flange(ql)[0], self.right.fkine_flange(qr)[0]]
        )  # (2, 4, 4)

    def _fkine_tcp_raw(self, q: np.ndarray) -> np.ndarray:
        q = np.asarray(q, dtype=np.float64)
        ql, qr = q[: self.n_left_joints], q[self.n_left_joints :]
        return np.stack(
            [self.left.fkine_tcp(ql)[0], self.right.fkine_tcp(qr)[0]]
        )  # (2, 4, 4)

    def fkine_all(self, q: np.ndarray) -> np.ndarray:
        q = np.asarray(q, dtype=np.float64)
        ql, qr = q[: self.n_left_joints], q[self.n_left_joints :]
        tfl = self.left.fkine_all(ql)   # (1, n_links, 4, 4)
        tfr = self.right.fkine_all(qr)  # (1, n_links, 4, 4)
        return np.stack([tfl[0], tfr[0]])  # (2, n_links, 4, 4)

    def gripper_scalar(self, q: np.ndarray) -> float:
        """Left arm gripper scalar (last joint of left arm)."""
        return float(np.asarray(q, dtype=np.float64)[self.n_left_joints - 1])

    def gripper_scalars(self, q: np.ndarray) -> np.ndarray:
        """Return [left_gripper, right_gripper], shape (2,)."""
        q = np.asarray(q, dtype=np.float64)
        ql = q[: self.n_left_joints]
        qr = q[self.n_left_joints :]
        return np.array(
            [self.left.gripper_scalar(ql), self.right.gripper_scalar(qr)],
            dtype=np.float64,
        )
