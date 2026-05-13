from typing import Literal

import numpy as np
from scipy.spatial.transform import Rotation as R

from src.caliball.robot.base import BaseTF
from src.caliball.robot.dual_arm.dual_arm_base import DualArmTF
from src.caliball.robot.urdf.aloha import AlohaCobotMagic

GRIPPER_MAX_DIST = 0.1

_PREFIX = {"left": "fl_", "right": "fr_"}
_DEFAULT_EEF = {"left": "fl_link6", "right": "fr_link6"}
_TIP_LINK = {"left": "fl_link8", "right": "fr_link8"}


def _build_joint_idx_map(robot):
    """robot 全关节名 → q 向量索引。"""
    pairs = []
    for link in robot.links:
        j = getattr(link, "joint", None)
        if j is None or not getattr(j, "isjoint", True):
            continue
        name = (getattr(j, "name", None) or "").strip()
        ji = getattr(j, "jindex", None)
        if not name or ji is None:
            continue
        pairs.append((int(ji), name))
    if pairs:
        pairs.sort(key=lambda x: x[0])
        if len(pairs) == int(getattr(robot, "n", len(pairs))):
            return {nm: i for i, nm in pairs}
    # 回退：按 links 顺序枚举
    result = []
    for link in robot.links:
        j = getattr(link, "joint", None)
        if j is not None and getattr(j, "isjoint", True):
            result.append((getattr(j, "jindex", len(result)), getattr(j, "name", "")))
    result.sort(key=lambda x: x[0])
    return {nm: i for i, nm in result}


class AlohaArmTF(BaseTF):
    """ALOHA 单臂。q: (7,) = 6 臂关节 + 1 夹爪距离 [0, 0.1]。

    持有双臂整体 robot（可共享），通过 side 确定关节前缀（fl_ / fr_）。
    夹爪内部展开为两对称 prismatic：joint7 = +g/2, joint8 = -g/2。

    fkine_eef:     腕部（eef_name）位姿，shape (1, 4, 4)。
    fkine_gripper: 指尖（link8）位姿，shape (1, 4, 4)。
    fkine_all:     本臂所有 link（含 mesh 对齐），shape (1, n_links, 4, 4)。
    """

    def __init__(
        self,
        side: Literal["left", "right"],
        name_list,
        eef_name: str,
        robot=None,
    ):
        self.side = side
        self.name_list = list(name_list)
        self.eef_name = eef_name or _DEFAULT_EEF[side]
        self.tip_link = _TIP_LINK[side]
        self._prefix = _PREFIX[side]

        self._robot = robot if robot is not None else AlohaCobotMagic()
        self._jmap = _build_joint_idx_map(self._robot)

        self.mesh_adjust = {name: np.eye(4) for name in self.name_list}
        for link in self._robot.links:
            if link.name not in self.mesh_adjust or len(link.geometry) == 0:
                continue
            geo = link.geometry[0]
            trans_mat = np.eye(4)
            trans_mat[:3, 3] = geo._wT[:3, 3]
            trans_mat[:3, :3] = R.from_quat(geo._wq).as_matrix()
            scale_mat = np.eye(4)
            scale_mat[:3, :3] = np.diag(geo.scale)
            self.mesh_adjust[link.name] = trans_mat @ scale_mat

    def _expand_q(self, q: np.ndarray) -> np.ndarray:
        """q: (7,) → 全 robot 关节向量 (nq,)，其余关节置 0。"""
        q = np.asarray(q, dtype=np.float64)
        p = self._prefix
        idx = self._jmap
        out = np.zeros(self._robot.n, dtype=np.float64)
        for i in range(6):
            jname = f"{p}joint{i + 1}"
            if jname in idx:
                out[idx[jname]] = q[i]
        gripper = float(np.clip(q[6], 0, GRIPPER_MAX_DIST))
        if f"{p}joint7" in idx:
            out[idx[f"{p}joint7"]] = gripper / 2
        if f"{p}joint8" in idx:
            out[idx[f"{p}joint8"]] = -gripper / 2  # 对称张开
        return out

    def fkine_eef(self, q: np.ndarray) -> np.ndarray:
        T = self._robot.fkine(self._expand_q(q), end=self.eef_name).A
        return T[np.newaxis]  # (1, 4, 4)

    def _fkine_gripper_raw(self, q: np.ndarray) -> np.ndarray:
        T = self._robot.fkine(self._expand_q(q), end=self.tip_link).A
        return T[np.newaxis]  # (1, 4, 4)

    def fkine_all(self, q: np.ndarray) -> np.ndarray:
        q_full = self._expand_q(q)
        tfs = np.array([self._robot.fkine(q_full, end=name).A for name in self.name_list])
        for i, name in enumerate(self.name_list):
            tfs[i] = tfs[i] @ self.mesh_adjust[name]
        return tfs[np.newaxis]  # (1, n_links, 4, 4)


class AlohaCobotMagicTF(DualArmTF):
    """ALOHA 双臂。q: (14,) = 左臂 7 + 右臂 7，顺序 [左, 右]。

    names 为全部 link 名（fl_* 自动分配左臂，fr_* 自动分配右臂）。
    eef_name: str 或 [left_eef, right_eef]，缺省为 fl_link6 / fr_link6。
    """

    def __init__(self, names, eef_name, grasp_point_rotation_align=None):
        all_names = list(names)
        left_names = [n for n in all_names if n.startswith("fl_")]
        right_names = [n for n in all_names if n.startswith("fr_")]

        if isinstance(eef_name, str):
            left_eef, right_eef = _DEFAULT_EEF["left"], eef_name
        else:
            seq = list(eef_name)
            left_eef = seq[0] if len(seq) >= 1 else _DEFAULT_EEF["left"]
            right_eef = seq[1] if len(seq) >= 2 else _DEFAULT_EEF["right"]

        _robot = AlohaCobotMagic()  # 加载一次，左右共享
        super().__init__(
            left=AlohaArmTF("left", left_names, left_eef, robot=_robot),
            right=AlohaArmTF("right", right_names, right_eef, robot=_robot),
            n_left_joints=7,
        )
        self.grasp_point_R_align = self._build_grasp_point_R_align(grasp_point_rotation_align)
