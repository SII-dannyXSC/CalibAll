import numpy as np
import roboticstoolbox as rtb
from scipy.spatial.transform import Rotation as R
from caliball.robot.rtb import RoboticsToolBoxTF
from caliball.robot.urdf.ur5e import Ur5e

class Ur5eTF(RoboticsToolBoxTF):
    def __init__(self, name_list, eef_name):
        super().__init__(name_list, eef_name)
        self._robot = Ur5e()
        
        self.mesh_adjust = {name: np.eye(4) for name in self.name_list}
        for link in self.robot.links:
            if not link.name in self.mesh_adjust or len(link.geometry) == 0:
                continue
            geo = link.geometry[0]
            t = geo._wT[:3, 3]
            q = geo._wq
            scale = geo.scale
            
            trans_mat = np.eye(4)
            trans_mat[:3,3 ] = t
            trans_mat[:3,:3] = R.from_quat(q).as_matrix()
            
            scale_mat = np.eye(4)
            scale_mat[:3,:3] = np.diag(scale)
            self.mesh_adjust[link.name] =  trans_mat @ scale_mat


    @property
    def robot(self):
        return self._robot

    def fkine_all(self, qpos):
        result = super().fkine_all(qpos)
        
        bc_num = len(result)
        for i in range(bc_num):
            for idx, name in enumerate(self.name_list):
                result[i][idx] = result[i][idx] @ self.mesh_adjust[name]
        
        return result

if __name__=="__main__":
    from omegaconf import OmegaConf
    cfg = OmegaConf.load("./caliball/config/robot/ur5e.yaml")
    robot = Ur5eTF(cfg.names, cfg.eef_name)
    import pdb;pdb.set_trace()
    input()