from abc import ABC, abstractmethod
import numpy as np

from src.caliball.robot.base import BaseTF

class RoboticsToolBoxTF(BaseTF, ABC):
    def __init__(self, name_list, eef_name):
        self.name_list = name_list
        self.eef_name = eef_name
    
    @property
    @abstractmethod
    def robot(self):
        pass
    
    def fkine(self, qpos):
        result = []
        for q in qpos:
            eef_pose = self.robot.fkine(q,end=self.eef_name).A
            result.append(eef_pose)
        result = np.array(result)
        return result
    
    def fkine_all(self, qpos):
        result = []
        for q in qpos:
            all_tf = []
            for name in self.name_list:
                pose = self.robot.fkine(q, end=name).A
                all_tf.append(pose)
            result.append(all_tf)
        result = np.array(result)
        return result
                