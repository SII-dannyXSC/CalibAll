from abc import ABC, abstractmethod

class BaseTF(ABC):
    def __init__(self):
        pass
    
    @abstractmethod
    def fkine(self,q):
        """
        前向运动学计算末端基座的位姿(最后一个 link)
        
        参数:
            q : torch.Tensor 或 np.ndarray, 关节角度，shape = (N, n_joints)
        
        返回:
            T_all : 末端的齐次变换矩阵，通常 shape = (N, 4, 4)
        """
        pass

    @abstractmethod
    def fkine_all(self, q):
        """
        前向运动学计算所有关节位姿
        
        参数:
            q : torch.Tensor 或 np.ndarray, 关节角度，shape = (N, n_joints)
        
        返回:
            T_all : 所有关节的齐次变换矩阵，通常 shape = (N, n_joints, 4, 4)
        """
        pass
