import numpy as np
import roboticstoolbox as rtb

from caliball.robot.base import BaseTF
class FrankaTF(BaseTF):
    JOINT_NUM = 7
    def __init__(self):
        self.robot = rtb.models.Panda()
    
    def fkine(self,qpos):
        result = []
        for q in qpos:
            q = q[:self.JOINT_NUM]
            all_tf = self.robot.fkine_all(q)
            eef_pose = all_tf[9].A
            # eef_pose = self.robot.fkine(q,end="panda_link8")
            result.append(eef_pose)
        result = np.array(result)
        return result

    def fkine_all(self, qpos):
        result = []
        for q in qpos:
            q = q[:self.JOINT_NUM]
            all_tf = self.robot.fkine_all(q)
            link_8 = all_tf[9]
            all_tf = all_tf[1:9]
            tf_hand = link_8 @ self.robot.grippers[0].links[0].A()
            all_tf = np.array([item.A for item in all_tf] + [tf_hand])
            
            result.append(all_tf)
        result = np.array(result)
        return result

    def fkine_gripper(self,qpos):
        left_R = self.robot.grippers[0].links[1].A()
        result = []
        for q in qpos:
            q = q[:self.JOINT_NUM]
            all_tf = self.robot.fkine_all(q)
            # eef_pose = all_tf[9].A
            link8 = self.robot.fkine(q,end="panda_link8")
            tf_hand = link8 @ self.robot.grippers[0].links[0].A()
            gripper_pose = tf_hand @ left_R @ left_R
            gripper_pose = gripper_pose.A
            result.append(gripper_pose)
        result = np.array(result)
        return result

if __name__=="__main__":
    angle = np.array(
        [
            0.02597709,
            0.18678349,
            -0.02388557,
            -2.58533829,
            -0.01678507,
            2.97463095,
            0.78241131,
            0.020833,
        ]
    )
    angle = np.expand_dims(angle,axis=0)
    tf = FrankaTF()
    import pdb;pdb.set_trace()
    result = tf.fkine(angle)
    