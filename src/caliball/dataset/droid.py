import numpy as np
import tensorflow as tf

from torch.utils.data import Dataset
import tensorflow_datasets as tfds

class DroidDataset(Dataset):
    def __init__(self, root_dir, name="droid", split="train") -> None:
        super().__init__()
        self.root_dir = root_dir
        with tf.device('/CPU:0'):
            ds = tfds.load(name, data_dir=root_dir, split=split, shuffle_files=False)
        
        total_video1 = []
        total_video2 = []
        total_qpos = []
        total_name = []
        total_action = []
        total_language = []
        total_state = []
        total_gripper = []
        total_wrist = []
        for episode in ds:
            file_path = episode["episode_metadata"]["file_path"].numpy().decode("utf-8")
            episode_v1 = []
            episode_v2 = []
            episode_qpos = []
            episode_action = []
            episode_language = []
            episode_state = []
            episode_gripper = []
            episode_wrist = []
            
            for i, step in enumerate(episode["steps"]):
                img1 = step["observation"]["exterior_image_1_left"].numpy()
                img2 = step["observation"]["exterior_image_2_left"].numpy()
                img_wrist = step["observation"]["wrist_image_left"].numpy()
                qpos = step["observation"]["joint_position"].numpy()
                
                action = step['action'].numpy()
                language = step['language_instruction'].numpy()
                state = step["observation"]['cartesian_position'].numpy()
                gripper = step["observation"]['gripper_position'].numpy()
                state =np.concatenate([state, gripper])
                
                episode_v1.append(img1)
                episode_v2.append(img2)
                episode_wrist.append(img_wrist)
                episode_qpos.append(qpos)
                episode_action.append(action)
                episode_language.append(language)
                episode_state.append(state)
                episode_gripper.append(gripper)
                
            episode_v1 = np.array(episode_v1)
            episode_v2 = np.array(episode_v2)
            episode_wrist = np.array(episode_wrist)
            episode_qpos = np.array(episode_qpos)
            
            episode_action = np.array(episode_action)
            episode_language = np.array(episode_language)
            episode_state = np.array(episode_state)
            episode_gripper = np.array(episode_gripper)
                
            total_name.append(file_path)
            total_video1.append(episode_v1)
            total_video2.append(episode_v2)
            total_wrist.append(episode_wrist)
            total_qpos.append(episode_qpos)
            
            total_action.append(episode_action)
            total_language.append(episode_language)
            total_state.append(episode_state)
            total_gripper.append(episode_gripper)
        
        self.total_name = total_name
        self.total_video1 = total_video1
        self.total_video2 = total_video2
        self.total_wrist = total_wrist
        self.total_qpos = total_qpos
        
        self.total_action = total_action
        self.total_language = total_language
        self.total_state = total_state
        self.total_gripper = total_gripper
        
        self.length = len(self.total_qpos)
            
 
    def __len__(self):
        """
        获取数据集的样本数量
        """
        return len(self.total_qpos)
    
    def __getitem__(self, idx):
        return dict(
            video = self.total_video1[idx],
            video1 = self.total_video1[idx],
            video2 = self.total_video2[idx],
            wrist = self.total_wrist[idx],
            states = self.total_qpos[idx],
            name= self.total_name[idx],
            action=self.total_action[idx],
            language=self.total_language[idx],
            state=self.total_state[idx],
            gripper=self.total_gripper[idx],
        )

if __name__=="__main__":
    dataset = DroidDataset("/cpfs02/user/xiesicheng.xsc/CalibAll/data",split="train[:10]")
    import pdb;pdb.set_trace()
    input()