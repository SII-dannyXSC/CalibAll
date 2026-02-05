import numpy as np
from torch.utils.data import Dataset
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset as _LeRobotDataset
from typing import Optional, List


class LeRobotDataset(Dataset):
    """
    LeRobot 2.1数据集读取器
    
    支持从HuggingFace Hub加载LeRobot格式的机器人数据集
    
    Args:
        repo_id: HuggingFace上的数据集repo ID，例如 "lerobot/aloha_mobile_cabinet"
        episodes: 要加载的episode索引列表，None表示加载所有episodes
        split: 数据集分割，默认为"train"
        root_dir: 本地缓存目录
        image_transforms: 图像变换函数
    """
    
    def __init__(
        self, 
        repo_id: str,
        episodes: Optional[List[int]] = None,
        split: str = "train",
        root_dir: Optional[str] = None,
        image_transforms=None,
    ) -> None:
        super().__init__()
        
        self.repo_id = repo_id
        self.split = split
        self.root_dir = root_dir
        self.image_transforms = image_transforms
        
        # 加载LeRobot数据集
        if episodes is not None:
            self.lerobot_ds = _LeRobotDataset(
                repo_id=repo_id,
                episodes=episodes,
                root=root_dir,
                video_backend="pyav",  # 添加这一行
            )
        else:
            self.lerobot_ds = _LeRobotDataset(
                repo_id=repo_id,
                root=root_dir,
                video_backend="pyav",  # 添加这一行
            )
        
        # 获取数据集元信息
        self.meta = self.lerobot_ds.meta
        self.fps = self.meta.fps
        self.robot_type = self.meta.robot_type
        self.camera_keys = self.meta.camera_keys if hasattr(self.meta, 'camera_keys') else []
        
        # 获取episode索引信息（不加载实际数据）
        self.episode_data_index = self.lerobot_ds.episode_data_index
        self.total_episodes = len(self.episode_data_index['from'])
        
        print(f"初始化LeRobot数据集: {self.repo_id}")
        print(f"总帧数: {len(self.lerobot_ds)}")
        print(f"总episodes: {self.total_episodes}")
        print(f"FPS: {self.fps}")
        print(f"机器人类型: {self.robot_type}")
        print(f"相机键: {self.camera_keys}")
        print("数据将在访问时按需加载")
    
    def __len__(self):
        """获取数据集中的episode数量"""
        return self.total_episodes
    
    def __getitem__(self, idx):
        """
        获取指定索引的episode数据（懒加载模式）
        
        Args:
            idx: episode索引
            
        Returns:
            dict: 包含以下键的字典
                - video: 主相机的视频帧数组 (T, H, W, C)
                - videos: 所有相机的视频帧字典
                - states: 状态序列 (T, state_dim)
                - actions: 动作序列 (T, action_dim)
                - name: episode名称
        """
        # 获取该episode的起止帧索引
        start_idx = self.episode_data_index['from'][idx].item()
        end_idx = self.episode_data_index['to'][idx].item()
        
        episode_frames = []
        episode_states = []
        episode_actions = []
        
        # 按需读取该episode的所有帧
        for frame_idx in range(start_idx, end_idx):
            data = self.lerobot_ds[frame_idx]
            
            # 提取图像数据（支持多个相机）
            frames = {}
            for cam_key in self.camera_keys:
                if cam_key in data:
                    img = data[cam_key].numpy()
                    if self.image_transforms is not None:
                        img = self.image_transforms(img)
                    frames[cam_key] = img
            
            # 提取状态和动作
            state_key = "observation.states.joint_position"
            action_key = "observation.states.end_effector"
            state = data[state_key].numpy() if state_key in data else None
            action = data[action_key].numpy() if action_key in data else None
            
            episode_frames.append(frames)
            episode_states.append(state)
            episode_actions.append(action)
        
        # 构建视频数据
        videos = {}
        for cam_key in self.camera_keys:
            frames_list = [frame[cam_key] for frame in episode_frames if cam_key in frame]
            if frames_list:
                # 将每帧从 (C, H, W) 转换为 (H, W, C)，并转换为 uint8 格式
                processed_frames = []
                for img in frames_list:
                    # Transpose: (C, H, W) -> (H, W, C)
                    if img.ndim == 3:
                        img = img.transpose(1, 2, 0)
                    # 如果是浮点数 [0, 1]，转换为 uint8 [0, 255]
                    if img.dtype in [np.float32, np.float64]:
                        img = (img * 255).clip(0, 255).astype(np.uint8)
                    processed_frames.append(img)
                videos[cam_key] = np.array(processed_frames)
        
        # 默认使用第一个相机作为主视频
        main_video = videos['observation.images.camera_right'] if self.camera_keys else None
        
        # 转换状态和动作为numpy数组
        states = np.array(episode_states) if episode_states and episode_states[0] is not None else None
        actions = np.array(episode_actions) if episode_actions and episode_actions[0] is not None else None
        
        return dict(
            video=main_video,
            videos=videos,
            states=states,
            actions=actions,
            name=f"{self.repo_id}_episode_{idx}",
        )
    
    def get_stats(self, max_episodes: Optional[int] = None):
        """
        获取数据集统计信息
        
        Args:
            max_episodes: 用于计算统计信息的最大episode数量，None表示使用所有episodes
        
        Returns:
            dict: 包含状态和动作的统计信息
        """
        print("正在计算数据集统计信息...")
        
        all_states = []
        all_actions = []
        
        num_episodes = min(max_episodes, self.total_episodes) if max_episodes else self.total_episodes
        
        for idx in range(num_episodes):
            sample = self[idx]
            if sample['states'] is not None:
                all_states.append(sample['states'])
            if sample['actions'] is not None:
                all_actions.append(sample['actions'])
            
            if (idx + 1) % 10 == 0:
                print(f"已处理 {idx + 1}/{num_episodes} episodes")
        
        stats = {}
        if all_states:
            all_states = np.concatenate(all_states, axis=0)
            stats['state_mean'] = np.mean(all_states, axis=0)
            stats['state_std'] = np.std(all_states, axis=0)
            stats['state_min'] = np.min(all_states, axis=0)
            stats['state_max'] = np.max(all_states, axis=0)
        
        if all_actions:
            all_actions = np.concatenate(all_actions, axis=0)
            stats['action_mean'] = np.mean(all_actions, axis=0)
            stats['action_std'] = np.std(all_actions, axis=0)
            stats['action_min'] = np.min(all_actions, axis=0)
            stats['action_max'] = np.max(all_actions, axis=0)
        
        print("统计信息计算完成")
        return stats


if __name__ == "__main__":
    # 使用示例
    
    # 1. 加载完整数据集
    # dataset = LeRobotDataset(
    #     repo_id="lerobot/aloha_mobile_cabinet",
    #     root_dir="/path/to/cache"
    # )
    
    # 2. 加载指定episodes
    # dataset = LeRobotDataset(
    #     repo_id="lerobot/aloha_mobile_cabinet",
    #     episodes=[0, 1, 2, 3, 4],
    #     root_dir="/path/to/cache"
    # )
    
    # 3. 测试数据加载
    # print(f"数据集大小: {len(dataset)}")
    # sample = dataset[0]
    # print(f"Episode名称: {sample['name']}")
    # print(f"视频形状: {sample['video'].shape if sample['video'] is not None else 'None'}")
    # print(f"状态形状: {sample['states'].shape if sample['states'] is not None else 'None'}")
    # print(f"动作形状: {sample['actions'].shape if sample['actions'] is not None else 'None'}")
    # print(f"相机数量: {len(sample['videos'])}")
    
    # 4. 获取统计信息
    # stats = dataset.get_stats()
    # print(f"状态均值: {stats.get('state_mean', 'N/A')}")
    # print(f"动作均值: {stats.get('action_mean', 'N/A')}")
    
    # 初始化非常快，不加载数据
    dataset = LeRobotDataset("/cpfs01/user/wenji.zj/dataspace/Data4QwenVLA/RoboMIND_lerobot_v2.1/benchmark1_1_compressed/franka_3rgb/put_the_red_apple_in_the_bowl")

    # 只在访问时才加载该episode的数据
    import pdb; pdb.set_trace()
    sample = dataset[0]  # 这时才读取episode 0的数据

    # 可以只用前10个episodes计算统计信息
    # stats = dataset.get_stats(max_episodes=10)

    print("LeRobot数据集加载器已准备就绪！")
    print("\n使用方法:")
    print("1. 加载数据集:")
    print('   dataset = LeRobotDataset("lerobot/aloha_mobile_cabinet")')
    print("\n2. 加载指定episodes:")
    print('   dataset = LeRobotDataset("lerobot/pusht", episodes=[0, 1, 2])')
    print("\n3. 获取数据:")
    print("   sample = dataset[0]")
    print("   video = sample['video']")
    print("   states = sample['states']")
    print("   actions = sample['actions']")
