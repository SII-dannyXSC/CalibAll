from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

dataset = LeRobotDataset(
    repo_id="/cpfs02/user/xiesicheng.xsc/project/CalibAll/data/caliball_anno_data/ur_1rgb/benchmark1_0_compressed/green_pepper_in_basket_1",  # 数据集名
)

print(len(dataset))
print(dataset[-1])
