
import time
import os
from PIL import Image
from src.caliball.robot.franka import FrankaTF
from omegaconf import OmegaConf
import numpy as np
import trimesh


from src.caliball.coarse_init import CoarseInit
from src.caliball.dataset.droid import DroidDataset
from src.caliball.dataset.lerobot_dataset import LeRobotDataset
from src.caliball.dataset.taco_play_dataset import TacoPlayDataset
from src.caliball.refinement import Refinement
from src.caliball.dataset.tfds_dataset import TfdsDataset
from src.caliball.utils.visual import sample_robot_points
import debugpy
debugpy.listen(("0.0.0.0", 10092))
print("🔍 Waiting for VSCode attach on 0.0.0.0:10092 ...")
debugpy.wait_for_client()

img_pil = Image.open("assets/test_img/source.png").convert("RGB")
p = (376, 131)

config = type('', (), {})()  # create empty config object  
config.ckpt_path = "ckpt/dinov2/dinov2_vitb14_pretrain.pth"
config.repo_dir = "third_party/dinov2"
config.dino_id = "dinov2_vitb14"
config.tracker_repo_dir = "third_party/co-tracker"
config.tracker_id = "cotracker3_offline"
config.tracker_ckpt_path = "ckpt/cotracker/scaled_offline.pth"
config.bpe_path = "third_party/sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz"
config.ckpt_path = "ckpt/sam3/sam3.pt"

# dataset = DroidDataset("/inspire/hdd/global_user/xiesicheng-253108120120/data/droid_example", name="droid_100",split="train[:10]")
# dataset = LeRobotDataset("/cpfs01/user/wenji.zj/dataspace/Data4QwenVLA/RoboMIND_lerobot_v2.1/benchmark1_1_compressed/franka_3rgb/put_the_red_apple_in_the_bowl")
# dataset = LeRobotDataset("/cpfs01/user/wenji.zj/dataspace/Data4QwenVLA/RoboMIND_lerobot_v2.1/benchmark1_1_compressed/franka_3rgb/put_the_red_apple_in_the_bowl")

ROOT = "/inspire/hdd/global_user/xiesicheng-253108120120/project/dzj/CalibAll/dataset/tfds/"
NAME = "toto"

dataset = TfdsDataset(
    root_dir=ROOT,
    name=NAME,
    split="train",       # 现在用 train 也不会在 init 爆内存（逐 episode 读）
    max_episodes=3,
    max_steps=200,       # 先小一点验证；跑通再放开/设为 None
)

tf = FrankaTF()

mesh_cfg = OmegaConf.load("src/caliball/config/robot/franka.yaml")
mesh_paths = mesh_cfg.mesh_paths

cnt = 0
for data in dataset:
    video = data["video"]    # T H W C
    # joint_angles = data["states"]  # T 6
    joint_angles = data["states"]  # T 6
    
    length = len(video)

    video = video[length//2:]
    joint_angles = joint_angles[length//2:]
    
    img = video[0]
    cur_qpos = joint_angles[0]
    
    tf_list = tf.fkine_all([cur_qpos])[0]

    save_path = f"results/checkqpos_{time.time()}_{cnt}"
    os.makedirs(save_path, exist_ok=True)

    Image.fromarray(img).save(os.path.join(save_path, "cur_img.png"))
    all_points = sample_robot_points(tf_list, mesh_paths)
    
    cloud = trimesh.points.PointCloud(all_points)
    cloud.export(os.path.join(save_path,"test.ply"))

    cnt += 1
    if cnt > 10:
        break