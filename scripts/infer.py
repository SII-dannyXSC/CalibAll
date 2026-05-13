
import time
import os
from PIL import Image

from src.caliball.coarse_init import CoarseInit
from src.caliball.dataset.droid import DroidDataset
from src.caliball.dataset.lerobot_dataset import LeRobotDataset
from src.caliball.dataset.taco_play_dataset import TacoPlayDataset
from src.caliball.dataset.berkeley_ur5_dataset import BerkeleyUr5Dataset
from src.caliball.dataset.tfds_dataset import TfdsDataset
from src.caliball.refinement import Refinement
import debugpy
# debugpy.listen(("0.0.0.0", 10092))
# print("🔍 Waiting for VSCode attach on 0.0.0.0:10092 ...")
# debugpy.wait_for_client()

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
# config.robot_type = "ur5e"
config.robot_type = "franka"

dataset = LeRobotDataset("/cpfs01/user/wenji.zj/dataspace/Data4QwenVLA/RoboMIND_lerobot_v2.1/benchmark1_1_compressed/franka_3rgb/put_the_red_apple_in_the_bowl")

# ROOT = "/inspire/hdd/global_user/xiesicheng-253108120120/project/dzj/CalibAll/dataset/tfds/"
# NAME = "berkeley_autolab_ur5"

# dataset = BerkeleyUr5Dataset(
#     root_dir=ROOT,
#     name=NAME,
#     split="train",       # 现在用 train 也不会在 init 爆内存（逐 episode 读）
#     max_episodes=3,
#     max_steps=200,       # 先小一点验证；跑通再放开/设为 None
# )

# ROOT = "/inspire/hdd/global_user/xiesicheng-253108120120/project/dzj/CalibAll/dataset/tfds/"
# NAME = "toto"

# dataset = TfdsDataset(
#     root_dir=ROOT,
#     name=NAME,
#     split="train",       # 现在用 train 也不会在 init 爆内存（逐 episode 读）
#     max_episodes=3,
#     max_steps=200,       # 先小一点验证；跑通再放开/设为 None
# )


corase_init_pipe = CoarseInit(config=config)
corase_init_pipe.to("cuda")
corase_init_pipe._init_recognizer(img_pil, p)
corase_init_pipe._init_intrinsic()

refinement_pipe = Refinement(config=config)

import numpy as np
# intrinsic = np.array([[531,0,636], [0,531,344], [0,0,4]]) / 4
# corase_init_pipe._init_intrinsic(intrinsic)
# intrinsic = corase_init_pipe._get_intrinsic(img_pil)
# import pdb;pdb.set_trace()

cnt = 0
data = dataset[1]
if 1:
# for data in dataset:
    videos = data["videos"]    # T H W C
    video = videos["observation.images.camera_left"]

    joint_angles = data["states"]  # T 6

    length = len(video)

    save_path = f"results/{time.time()}_{cnt}"
    os.makedirs(save_path, exist_ok=True)

    # frame_path = f"{save_path}/frames"
    # os.makedirs(frame_path, exist_ok=True)
    # for i, img in enumerate(video):
    #     img_save = Image.fromarray(img)
    #     img_save.save(os.path.join(frame_path, f"frame_{i:04d}.png"))
    # exit()

    start = 51
    # start = 0
    # end = min(length, start + 60)
    end = 125

    video = video[start:end]
    joint_angles = joint_angles[start:end]

    extrinsic, intrinsic = corase_init_pipe.get_extrinsic(video=video, joint_angles=joint_angles, img_idx=0, save_path=save_path)
    print(f"{extrinsic=}")
    print(f"{intrinsic=}")

    result, loss_dict = refinement_pipe.refine(video=video, joint_angles=joint_angles, intrinsic=intrinsic, extrinsic=extrinsic, base_path=save_path)
    exit()
