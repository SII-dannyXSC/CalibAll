from PIL import Image

from caliball.coarse_init import CoarseInit
from caliball.dataset.droid import DroidDataset
from caliball.dataset.lerobot_dataset import LeRobotDataset
from caliball.refinement import Refinement
import debugpy
# debugpy.listen(("0.0.0.0", 10092))
# print("🔍 Waiting for VSCode attach on 0.0.0.0:10092 ...")
# debugpy.wait_for_client()

img_pil = Image.open("data/test_img/robot/ep_0_frame_100_origin.png").convert("RGB")
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

# dataset = DroidDataset("/cpfs02/user/xiesicheng.xsc/CalibAll/data",split="train[:10]")
# dataset = LeRobotDataset("/cpfs01/user/wenji.zj/dataspace/Data4QwenVLA/RoboMIND_lerobot_v2.1/benchmark1_1_compressed/franka_3rgb/put_the_red_apple_in_the_bowl")
dataset = LeRobotDataset("/cpfs01/user/wenji.zj/dataspace/Data4QwenVLA/RoboMIND_lerobot_v2.1/benchmark1_1_compressed/franka_3rgb/put_the_red_apple_in_the_bowl")
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
for data in dataset:
    video = data["video"]    # T H W C
    joint_angles = data["states"]  # T 6

    length = len(video)

    video = video[length//2:]
    joint_angles = joint_angles[length//2:]

    extrinsic, intrinsic = corase_init_pipe.get_extrinsic(video=video, joint_angles=joint_angles, img_idx=0)
    print(f"{extrinsic=}")
    
    # result, loss_dict = refinement_pipe.render(video=video, joint_angles=joint_angles, intrinsic=intrinsic, extrinsic=extrinsic)
    result, loss_dict = refinement_pipe.refine(video=video, joint_angles=joint_angles, intrinsic=intrinsic, extrinsic=extrinsic)

    cnt += 1
    if cnt > 10:
        break