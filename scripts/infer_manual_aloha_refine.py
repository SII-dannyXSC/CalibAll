
import time
import os
from PIL import Image
import numpy as np
import json

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

json_path = "/cpfs02/user/xiesicheng.xsc/project/CalibAll/manual_label/robomind.agilex_3rgb.1_potatooven.observation.images.camera_front.config.json"

with open(json_path, "r") as f:
    config = json.load(f)

dataset_name = config["dataset_name"]
task_name = config["task_name"]
task_path = config["task_path"]
robot_type = "arx5_robotwin"
episode_idx = config["episode_idx"]
camera_name = config["camera_name"]
start_idx = config["start_idx"]
end_idx = config["end_idx"]
tracking_point = config["tracking_point"]
mask_frame_idx = config["mask_frame_idx"]
mask_save_path = config["mask_save_path"]
mask = np.load(mask_save_path)


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
config.robot_type = robot_type

dataset = LeRobotDataset(
    task_path,
    state_key = "observation.states.joint_position_right",
)

corase_init_pipe = CoarseInit(config=config)
corase_init_pipe.to("cuda")
corase_init_pipe._init_recognizer(img_pil, p)
corase_init_pipe._init_intrinsic()

refinement_pipe = Refinement(config=config)

cnt = 0
data = dataset[episode_idx]
if 1:
# for data in dataset:
    videos = data["videos"]    # T H W C
    video = videos[camera_name]

    joint_angles = data["states"]  # T 7
    joint_angles = np.concatenate([np.zeros((joint_angles.shape[0], 7)), joint_angles], axis=1)
    joint_angles[..., -1] = 0.1

    length = len(video)

    save_path = f"results/{dataset_name}.{task_name}.{camera_name}.{time.time()}_{cnt}"
    os.makedirs(save_path, exist_ok=True)

    start = start_idx
    end = end_idx

    video = video[start:end]
    joint_angles = joint_angles[start:end]

    # extrinsic, intrinsic = corase_init_pipe.get_extrinsic(video=video, joint_angles=joint_angles, tracking_point=tracking_point, img_idx=0, save_path=save_path)
    intrinsic = np.array(
        [[501.63818,   0.,      320.     ],
         [  0.,      501.6382,  240.     ],
         [  0.,        0.,        1.     ]]
    )
    extrinsic = np.array(
        [[-0.0145, -0.9980,  0.0616, -0.0309],
        [ 0.5151, -0.0603, -0.8550,  0.3947],
        [ 0.8570,  0.0193,  0.5149, -0.4887],
        [ 0.0000,  0.0000,  0.0000,  1.0000]]
    )
    print(f"{extrinsic=}")
    print(f"{intrinsic=}")

    result, loss_dict = refinement_pipe.refine( 
                                                video=video, 
                                                joint_angles=joint_angles, 
                                                intrinsic=intrinsic, 
                                                extrinsic=extrinsic, 
                                                base_path=save_path,
                                                mask=mask,
                                                mask_id=mask_frame_idx-start_idx,
                                                max_steps=10000
                                                )
    exit()
