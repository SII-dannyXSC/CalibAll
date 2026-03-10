from PIL import Image
import debugpy
import numpy as np
import os
import cv2

# from src.caliball.dataset.lerobot_dataset import LeRobotDataset
from src.caliball.utils.intrinsic_estimator import build_intrinsic_estimator
from src.caliball.dataset.tfds_dataset import TfdsDataset
# debugpy.listen(("0.0.0.0", 10092))
# print("🔍 Waiting for VSCode attach on 0.0.0.0:10092 ...")
# debugpy.wait_for_client()

# dataset = DroidDataset("/cpfs02/user/xiesicheng.xsc/CalibAll/data",split="train[:10]")
# dataset = LeRobotDataset("/cpfs01/user/wenji.zj/dataspace/Data4QwenVLA/RoboMIND_lerobot_v2.1/benchmark1_1_compressed/franka_3rgb/put_the_red_apple_in_the_bowl")

ROOT = "/inspire/hdd/global_user/xiesicheng-253108120120/project/dzj/CalibAll/dataset/tfds/"
NAME = "toto"

dataset = TfdsDataset(
    root_dir=ROOT,
    name=NAME,
    split="train",       # 现在用 train 也不会在 init 爆内存（逐 episode 读）
    max_episodes=3,
    max_steps=200,       # 先小一点验证；跑通再放开/设为 None
    action_key="world_vector"
)

extrinsic = np.array([[ 0.8986,  0.4315,  0.0795, -0.3944],
        [ 0.2157, -0.2767, -0.9364,  0.0679],
        [-0.3821,  0.8586, -0.3417,  1.1462],
        [ 0.0000,  0.0000,  0.0000,  1.0000]])

intrinsic_estimator = build_intrinsic_estimator()

# K = np.array([[572.993 ,  0.     , 320.   ],
#               [  0.     , 572.993, 240.   ],        
#               [  0.     ,  0.     , 1.   ]])

cnt = 0
for data in dataset:
    video = data["video"]    # T H W C
    joint_angles = data["states"]  # T 6
    eef_pose = data['action']
    

    length = len(video)
    
    img_pil = Image.fromarray(video[0])
    _intrinsic, origin_width, origin_height = intrinsic_estimator.estimate(img_pil=img_pil)
    width, height = img_pil.size

    _intrinsic[0, :3] *= 1.0 * width / origin_width
    _intrinsic[1, :3] *= 1.0 * height / origin_height
    K = _intrinsic
    print(f"{K=}")
    
    def p_to_camera(point_3d):
        # 将点扩展为齐次坐标
        pt_w = np.append(point_3d, 1)      # shape (4,)
        pt_c = extrinsic @ pt_w            # 相机坐标系
        pt_c = pt_c[:3]                    # shape (3,)
        return pt_c
    
    def project_point(point_3d, extrinsic, K):
        """
        将世界坐标3d点投影到像素坐标点
        :param point_3d: np.array([3,])
        :param extrinsic: [4,4]
        :param K: 内参 [3,3]
        :return: (u, v)
        """
        # 将点扩展为齐次坐标
        pt_w = np.append(point_3d, 1)      # shape (4,)
        pt_c = extrinsic @ pt_w            # 相机坐标系
        pt_c = pt_c[:3]                    # shape (3,)
        if pt_c[2] == 0:
            pt_c[2] = 1e-5
        px = K @ pt_c
        u = px[0] / px[2]
        v = px[1] / px[2]
        return int(round(u)), int(round(v))


    vis_video = video.copy()
    height, width = vis_video[0].shape[:2]
    output_dir = "./visualized_videos"
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir,f"test_{cnt}.mp4")

    # 重新打开视频写器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 15  # 你可以根据数据实际帧率修改
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    for idx in range(len(vis_video)):
        frame = vis_video[idx]
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) if frame.shape[2] == 3 else frame

        if idx == 0:
            continue

        H = 1   # 固定未来窗口 10 帧
        eps = 1e-4  # 趋势判定阈值（根据你的单位可调）

        # 可视化eef位置
        if eef_pose is not None and len(eef_pose.shape) > 1 and eef_pose.shape[1] >= 3:
            eef_pos = eef_pose[idx-1][:3]
            u, v = project_point(eef_pos, extrinsic, K)
            # eef_pos = p_to_camera(eef_pos)
            cv2.circle(frame_bgr, (u, v), 8, (0,0,255), -1)

            # ===== 未来趋势 (方案B：未来窗口平均速度) =====
            if idx < len(eef_pose) - 1:
                j = min(idx + H, len(eef_pose) - 1)

                start_point_world = eef_pose[idx-1][:3]
                end_point_world = eef_pose[j][:3]
                
                start_point_cam = p_to_camera(start_point_world)
                end_point_cam = p_to_camera(end_point_world)
                
                print(f"{start_point_world=}")
                print(f"{end_point_world=}")
                print(f"{start_point_cam=}")
                print(f"{end_point_cam=}")
                
                delta = end_point_cam - end_point_cam
                dt = max(j - idx, 1)  # 防止除0
                vdir = delta / dt     # 平均“帧位移”

                # 方向映射（带静止判断）
                def axis_label(val, pos, neg):
                    if val > eps:
                        return pos
                    elif val < -eps:
                        return neg
                    else:
                        return "still"

                # x_dir = axis_label(vdir[0], "forward",  "backward")
                # y_dir = axis_label(vdir[1], "left",     "right")
                # z_dir = axis_label(vdir[2], "upward",   "downward")
                x_dir = axis_label(vdir[0], "+",  "-")
                y_dir = axis_label(vdir[1], "+",     "-")
                z_dir = axis_label(vdir[2], "+",   "-")
                txt = f"x:{x_dir} y:{y_dir} z:{z_dir}"
            else:
                txt = "x:? y:? z:?"

            cv2.putText(frame_bgr, txt, (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        writer.write(frame_bgr)
    writer.release()
    print(f"视频已保存到 {out_path} (包含速度方向可视化)")

    cnt += 1
    if cnt > 10:    
        break