from PIL import Image
import debugpy
import numpy as np
import os

from caliball.dataset.lerobot_dataset import LeRobotDataset
# debugpy.listen(("0.0.0.0", 10092))
# print("🔍 Waiting for VSCode attach on 0.0.0.0:10092 ...")
# debugpy.wait_for_client()

# dataset = DroidDataset("/cpfs02/user/xiesicheng.xsc/CalibAll/data",split="train[:10]")
dataset = LeRobotDataset("/cpfs01/user/wenji.zj/dataspace/Data4QwenVLA/RoboMIND_lerobot_v2.1/benchmark1_1_compressed/franka_3rgb/put_the_red_apple_in_the_bowl")

extrinsic = np.array([[ 0.5394, -0.8300, -0.1421, -0.4130],
        [-0.5207, -0.1961, -0.8309,  0.6093],
        [ 0.6618,  0.5222, -0.5379,  0.7001],
        [ 0.0000,  0.0000,  0.0000,  1.0000]])

K = np.array([[572.993 ,  0.     , 320.   ],
              [  0.     , 572.993, 240.   ],        
              [  0.     ,  0.     , 1.   ]])

cnt = 0
for data in dataset:
    video = data["video"]    # T H W C
    joint_angles = data["states"]  # T 6
    eef_pose = data['actions']

    length = len(video)
    
    # 将末端执行器（eef_pose）的3D位置通过extrinsic和K投影到2D图像上，并显示在视频帧上

    import cv2

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
    output_dir = "/cpfs02/user/xiesicheng.xsc/CalibAll/visualized_videos"
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir,f"test_{cnt}.mp4")

    # 保存末端执行器位置到一个list
    eef_pos_list = []
    if eef_pose is not None and len(eef_pose.shape) > 1 and eef_pose.shape[1] >= 3:
        for idx in range(len(vis_video)):
            eef_pos = eef_pose[idx][:3]   # x, y, z (世界坐标)
            eef_pos_list.append(eef_pos)

        # 计算每个时刻的速度方向
        eef_pos_arr = np.array(eef_pos_list)  # (T, 3)
        velocities = np.diff(eef_pos_arr, axis=0)  # (T-1, 3)
        directions = np.sign(velocities) # 方向（每个分量）
        # 你也可以取整体方向，例如单位向量：
        norm = np.linalg.norm(velocities, axis=1, keepdims=True)
        velocity_dirs = velocities / (norm + 1e-8)
        # velocity_dirs现在是(T-1, 3)，表示每帧的速度方向（单位向量）

    # 把速度方向（x,y,z的正负）写在每一帧上

    # 重新打开视频写器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 15  # 你可以根据数据实际帧率修改
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    for idx in range(len(vis_video)):
        frame = vis_video[idx]
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) if frame.shape[2] == 3 else frame

        # 可视化eef位置
        if eef_pose is not None and len(eef_pose.shape) > 1 and eef_pose.shape[1] >= 3:
            eef_pos = eef_pose[idx][:3]
            u, v = project_point(eef_pos, extrinsic, K)
            cv2.circle(frame_bgr, (u, v), 8, (0,0,255), -1)

            # 速度方向可视化（第0帧无velocity）
            if idx > 0:
                vdir = velocity_dirs[idx-1]
                # 按要求映射方向
                x_dir = "forward" if vdir[0] > 0 else "backward"
                y_dir = "left" if vdir[1] > 0 else "right"
                z_dir = "upward" if vdir[2] > 0 else "downward"
                txt = f"x:{x_dir} y:{y_dir} z:{z_dir}"
                # 写在图像左上角
                cv2.putText(frame_bgr, txt, (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            else:
                # 第一帧速度未知
                cv2.putText(frame_bgr, "x:? y:? z:?", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        writer.write(frame_bgr)

    writer.release()
    print(f"视频已保存到 {out_path} (包含速度方向可视化)")

    cnt += 1
    if cnt > 10:    
        break