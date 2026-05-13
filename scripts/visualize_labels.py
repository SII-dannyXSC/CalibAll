"""
visualize_labels.py

读取标注 JSON 文件 + 原始视频帧，将以下信息叠加到图像上并输出视频：
  · 2D EEF 点      (红色圆点 + 标签)
  · 2D Gripper 点  (蓝色圆点 + 标签)
  · Mask           (半透明彩色蒙版，不含/含 gripper 两版)
  · BBox           (矩形框，不含/含 gripper 两版)

用法：
    python scripts/visualize_labels.py \
        --json_path  label_result/put_the_red_apple_in_the_bowl/episode_000000.json \
        --task_path  /path/to/franka_3rgb/put_the_red_apple_in_the_bowl \
        --output_dir label_result/vis/episode_000000 \
        --cameras    camera_left

Berkeley / OXE 单相机（JSON 顶层键常为 ``image``，可省略 ``--cameras`` 由脚本从 JSON 推断）：
    python scripts/visualize_labels.py \
        --json_path  label_out/berkeley_autolab_ur5/episode_000000.json \
        --task_path  /path/to/lerobot_2.1/berkeley_autolab_ur5 \
        --output_dir label_out/berkeley_autolab_ur5/vis \
        --first_frame_only

批量（对某个 task 的所有 episode）：
    python scripts/visualize_labels.py \
        --json_dir   label_result/put_the_red_apple_in_the_bowl \
        --task_path  /path/to/franka_3rgb/put_the_red_apple_in_the_bowl \
        --output_dir label_result/vis \
        --cameras    camera_left
"""

import argparse
import glob
import json
import os
import subprocess
import sys

import cv2
import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _get_ffmpeg_exe() -> str:
    """优先使用 imageio-ffmpeg 自带二进制（含 libx264），回退到系统 ffmpeg。"""
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return "ffmpeg"


_FFMPEG = _get_ffmpeg_exe()
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


C_EEF = (0, 0, 255)
C_GRIP = (255, 0, 0)
C_MASK_ALL = (0, 255, 128)
C_MASK_ARM = (255, 255, 0)
C_MASK_GRIP = (0, 180, 255)
C_BBOX_ALL = (0, 255, 0)       # 绿：arm + gripper
C_BBOX_ARM = (255, 200, 0)     # 黄：arm only
C_BBOX_GRIP = (0, 128, 255)    # 蓝：gripper only

# 双臂右臂用不同颜色
_ARM_PALETTE = [
    # left / single
    dict(mask_arm=C_MASK_ARM, mask_grip=C_MASK_GRIP,
         bbox_all=C_BBOX_ALL, bbox_arm=C_BBOX_ARM, bbox_grip=C_BBOX_GRIP,
         grip=C_GRIP),
    # right
    dict(mask_arm=(0, 255, 180), mask_grip=(200, 100, 255),
         bbox_all=(0, 180, 80), bbox_arm=(100, 200, 80), bbox_grip=(180, 80, 255),
         grip=(0, 80, 255)),
]


def decode_mask(rle, shape):
    """将 label.py 输出的 mask RLE 解码为 (H, W) uint8 二值图。"""
    if rle is None:
        return None

    mask = None

    if isinstance(rle, dict) and rle.get("format") == "simple_rle":
        rle_shape = tuple(rle["size"])
        arr = np.zeros(rle_shape[0] * rle_shape[1], dtype=np.uint8)
        idx = 0
        for val, cnt in rle["counts"]:
            arr[idx: idx + cnt] = val
            idx += cnt
        mask = arr.reshape(rle_shape)
    else:
        try:
            from pycocotools import mask as coco_mask
            rle_bytes = dict(rle)
            if isinstance(rle_bytes["counts"], str):
                rle_bytes["counts"] = rle_bytes["counts"].encode("utf-8")
            mask = coco_mask.decode(rle_bytes).astype(np.uint8)
        except ImportError:
            return None

    if mask is None:
        return None

    tH, tW = shape
    if mask.shape != (tH, tW):
        mask = cv2.resize(mask, (tW, tH), interpolation=cv2.INTER_NEAREST)

    return mask


def overlay_mask(img_bgr, mask, color_bgr, alpha=0.45):
    if mask is None or not np.any(mask):
        return img_bgr
    overlay = img_bgr.copy()
    overlay[mask > 0] = color_bgr
    return cv2.addWeighted(overlay, alpha, img_bgr, 1 - alpha, 0)


def draw_bbox(img_bgr, bbox, color_bgr, label="", thickness=2):
    if bbox is None:
        return img_bgr
    x1, y1, x2, y2 = [int(v) for v in bbox]
    cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color_bgr, thickness)
    if label:
        cv2.putText(img_bgr, label, (x1, max(y1 - 4, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color_bgr, 1, cv2.LINE_AA)
    return img_bgr


def draw_point(img_bgr, uv, color_bgr, label="", radius=6, thickness=-1):
    if uv is None:
        return img_bgr
    u, v = int(uv[0]), int(uv[1])
    H, W = img_bgr.shape[:2]
    if not (0 <= u < W and 0 <= v < H):
        return img_bgr
    cv2.circle(img_bgr, (u, v), radius, color_bgr, thickness)
    cv2.circle(img_bgr, (u, v), radius, (255, 255, 255), 1)
    if label:
        cv2.putText(img_bgr, label, (u + 8, v - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color_bgr, 1, cv2.LINE_AA)
    return img_bgr


def draw_axes(img_bgr, K, xyz_cam, rot_mat_flat, scale=0.15, thickness=6):
    """
    在图像上绘制末端执行器的三个旋转轴（X=红, Y=绿, Z=蓝）。

    xyz_cam 和 rot_mat_flat 均已在相机坐标系中（即 xyz_mat_g 的前 12 个元素），
    直接用内参 K 投影，不需要外参。

    K:            (3,3) 相机内参
    xyz_cam:      (3,) 末端位置（相机坐标系，单位 m）
    rot_mat_flat: (9,) 旋转矩阵按行展平 [r00,r01,r02, r10,r11,r12, r20,r21,r22]
    scale:        轴长度（米）
    """
    try:
        K = np.array(K, dtype=np.float64)
        origin = np.array(xyz_cam, dtype=np.float64)
        R = np.array(rot_mat_flat, dtype=np.float64).reshape(3, 3)

        # 4 points in camera frame: origin + 3 axis tips
        pts = np.vstack([
            origin,
            origin + scale * R[:, 0],   # X body axis
            origin + scale * R[:, 1],   # Y body axis
            origin + scale * R[:, 2],   # Z body axis
        ])  # (4, 3)

        H, W = img_bgr.shape[:2]

        def project(p):
            if p[2] <= 1e-4:
                return None
            uv = K @ p
            u, v = int(round(uv[0] / uv[2])), int(round(uv[1] / uv[2]))
            return (u, v) if (0 <= u < W and 0 <= v < H) else None

        o = project(pts[0])
        if o is None:
            return img_bgr

        colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]  # X=red, Y=green, Z=blue (BGR)
        axis_labels = ["X", "Y", "Z"]
        for i, (color, lbl) in enumerate(zip(colors, axis_labels)):
            tip = project(pts[i + 1])
            if tip is None:
                continue
            cv2.arrowedLine(img_bgr, o, tip, color, thickness, tipLength=0.15, line_type=cv2.LINE_AA)
            cv2.putText(img_bgr, lbl, (tip[0] + 10, tip[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)
    except Exception:
        pass
    return img_bgr


def draw_gripper_state(img_bgr, gripper_val, pos=(10, 20)):
    if gripper_val is None:
        return img_bgr
    state = f"gripper: {gripper_val:.2f}"
    cv2.putText(img_bgr, state, pos, cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (200, 200, 200), 1, cv2.LINE_AA)
    return img_bgr


class FfmpegVideoReader:
    def __init__(self, path: str):
        import av as _av
        self._container = _av.open(path)
        self._stream = self._container.streams.video[0]
        self.width = self._stream.width
        self.height = self._stream.height
        self.n_frames = self._stream.frames if self._stream.frames else -1
        self._iter = self._container.decode(self._stream)

    def read(self):
        try:
            frame = next(self._iter)
            bgr = frame.to_ndarray(format="bgr24")
            return True, bgr
        except StopIteration:
            return False, None

    def release(self):
        self._container.close()


class FfmpegVideoWriter:
    def __init__(self, path, fps, width, height):
        self._proc = subprocess.Popen(
            [_FFMPEG, "-y",
             "-f", "rawvideo", "-vcodec", "rawvideo",
             "-pix_fmt", "bgr24", "-s", f"{width}x{height}",
             "-r", str(fps), "-i", "pipe:0",
             "-vcodec", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
             path],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def write(self, frame):
        self._proc.stdin.write(frame.tobytes())

    def release(self):
        if self._proc.stdin:
            self._proc.stdin.close()
        self._proc.wait()


def _render_frame(bgr, frame, H, W,
                  show_mask, show_bbox, show_eef, show_grip, mask_alpha,
                  K=None, show_axes=True, axes_scale=0.05, **_):
    mask_shape = (H, W)

    # 收集所有非 placeholder 的臂（保留顺序，最多取前 2 臂颜色）
    active_arms = [(name, a) for name, a in frame.get("arms", {}).items()
                   if not a.get("is_placeholder", False)]

    full     = bgr.copy()
    mask_vis = bgr.copy()
    bbox_vis = bgr.copy()
    pts_vis  = bgr.copy()

    for arm_i, (arm_name, arm) in enumerate(active_arms):
        pal = _ARM_PALETTE[min(arm_i, len(_ARM_PALETTE) - 1)]
        tag = arm_name[0].upper()  # "L" or "R" or first char of arm name

        mask_arm  = decode_mask(arm.get("mask_without_gripper"), mask_shape)
        mask_grip = decode_mask(arm.get("mask_gripper"),         mask_shape)
        grip_uv   = arm.get("uv")
        bbox_all  = arm.get("bbox_with_gripper")
        bbox_arm  = arm.get("bbox_without_gripper")
        bbox_grip = arm.get("bbox_gripper")
        xyz_euler_g = arm.get("xyz_euler_g")
        gripper_val = float(xyz_euler_g[-1]) if xyz_euler_g else None

        if show_mask:
            full = overlay_mask(full, mask_arm,  pal["mask_arm"],  alpha=mask_alpha)
            full = overlay_mask(full, mask_grip, pal["mask_grip"], alpha=mask_alpha * 0.6)
        if show_bbox:
            full = draw_bbox(full, bbox_all,  pal["bbox_all"],  f"{tag}:robot")
            full = draw_bbox(full, bbox_arm,  pal["bbox_arm"],  f"{tag}:arm")
            full = draw_bbox(full, bbox_grip, pal["bbox_grip"], f"{tag}:grip")
        if show_grip:
            full = draw_point(full, grip_uv, pal["grip"], tag)
        full = draw_gripper_state(full, gripper_val,
                                  pos=(10, 20 + arm_i * 18))

        mask_vis = overlay_mask(mask_vis, mask_arm,  pal["mask_arm"],  alpha=mask_alpha)
        mask_vis = overlay_mask(mask_vis, mask_grip, pal["mask_grip"], alpha=mask_alpha * 0.6)

        bbox_vis = draw_bbox(bbox_vis, bbox_all,  pal["bbox_all"],  f"{tag}:robot")
        bbox_vis = draw_bbox(bbox_vis, bbox_arm,  pal["bbox_arm"],  f"{tag}:arm")
        bbox_vis = draw_bbox(bbox_vis, bbox_grip, pal["bbox_grip"], f"{tag}:grip")

        pts_vis = draw_point(pts_vis, grip_uv, pal["grip"], tag)
        pts_vis = draw_gripper_state(pts_vis, gripper_val,
                                     pos=(10, 20 + arm_i * 18))

        # ── rotation axes ─────────────────────────────────────────────────────
        if show_axes and K is not None:
            xyz_mat_g = arm.get("xyz_mat_g")
            if xyz_mat_g and not arm.get("is_placeholder", False):
                xyz_cam = xyz_mat_g[:3]
                rot_flat = xyz_mat_g[3:12]
                draw_axes(full,    K, xyz_cam, rot_flat, scale=axes_scale)
                draw_axes(pts_vis, K, xyz_cam, rot_flat, scale=axes_scale)

    return full, mask_vis, bbox_vis, pts_vis


def _camera_keys_from_label_json(label_data: dict):
    """除 ``video_seg`` 外，含 ``frames`` 子字段的顶层键视为相机名。"""
    keys = []
    for k, v in label_data.items():
        if k == "video_seg":
            continue
        if isinstance(v, dict) and "frames" in v:
            keys.append(k)
    return keys


def visualize_episode_all_cameras(json_path, task_path, output_dir,
                                  camera_names, fps=15,
                                  show_mask=True, show_bbox=True,
                                  show_eef=True, show_grip=True,
                                  mask_alpha=0.45,
                                  show_axes=True, axes_scale=0.05,
                                  first_frame_only=False):
    with open(json_path, "r", encoding="utf-8") as f:
        label_data = json.load(f)

    if not camera_names:
        camera_names = _camera_keys_from_label_json(label_data)
        if not camera_names:
            print("[ERROR] 未指定 --cameras，且无法从 JSON 推断相机键")
            return
        print(f"[INFO] 未指定 --cameras，使用 JSON 中的相机: {camera_names}")

    ep_idx = int(os.path.splitext(os.path.basename(json_path))[0].split("_")[-1])
    ep_tag = f"ep{ep_idx:06d}"
    os.makedirs(output_dir, exist_ok=True)

    for cam_name in camera_names:
        if cam_name not in label_data:
            print(f"[SKIP] JSON 中不含相机 {cam_name}")
            continue

        cam_data = label_data[cam_name]
        K_mat   = cam_data.get("intrinsic") if isinstance(cam_data, dict) else None
        T_ext   = cam_data.get("extrinsic") if isinstance(cam_data, dict) else None

        cam_key = f"{cam_name}"
        vid_pattern = os.path.join(task_path, "videos", "chunk-*", cam_key, f"episode_{ep_idx:06d}.mp4")
        matches = sorted(glob.glob(vid_pattern))
        if not matches:
            print(f"[SKIP] 找不到视频: {vid_pattern}")
            continue
        video_path = matches[0]

        info_list = cam_data.get("frames", []) if isinstance(cam_data, dict) else cam_data
        tag = f"{ep_tag}_{cam_name}"

        try:
            reader = FfmpegVideoReader(video_path)
        except Exception as e:
            print(f"[WARN] 无法打开视频 {video_path}: {e}")
            continue

        H, W = reader.height, reader.width

        render_kwargs = dict(
            show_mask=show_mask, show_bbox=show_bbox,
            show_eef=show_eef, show_grip=show_grip, mask_alpha=mask_alpha,
            K=K_mat, T_ext=T_ext, show_axes=show_axes, axes_scale=axes_scale,
        )

        if first_frame_only:
            ok, bgr = reader.read()
            reader.release()
            if not ok:
                print(f"[WARN] 无法读取第一帧: {video_path}")
                continue
            info = info_list[0] if info_list else {}
            full, mask_vis, bbox_vis, pts_vis = _render_frame(
                bgr, info, H, W, **render_kwargs)
            cv2.imwrite(os.path.join(output_dir, f"{tag}_full.jpg"), full)
            cv2.imwrite(os.path.join(output_dir, f"{tag}_mask.jpg"), mask_vis)
            cv2.imwrite(os.path.join(output_dir, f"{tag}_bbox.jpg"), bbox_vis)
            cv2.imwrite(os.path.join(output_dir, f"{tag}_points.jpg"), pts_vis)
            print(f"[OK] episode {ep_idx} / {cam_name}  →  {output_dir}/")
            print(f"     {tag}_{{full,mask,bbox,points}}.jpg  [first-frame-only]")
        else:
            T = len(info_list) if reader.n_frames < 0 else min(reader.n_frames, len(info_list))
            writers = {
                "full":   FfmpegVideoWriter(os.path.join(output_dir, f"{tag}_full.mp4"),   fps, W, H),
                "mask":   FfmpegVideoWriter(os.path.join(output_dir, f"{tag}_mask.mp4"),   fps, W, H),
                "bbox":   FfmpegVideoWriter(os.path.join(output_dir, f"{tag}_bbox.mp4"),   fps, W, H),
                "points": FfmpegVideoWriter(os.path.join(output_dir, f"{tag}_points.mp4"), fps, W, H),
            }
            i = 0
            while i < T:
                ok, bgr = reader.read()
                if not ok:
                    break
                full, mask_vis, bbox_vis, pts_vis = _render_frame(
                    bgr, info_list[i], H, W, **render_kwargs)
                writers["full"].write(full)
                writers["mask"].write(mask_vis)
                writers["bbox"].write(bbox_vis)
                writers["points"].write(pts_vis)
                i += 1
            reader.release()
            for w in writers.values():
                w.release()
            print(f"[OK] episode {ep_idx} / {cam_name}  →  {output_dir}/")
            print(f"     {tag}_{{full,mask,bbox,points}}.mp4")


def parse_args():
    parser = argparse.ArgumentParser(description="将标注 JSON 可视化为视频（支持多相机）")
    parser.add_argument("--json_path", type=str, default=None,
                        help="单个 episode JSON 文件路径")
    parser.add_argument("--json_dir", type=str, default=None,
                        help="包含多个 episode_*.json 的目录（批量处理）")
    parser.add_argument("--task_path", type=str, required=True,
                        help="LeRobot 数据集任务目录（用于读取原始视频帧）")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="可视化视频输出目录")
    parser.add_argument(
        "--cameras",
        nargs="*",
        default=None,
        help="逻辑相机名（对应 JSON 顶层键与 videos/.../observation.images.<name>/）。"
        " 省略则从 JSON 自动推断（排除 video_seg）。默认（在未省略时）为 RoboMIND 三视角。",
    )
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--no_mask", action="store_true")
    parser.add_argument("--no_bbox", action="store_true")
    parser.add_argument("--no_eef", action="store_true")
    parser.add_argument("--no_grip", action="store_true")
    parser.add_argument("--no_axes", action="store_true",
                        help="不绘制旋转坐标轴（默认绘制）")
    parser.add_argument("--axes_scale", type=float, default=0.05,
                        help="旋转轴长度（单位：米，默认 0.05）")
    parser.add_argument("--alpha", type=float, default=0.45, help="mask 不透明度 (0~1)")
    parser.add_argument("--first_frame_only", action="store_true",
                        help="只保存第一帧为 JPG，不生成完整视频")
    parser.add_argument("--episode", type=int, default=None,
                        help="仅处理指定索引的 episode")
    return parser.parse_args()


def main():
    args = parse_args()

    # nargs='*'：未写 --cameras → None；``--cameras`` 无参数 → []；二者均在 visualize 内从 JSON 推断
    cams = args.cameras
    if cams is not None and len(cams) == 0:
        cams = None

    kwargs = dict(
        task_path=args.task_path,
        camera_names=cams,
        fps=args.fps,
        show_mask=not args.no_mask,
        show_bbox=not args.no_bbox,
        show_eef=not args.no_eef,
        show_grip=not args.no_grip,
        mask_alpha=args.alpha,
        show_axes=not args.no_axes,
        axes_scale=args.axes_scale,
        first_frame_only=args.first_frame_only,
    )

    if args.json_path:
        ep_name = os.path.splitext(os.path.basename(args.json_path))[0]
        out = os.path.join(args.output_dir, ep_name)
        visualize_episode_all_cameras(json_path=args.json_path, output_dir=out, **kwargs)

    elif args.json_dir:
        json_files = sorted([
            f for f in os.listdir(args.json_dir)
            if f.startswith("episode_") and f.endswith(".json")
        ])
        if not json_files:
            print(f"[ERROR] {args.json_dir} 中没有 episode_*.json 文件")
            sys.exit(1)
        if args.episode is not None:
            target = f"episode_{args.episode:06d}.json"
            json_files = [f for f in json_files if f == target]
            if not json_files:
                print(f"[ERROR] 未找到 {target}")
                sys.exit(1)
            print(f"[INFO] 仅处理 episode {args.episode}: {target}")
        else:
            print(f"[INFO] 找到 {len(json_files)} 个 episode，开始可视化 ...")
        for fname in json_files:
            ep_name = os.path.splitext(fname)[0]
            json_path = os.path.join(args.json_dir, fname)
            out = os.path.join(args.output_dir, ep_name)
            visualize_episode_all_cameras(json_path=json_path, output_dir=out, **kwargs)

    else:
        print("[ERROR] 请指定 --json_path 或 --json_dir")
        sys.exit(1)

    print("[DONE]")


if __name__ == "__main__":
    main()
