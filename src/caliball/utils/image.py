import numpy as np
from PIL import Image
from matplotlib.colors import to_rgb
import torch


def save_imgs(img_list, path = "concat.png"):# 转成 NumPy，并归一化到 [0,255] uint8
    def tensor_to_uint8(img_tensor):
        img_np = img_tensor.numpy()
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min()) * 255
        return img_np.astype(np.uint8)

    img1_np = tensor_to_uint8(img_list[0])
    img2_np = tensor_to_uint8(img_list[1])

    # 横向拼接 axis=1
    concat_np = np.concatenate([img1_np, img2_np], axis=1)

    # 保存
    img = Image.fromarray(concat_np, mode='L')
    img.save(path)
 
def save_img(img, path = "concat.png"):# 转成 NumPy，并归一化到 [0,255] uint8
    def tensor_to_uint8(img_tensor):
        img_np = img_tensor.numpy()
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min()) * 255
        return img_np.astype(np.uint8)

    img1_np = tensor_to_uint8(img)

    # 保存
    img = Image.fromarray(img1_np, mode='L')
    img.save(path)
    
def add_mask(cur_rgb, cur_mask, color = [220, 172, 47], alpha = 0.7):    #np.array
    # --- 步骤 1: 将灰度 Mask 转换为布尔索引 ---
    # 我们通常假设 Mask 中大于 0 的像素是前景区域 (目标物体)
    # 这一步创建了一个布尔数组，用于选择前景像素
    mask_indices = (cur_mask > 0) # 形状: (H, W), dtype: bool

    # --- 步骤 2: 创建彩色层 (Overlay Color) ---
    # 选择你想要的叠加颜色 (这里使用亮蓝色 [R, G, B])
    # 你可以调整 (0, 0, 255) 来改变颜色，例如 (255, 0, 0) 是红色
    # overlay_color_rgb = [47, 172, 220] # 亮蓝色 (RGB 格式)
    overlay_color_rgb = color 

    color_layer = np.zeros_like(cur_rgb, dtype=np.uint8) 
    # 只在 Mask 区域填充颜色
    color_layer[mask_indices] = overlay_color_rgb[mask_indices]

    # --- 步骤 3: 执行加权平均叠加 (Alpha Blending) ---
    # alpha = 0.7 # 透明度 (0.0 - 1.0)

    # 1. 转换为浮点数进行精确计算
    current_pixels_float = cur_rgb.astype(np.float32)
    color_pixels_float = color_layer.astype(np.float32)

    # 2. 拷贝原始图像用于叠加，并设置为浮点数
    overlay = current_pixels_float.copy() 

    # 3. 仅对 Mask 区域的像素执行加权平均
    # 使用 NumPy 广播机制，[mask_indices] 选择 (N, 3) 形状的像素集
    overlay[mask_indices] = (current_pixels_float[mask_indices] * (1 - alpha) + 
                            color_pixels_float[mask_indices] * alpha)

    # 4. 将最终结果转换回 uint8
    final_overlay = overlay.astype(np.uint8)
    return final_overlay

def save_mask_as_image(mask, color="r", save_path="mask.png"):
    """保存单独的 mask 为图片（带透明度的 RGBA 格式）"""
    # 如果是 torch tensor，转换为 numpy
    if torch.is_tensor(mask):
        mask = mask.detach().cpu().numpy()
    
    # 处理可能的 3D mask (如 (1, H, W) 或 (H, W, 1))
    if mask.ndim == 3:
        if mask.shape[0] == 1:
            mask = mask[0]  # (1, H, W) -> (H, W)
        elif mask.shape[-1] == 1:
            mask = mask[..., 0]  # (H, W, 1) -> (H, W)
        else:
            # 如果是多通道，取第一个通道
            mask = mask[..., 0] if mask.shape[-1] < mask.shape[0] else mask[0]
    
    im_h, im_w = mask.shape
    mask_img = np.zeros((im_h, im_w, 4), dtype=np.float32)
    mask_img[..., :3] = to_rgb(color)
    mask_img[..., 3] = mask * 0.5  # 透明度
    
    # 转换为 uint8 并保存
    mask_img_uint8 = (mask_img * 255).astype(np.uint8)
    Image.fromarray(mask_img_uint8, mode='RGBA').save(save_path)
    print(f"Mask saved to: {save_path}")


def save_image_with_mask(image, mask, color="r", alpha=0.5, save_path="image_with_mask.png"):
    """
    保存带有 mask 叠加的原图
    
    Args:
        image: PIL Image 或 numpy array (H, W, 3)
        mask: numpy array (H, W), 值为 0 或 1
        color: mask 颜色
        alpha: mask 透明度，0-1 之间
        save_path: 保存路径
    """
    # 如果是 torch tensor，转换为 numpy
    if torch.is_tensor(mask):
        mask = mask.detach().cpu().numpy()
    
    # 处理可能的 3D mask
    if mask.ndim == 3:
        if mask.shape[0] == 1:
            mask = mask[0]
        elif mask.shape[-1] == 1:
            mask = mask[..., 0]
        else:
            mask = mask[..., 0] if mask.shape[-1] < mask.shape[0] else mask[0]
    
    # 转换 image 为 numpy array
    if isinstance(image, Image.Image):
        img_array = np.array(image)
    else:
        img_array = image.copy()
    
    # 确保是 uint8 格式
    if img_array.dtype != np.uint8:
        if img_array.max() <= 1.0:
            img_array = (img_array * 255).astype(np.uint8)
        else:
            img_array = img_array.astype(np.uint8)
    
    # 创建 mask 颜色层
    im_h, im_w = mask.shape
    mask_color = np.array(to_rgb(color)) * 255  # RGB 值
    
    # 在原图上叠加 mask
    result = img_array.copy()
    for c in range(3):
        result[:, :, c] = np.where(
            mask > 0,
            img_array[:, :, c] * (1 - alpha) + mask_color[c] * alpha,
            img_array[:, :, c]
        )
    
    # 保存
    Image.fromarray(result.astype(np.uint8)).save(save_path)
    print(f"Image with mask saved to: {save_path}")
