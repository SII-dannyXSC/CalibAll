import os
import cv2
import numpy as np
import torch
from typing import List, Dict, Tuple, Optional
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator



class MaskExtractor:
    """使用 SAM 2 (Segment Anything Model 2) 进行图像分割"""
    
    def __init__(
        self, 
        model_cfg: str = "sam2_hiera_l.yaml",
        checkpoint: str = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        初始化 SAM 2 模型
        
        Args:
            model_cfg: 模型配置文件名 (sam2_hiera_t.yaml, sam2_hiera_s.yaml, sam2_hiera_b.yaml, sam2_hiera_l.yaml)
            checkpoint: 模型权重路径，如果为 None 则自动下载
            device: 运行设备 ("cuda" 或 "cpu")
        """
        self.device = device
        
        # 构建 SAM 2 模型
        if checkpoint is None:
            # 自动下载模型
            checkpoint = self._download_checkpoint(model_cfg)
        
        self.model = build_sam2(model_cfg, checkpoint, device=device)
        
        # 创建自动 mask 生成器
        self.mask_generator = SAM2AutomaticMaskGenerator(
            model=self.model,
            points_per_side=32,
            pred_iou_thresh=0.7,
            stability_score_thresh=0.92,
            crop_n_layers=1,
            crop_n_points_downscale_factor=2,
            min_mask_region_area=100,
        )
    
    def _download_checkpoint(self, model_cfg: str) -> str:
        """下载对应的模型权重"""
        # 模型映射
        checkpoint_map = {
            "sam2_hiera_t.yaml": "sam2_hiera_tiny.pt",
            "sam2_hiera_s.yaml": "sam2_hiera_small.pt",
            "sam2_hiera_b.yaml": "sam2_hiera_base_plus.pt",
            "sam2_hiera_l.yaml": "sam2_hiera_large.pt",
        }
        
        checkpoint_name = checkpoint_map.get(model_cfg, "sam2_hiera_large.pt")
        
        # 下载路径 (通常会自动缓存到 ~/.cache/torch/hub/checkpoints/)
        from sam2.utils.misc import download_checkpoint
        checkpoint_path = download_checkpoint(checkpoint_name)
        
        return checkpoint_path
    
    def extract_masks(
        self, 
        image_path: str,
        min_area: Optional[int] = None,
        max_area: Optional[int] = None
    ) -> List[Dict]:
        """
        从图片中提取所有分割 masks
        
        Args:
            image_path: 图片路径
            min_area: 最小 mask 面积阈值
            max_area: 最大 mask 面积阈值
            
        Returns:
            masks: List of dict, 每个 dict 包含:
                - segmentation: (H, W) bool array, mask 二值图
                - area: int, mask 面积
                - bbox: [x, y, w, h], bounding box
                - predicted_iou: float, 预测的 IoU 分数
                - stability_score: float, 稳定性分数
        """
        # 读取图片
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法读取图片: {image_path}")
        
        # BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 生成 masks
        masks = self.mask_generator.generate(image_rgb)
        
        # 过滤 masks
        if min_area is not None or max_area is not None:
            filtered_masks = []
            for mask in masks:
                area = mask['area']
                if min_area is not None and area < min_area:
                    continue
                if max_area is not None and area > max_area:
                    continue
                filtered_masks.append(mask)
            masks = filtered_masks
        
        return masks
    
    def visualize_masks(
        self, 
        image_path: str, 
        masks: List[Dict],
        output_path: Optional[str] = None,
        alpha: float = 0.5
    ) -> np.ndarray:
        """
        可视化分割结果
        
        Args:
            image_path: 原始图片路径
            masks: extract_masks 返回的 masks
            output_path: 输出图片路径，如果为 None 则不保存
            alpha: mask 透明度
            
        Returns:
            vis_image: 可视化结果图片
        """
        # 读取图片
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法读取图片: {image_path}")
        
        # 按面积排序（大的在后面，这样小的会覆盖在大的上面）
        sorted_masks = sorted(masks, key=lambda x: x['area'], reverse=True)
        
        # 创建可视化图片
        vis_image = image.copy()
        
        # 为每个 mask 生成随机颜色
        for mask in sorted_masks:
            color = np.random.randint(0, 255, size=3, dtype=np.uint8)
            mask_bool = mask['segmentation']
            
            # 应用 mask
            vis_image[mask_bool] = vis_image[mask_bool] * (1 - alpha) + color * alpha
            
            # 绘制边界框
            x, y, w, h = mask['bbox']
            cv2.rectangle(vis_image, (int(x), int(y)), (int(x+w), int(y+h)), 
                         color.tolist(), 2)
        
        # 保存结果
        if output_path is not None:
            cv2.imwrite(output_path, vis_image)
            print(f"可视化结果已保存到: {output_path}")
        
        return vis_image
    
    def extract_and_save_individual_masks(
        self,
        image_path: str,
        output_dir: str,
        min_area: Optional[int] = None
    ) -> List[str]:
        """
        提取 masks 并单独保存每个 mask
        
        Args:
            image_path: 图片路径
            output_dir: 输出目录
            min_area: 最小面积阈值
            
        Returns:
            mask_paths: 保存的 mask 文件路径列表
        """
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 提取 masks
        masks = self.extract_masks(image_path, min_area=min_area)
        
        # 保存每个 mask
        mask_paths = []
        for i, mask in enumerate(masks):
            mask_bool = mask['segmentation']
            mask_uint8 = (mask_bool * 255).astype(np.uint8)
            
            # 保存路径
            mask_path = os.path.join(output_dir, f"mask_{i:04d}.png")
            cv2.imwrite(mask_path, mask_uint8)
            mask_paths.append(mask_path)
        
        print(f"已保存 {len(mask_paths)} 个 masks 到: {output_dir}")
        return mask_paths


def main():
    """示例用法"""
    # 创建 mask 提取器
    extractor = MaskExtractor(
        model_cfg="sam2_hiera_l.yaml",  # 可选: tiny, small, base, large
        device="cuda"
    )
    
    # 图片路径
    image_path = "path/to/your/image.jpg"
    
    # 提取 masks
    print("正在提取 masks...")
    masks = extractor.extract_masks(
        image_path,
        min_area=100  # 最小面积阈值
    )
    print(f"找到 {len(masks)} 个分割区域")
    
    # 可视化结果
    print("可视化结果...")
    extractor.visualize_masks(
        image_path,
        masks,
        output_path="segmentation_result.jpg"
    )
    
    # 保存单独的 masks
    print("保存单独的 masks...")
    extractor.extract_and_save_individual_masks(
        image_path,
        output_dir="masks_output",
        min_area=100
    )
    
    print("完成！")


if __name__ == "__main__":
    main()
