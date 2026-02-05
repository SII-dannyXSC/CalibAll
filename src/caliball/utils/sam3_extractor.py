import torch
import numpy as np
import cv2
import os
from typing import Optional, List, Dict
from PIL import Image
from matplotlib.colors import to_rgb

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

from src.caliball.utils.image import save_mask_as_image, save_image_with_mask

class Sam3Extractor:
    def __init__(self, bpe_path, ckpt_path):
        self.bpe_path = bpe_path
        self.ckpt_path = ckpt_path
        self.model = build_sam3_image_model(bpe_path=self.bpe_path, checkpoint_path=self.ckpt_path)
        self.processor = Sam3Processor(self.model)

    def extract_masks(self, img_pil, prompt="robotic arm"):
        if isinstance(img_pil, str):
            img_pil = Image.open(img_pil)
        elif isinstance(img_pil, np.ndarray):
            img_pil = Image.fromarray(img_pil)
        else:
            assert isinstance(img_pil, Image.Image)
        inference_state = self.processor.set_image(img_pil)
        output = self.processor.set_text_prompt(state=inference_state, prompt=prompt)
        masks, boxes, scores = output["masks"], output["boxes"], output["scores"]
        
        if len(masks) > 0:
            best_mask = masks[0]
            return best_mask
        else:
            return None














if __name__ == "__main__":
    # 初始化模型
    bpe_path = f"/cpfs02/user/xiesicheng.xsc/CalibAll/third_party/sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz"
    ckpt_path = f"/cpfs02/user/xiesicheng.xsc/CalibAll/ckpt/sam3/sam3.pt"
    model = build_sam3_image_model(bpe_path=bpe_path, checkpoint_path=ckpt_path)
    processor = Sam3Processor(model)

    # 加载图像
    image_path = "/cpfs02/user/xiesicheng.xsc/CalibAll/test.png"
    image = Image.open(image_path)
    inference_state = processor.set_image(image)

    # 使用文本提示
    output = processor.set_text_prompt(state=inference_state, prompt="robotic arm")
    masks, boxes, scores = output["masks"], output["boxes"], output["scores"]

    # 保存结果（使用第一个 mask，通常是得分最高的）
    if len(masks) > 0:
        best_mask = masks[0]  # 选择第一个 mask
        
        print(f"Total masks found: {len(masks)}")
        print(f"Mask type: {type(best_mask)}")
        print(f"Mask shape: {best_mask.shape}")
        print(f"Best mask score: {scores[0] if len(scores) > 0 else 'N/A'}")
        
        # 保存单独的 mask
        save_mask_as_image(best_mask, color="r", save_path="robot_mask.png")
        
        # 保存带 mask 的原图
        save_image_with_mask(image, best_mask, color="r", alpha=0.5, save_path="robot_with_mask.png")
    else:
        print("No masks found!")
