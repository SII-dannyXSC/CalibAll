"""SAM3-based mask extraction for robot segmentation."""

import torch
import numpy as np
import cv2
import os
from typing import Optional, List, Dict
from PIL import Image
from matplotlib.colors import to_rgb

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

from caliball.utils.image import save_mask_as_image, save_image_with_mask


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
