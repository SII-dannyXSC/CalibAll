"""Monocular intrinsic-parameter estimation (MoGe)."""

import os
import torch
from PIL import Image
from torchvision import transforms as TF
import numpy as np
import cv2
from moge.model.v2 import MoGeModel

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
_DEFAULT_LOCAL_PATH = os.path.join(_PROJECT_ROOT, "ckpt", "moge")
_DEFAULT_HF_ID = "Ruicheng/moge-2-vitl-normal"


def preprocess_images(img_pil, mode="crop"):
    # Validate mode
    if mode not in ["crop", "pad"]:
        raise ValueError("Mode must be either 'crop' or 'pad'")

    images = []
    shapes = set()
    to_tensor = TF.ToTensor()
    target_size = 518

    img = img_pil

    # If there's an alpha channel, blend onto white background:
    if img.mode == "RGBA":
        background = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(background, img)

    img = img.convert("RGB")

    width, height = img.size

    if mode == "pad":
        if width >= height:
            new_width = target_size
            new_height = round(height * (new_width / width) / 14) * 14
        else:
            new_height = target_size
            new_width = round(width * (new_height / height) / 14) * 14
    else:  # mode == "crop"
        new_width = target_size
        new_height = round(height * (new_width / width) / 14) * 14

    img = img.resize((new_width, new_height), Image.Resampling.BICUBIC)
    img = to_tensor(img)

    if mode == "crop" and new_height > target_size:
        start_y = (new_height - target_size) // 2
        img = img[:, start_y : start_y + target_size, :]

    if mode == "pad":
        h_padding = target_size - img.shape[1]
        w_padding = target_size - img.shape[2]

        if h_padding > 0 or w_padding > 0:
            pad_top = h_padding // 2
            pad_bottom = h_padding - pad_top
            pad_left = w_padding // 2
            pad_right = w_padding - pad_left

            img = torch.nn.functional.pad(
                img, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=1.0
            )

    shapes.add((img.shape[1], img.shape[2]))
    images.append(img)

    images = torch.stack(images)

    if images.dim() == 3:
        images = images.unsqueeze(0)

    return images


class MoGeEstimator:
    def __init__(self, model_id=None, device=None):
        self.device = device
        if model_id is None:
            model_id = _DEFAULT_HF_ID
        self.moge = MoGeModel.from_pretrained(model_id)

        if device is not None:
            self.to(device)

    def to(self, device):
        self.device = device
        self.moge.to(device)

    def estimate(self, img_pil):
        input_image = np.array(img_pil.convert("RGB"))
        input_image = torch.tensor(input_image / 255.0, dtype=torch.float32, device=self.device).permute(2, 0, 1)

        model_output = self.moge.infer(input_image)
        intrinsic = model_output['intrinsics'].cpu().detach().numpy()
        return intrinsic, 1.0, 1.0


def build_intrinsic_estimator(model_id=None):
    return MoGeEstimator(model_id=model_id)
