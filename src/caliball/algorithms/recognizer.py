"""Feature recognizer — DINOv2 featurizer + cosine-similarity matcher.

Merges ``caliball.utils.feature_extractor`` and ``caliball.pipeline.recognition``
into a single module.
"""

from torchvision.transforms import PILToTensor
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from PIL import Image


# ---------------------------------------------------------------------------
# DINOv2 feature extractor
# ---------------------------------------------------------------------------

class DINOv2Featurizer:
    def __init__(self, repo_dir='facebookresearch/dinov2', model_id='dinov2_vitb14', local_ckpt_path=None, device=None):
        if local_ckpt_path is None:
            self.model = torch.hub.load(repo_dir, model_id)
        else:
            self.model = torch.hub.load(
                repo_dir,
                model_id,
                source="local",
                trust_repo=True,
                pretrained=False,
            )
            ckpt = torch.load(
                local_ckpt_path,
                map_location="cpu"
            )
            if "model" in ckpt:
                ckpt = ckpt["model"]
            self.model.load_state_dict(ckpt, strict=False)
        self.model.eval()

        self.device = None
        if device is not None:
            self.to(device)

    def to(self, device):
        self.device = torch.device(device)
        self.model.to(self.device)
        return self

    @torch.no_grad()
    def forward(self, img_tensor, block_index=None):
        feats_dict = self.model.get_intermediate_layers(img_tensor, n=1)
        out = feats_dict[0]  # shape (B, num_tokens, C)

        B, N, C = out.shape
        h = w = int(N ** 0.5)
        out = out.transpose(1, 2).reshape(B, C, h, w)

        return out


def build_feature_extractor(config):
    return DINOv2Featurizer(local_ckpt_path=config.dinov2_ckpt_path, repo_dir=config.repo_dir, model_id=config.dino_id)


# ---------------------------------------------------------------------------
# Recognizer — cosine-similarity keypoint matcher
# ---------------------------------------------------------------------------

class Recognizer:
    def __init__(self, feature_extractor, img_pil=None, p=None, img_size=784, device="cuda"):
        self.img_size = img_size
        self.device = device
        self.feature_extractor = feature_extractor

        self.img_pil = img_pil
        self.p = p
        self.vec = None
        self.feature = None
        if self.img_pil is not None:
            self.reset(img_pil, p)

    def to(self, device):
        self.device = device
        self.feature_extractor.to(device)

    def img_pil_to_feature(self, img_pil):
        assert self.img_pil is not None
        img_pil = img_pil.resize((self.img_size, self.img_size))
        img_tensor = (PILToTensor()(img_pil) / 255.0 - 0.5) * 2
        img_tensor = img_tensor.unsqueeze(0).to(self.device)
        feature = self.feature_extractor.forward(img_tensor, block_index=1)
        return feature

    def reset(self, img_pil, p):
        self.img_pil = img_pil
        self.p = p

        x, y = p
        feature = self.img_pil_to_feature(img_pil)
        num_channel = feature.size(1)

        src_ft = feature
        src_ft = nn.Upsample(size=(self.img_size, self.img_size), mode='bilinear')(src_ft)
        vec = src_ft[0, :, y, x].view(1, num_channel)  # 1, C

        self.feature = feature
        self.vec = vec

    def get_uv(self, target_img_pil):
        w, h = target_img_pil.size

        target_img_pil_resized = target_img_pil.resize((self.img_size, self.img_size))
        feature = self.img_pil_to_feature(target_img_pil_resized)
        num_channel = feature.size(1)

        trg_ft = nn.Upsample(size=(self.img_size, self.img_size), mode='bilinear')(feature)  # N, C, H, W
        trg_vec = trg_ft.view(1, num_channel, -1)  # N, C, HW

        src_vec = F.normalize(self.vec)  # 1, C
        trg_vec = F.normalize(trg_vec)  # N, C, HW
        cos_map = torch.matmul(src_vec, trg_vec).view(1, self.img_size, self.img_size).float().cpu().numpy()  # N, H, W

        max_yx = np.unravel_index(cos_map[0].argmax(), cos_map[0].shape)

        x_orig = max_yx[1] * (w / self.img_size)
        y_orig = max_yx[0] * (h / self.img_size)

        return x_orig, y_orig
