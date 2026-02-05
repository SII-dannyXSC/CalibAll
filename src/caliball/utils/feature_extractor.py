import torch
import torch.nn as nn
import torch.nn.functional as F

class DINOv2Featurizer:
    def __init__(self ,repo_dir='facebookresearch/dinov2', model_id='dinov2_vitb14', local_ckpt_path = None, device=None):
        if local_ckpt_path is None:
            self.model = torch.hub.load(repo_dir, model_id)
        else:
            self.model = torch.hub.load(
                repo_dir,
                model_id,
                source="local",
                trust_repo=True,
            )
            # 手动加载本地 ckpt
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
        # 获取最后层的 token embedding（去掉 CLS token）
        feats_dict = self.model.get_intermediate_layers(img_tensor, n=1)  # list[tuple]
        print(f"{feats_dict[0].shape=}")
        out = feats_dict[0]  # shape (B, num_tokens, C)

        # 推断特征图大小（ViT14 -> 每 patch = 14x14）
        B, N, C = out.shape
        h = w = int(N ** 0.5)
        out = out.transpose(1, 2).reshape(B, C, h, w)

        return out

def build_feature_extractor(config):
    return DINOv2Featurizer(local_ckpt_path=config.ckpt_path, repo_dir=config.repo_dir, model_id=config.dino_id)
    # return DINOv2Featurizer()