from torchvision.transforms import PILToTensor
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from PIL import Image
from matplotlib import cm

class Recognizer:
    def __init__(self, feature_extractor, img_pil = None, p =None , img_size=784, device="cuda"):
        self.img_size = img_size
        self.device = device
        self.feature_extractor = feature_extractor
        
        self.img_pil = img_pil
        self.p = p
        self.vec = None
        self.feature = None
        if self.img_pil is not None:
            self.reset(img_pil,p)

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
        
        x,y = p
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
        
        trg_ft = nn.Upsample(size=(self.img_size, self.img_size), mode='bilinear')(feature) # N, C, H, W
        trg_vec = trg_ft.view(1, num_channel, -1) # N, C, HW
        
        src_vec = F.normalize(self.vec) # 1, C
        trg_vec = F.normalize(trg_vec) # N, C, HW
        cos_map = torch.matmul(src_vec, trg_vec).view(1, self.img_size, self.img_size).cpu().numpy() # N, H, W

        max_yx = np.unravel_index(cos_map[0].argmax(), cos_map[0].shape)
        
        x_orig = max_yx[1] * (w / self.img_size)
        y_orig = max_yx[0] * (h / self.img_size)
        
        return x_orig,y_orig
    
    def plot_img(self, target_img_pil):
        w, h = target_img_pil.size
        
        target_img_pil_resized = target_img_pil.resize((self.img_size, self.img_size))
        feature = self.img_pil_to_feature(target_img_pil_resized)
        num_channel = feature.size(1)
        
        trg_ft = nn.Upsample(size=(self.img_size, self.img_size), mode='bilinear')(feature) # N, C, H, W
        trg_vec = trg_ft.view(1, num_channel, -1) # N, C, HW
        
        src_vec = F.normalize(self.vec) # 1, C
        trg_vec = F.normalize(trg_vec) # N, C, HW
        cos_map = torch.matmul(src_vec, trg_vec).view(1, self.img_size, self.img_size).cpu().numpy() # N, H, W

        max_yx = np.unravel_index(cos_map[0].argmax(), cos_map[0].shape)
        heatmap = cos_map[0]
        heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap))  # Normalize to [0, 1]
        
        # Resize heatmap back to original resolution
        heatmap_resized = np.array(Image.fromarray((255 * heatmap).astype(np.uint8)).resize((w, h), Image.BILINEAR))

        # Apply a colormap to the heatmap (e.g., viridis)
        heatmap_colored = cm.viridis(heatmap_resized)  # Applying the 'viridis' colormap
        heatmap_colored = (heatmap_colored[:, :, :3] * 255).astype(np.uint8)  # Remove the alpha channel and scale to [0, 255]

        # Convert to RGBA format and add alpha channel for transparency
        alpha_channel = (255 * heatmap).astype(np.uint8)  # Use heatmap for transparency
        alpha_channel_resized = np.array(Image.fromarray(alpha_channel).resize((w, h), Image.BILINEAR))

        # Concatenate RGB and alpha channel
        # heatmap_resized_rgba = np.concatenate([heatmap_colored, alpha_channel_resized[..., None]], axis=-1)  # Add alpha channel

        # Overlay the heatmap on the original image
        alpha = 0.8
        target_img_array = np.array(target_img_pil.convert("RGB"))
        overlay_img = Image.fromarray(np.uint8(target_img_array * (1-alpha) + heatmap_colored * alpha))  # Blend with transparency

        # Save the resulting image
        overlay_img.save('heatmap_overlay_colored.png')  # Save with color

        # 👉 Save original image as well
        target_img_pil.save('original_image.png')

        # fig, ax = plt.subplots(figsize=(3, 3))
        # plt.tight_layout()
        # ax.imshow(target_img_pil_resized)
        # ax.imshow(255 * heatmap, alpha=0.45, cmap='viridis')
        # ax.axis('off')
        # ax.scatter(max_yx[1].item(), max_yx[0].item(), c='r', s=70)
        # plt.show()
        
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        
        x_orig = max_yx[1] * (w / self.img_size)
        y_orig = max_yx[0] * (h / self.img_size)
        print(max_yx[0],max_yx[1])
        print(x_orig, y_orig)
        print(target_img_pil.size)

        # 左边：resize 后的图 + heatmap
        axes[0].imshow(target_img_pil_resized)
        axes[0].imshow(255 * heatmap, alpha=0.45, cmap='viridis')
        axes[0].scatter(max_yx[1], max_yx[0], c='r', s=70)
        axes[0].set_title('Resized Image')
        axes[0].axis('off')

        # 右边：原图 + 映射点
        axes[1].imshow(target_img_pil)
        axes[1].scatter(x_orig, y_orig, c='r', s=70)
        axes[1].set_title('Original Image')
        axes[1].axis('off')

        plt.tight_layout()
        plt.show()

        
        return x_orig,y_orig


