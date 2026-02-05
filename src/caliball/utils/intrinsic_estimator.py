import torch
from PIL import Image
from torchvision import transforms as TF
import numpy as np
import cv2
import torch
# from moge.model.v1 import MoGeModel
from moge.model.v2 import MoGeModel # Let's try MoGe-2

# from vggt.models.vggt import VGGT
# from vggt.utils.pose_enc import pose_encoding_to_extri_intri

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
        # Create white background
        background = Image.new("RGBA", img.size, (255, 255, 255, 255))
        # Alpha composite onto the white background
        img = Image.alpha_composite(background, img)

    # Now convert to "RGB" (this step assigns white for transparent areas)
    img = img.convert("RGB")

    width, height = img.size

    if mode == "pad":
        # Make the largest dimension 518px while maintaining aspect ratio
        if width >= height:
            new_width = target_size
            new_height = round(height * (new_width / width) / 14) * 14  # Make divisible by 14
        else:
            new_height = target_size
            new_width = round(width * (new_height / height) / 14) * 14  # Make divisible by 14
    else:  # mode == "crop"
        # Original behavior: set width to 518px
        new_width = target_size
        # Calculate height maintaining aspect ratio, divisible by 14
        new_height = round(height * (new_width / width) / 14) * 14

    # Resize with new dimensions (width, height)
    img = img.resize((new_width, new_height), Image.Resampling.BICUBIC)
    img = to_tensor(img)  # Convert to tensor (0, 1)

    # Center crop height if it's larger than 518 (only in crop mode)
    if mode == "crop" and new_height > target_size:
        start_y = (new_height - target_size) // 2
        img = img[:, start_y : start_y + target_size, :]

    # For pad mode, pad to make a square of target_size x target_size
    if mode == "pad":
        h_padding = target_size - img.shape[1]
        w_padding = target_size - img.shape[2]

        if h_padding > 0 or w_padding > 0:
            pad_top = h_padding // 2
            pad_bottom = h_padding - pad_top
            pad_left = w_padding // 2
            pad_right = w_padding - pad_left

            # Pad with white (value=1.0)
            img = torch.nn.functional.pad(
                img, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=1.0
            )

    shapes.add((img.shape[1], img.shape[2]))
    images.append(img)

    images = torch.stack(images)  # concatenate images

    if images.dim() == 3:
        images = images.unsqueeze(0)

    return images

# class InrinsicEstimator:
#     def __init__(self, device=None):
#         self.device = device
#         self.vggt = VGGT.from_pretrained("facebook/VGGT-1B")
        
#         if device is not None:
#             self.to(device)

#     def to(self, device):
#         self.device = device
#         self.vggt.to(device)

#     def estimate(self, img_pil, mode="crop"):
#         # Preprocess image
#         img_tensor = preprocess_images(img_pil, mode=mode).to(self.device)
#         img_tensor = img_tensor[None]   # add batch dimension
#         height, width = img_tensor.shape[-2], img_tensor.shape[-1]
#         with torch.no_grad():
#             aggregated_tokens_list, ps_idx = self.vggt.aggregator(img_tensor)

#         # Predict Cameras
#         pose_enc = self.vggt.camera_head(aggregated_tokens_list)[-1]
#         # Extrinsic and intrinsic matrices, following OpenCV convention (camera from world)
#         extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, img_tensor.shape[-2:])

#         intrinsic = intrinsic[0][0].cpu().detach().numpy()  # 3x3
        
#         return intrinsic, width, height


class MoGeEstimator:
    def __init__(self, model_id = "Ruicheng/moge-2-vitl-normal", device=None):
        self.device = device
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


def build_intrinsic_estimator():
    return MoGeEstimator()