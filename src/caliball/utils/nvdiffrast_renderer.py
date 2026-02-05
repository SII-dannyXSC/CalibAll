import matplotlib.pyplot as plt
import numpy as np
import torch
import nvdiffrast.torch as dr
import trimesh

from caliball.utils.nvdiffrast_utils import K_to_projection, transform_pos


class NVDiffrastRenderer:
    def __init__(self, image_size, device="cuda"):
        """
        image_size: H,W
        """
        # self.
        self.H, self.W = image_size
        self.resolution = image_size
        blender2opencv = torch.tensor([[1, 0, 0, 0],
                                       [0, -1, 0, 0],
                                       [0, 0, -1, 0],
                                       [0, 0, 0, 1]]).to(device=device).float()
        self.opencv2blender = torch.inverse(blender2opencv)
        self.glctx = dr.RasterizeCudaContext(device=device)

    def render_mask(self, verts, faces, K, object_pose, anti_aliasing=True):
        """
        @param verts: N,3, torch.tensor, float, cuda
        @param faces: M,3, torch.tensor, int32, cuda
        @param K: 3,3 torch.tensor, float ,cuda
        @param object_pose: 4,4 torch.tensor, float, cuda
        @return: mask: 0 to 1, HxW torch.cuda.FloatTensor
        """
        device = object_pose.device
        self.opencv2blender= self.opencv2blender.to(device=device)
        
        proj = K_to_projection(K, self.H, self.W, device=device)

        pose = self.opencv2blender @ object_pose

        pos_clip = transform_pos(proj @ pose, verts,device=device)

        rast_out, _ = dr.rasterize(self.glctx, pos_clip, faces, resolution=self.resolution)
        if anti_aliasing:
            vtx_color = torch.ones(verts.shape, dtype=torch.float, device=verts.device)
            color, _ = dr.interpolate(vtx_color[None, ...], rast_out, faces)
            color = dr.antialias(color, rast_out, pos_clip, faces)
            mask = color[0, :, :, 0]
        else:
            mask = rast_out[0, :, :, 2] > 0
        mask = torch.flip(mask, dims=[0])
        return mask

    def render_color(self, verts, faces, K, object_pose):
        """
        @param verts: N,3, torch.tensor, float, cuda
        @param faces: M,3, torch.tensor, int32, cuda
        @param K: 3,3 torch.tensor, float ,cuda
        @param object_pose: 4,4 torch.tensor, float, cuda
        @return: mask: 0 to 1, HxW torch.cuda.FloatTensor
        """
        device = object_pose.device
        self.opencv2blender= self.opencv2blender.to(device=device)
        
        proj = K_to_projection(K, self.H, self.W, device=device)

        pose = self.opencv2blender @ object_pose

        pos_clip = transform_pos(proj @ pose, verts,device=device)

        rast_out, _ = dr.rasterize(self.glctx, pos_clip, faces, resolution=self.resolution)
        color_value = torch.rand(3, device=device).view(1,3)  # RGB 0~1
        vtx_color = color_value.expand(verts.shape[0], 3)  # N,3
        vtx_color = vtx_color.to(dtype=torch.float, device=verts.device)
        vtx_color = vtx_color.contiguous()
        # vtx_color = torch.ones(verts.shape, dtype=torch.float, device=verts.device)
        # vtx_color[:, 0] = 1.0  # R
        # vtx_color[:, 1] = 0.5  # G
        # vtx_color[:, 2] = 0.0  # B
        color, _ = dr.interpolate(vtx_color[None, ...], rast_out, faces)
        color = dr.antialias(color, rast_out, pos_clip, faces)
        color = color[0]
        color = torch.flip(color, dims=[0])
        return color
    
    def render_all(self, verts, faces, K, object_pose, mask_color = None):
        """
        @param verts: N,3, torch.tensor, float, cuda
        @param faces: M,3, torch.tensor, int32, cuda
        @param K: 3,3 torch.tensor, float ,cuda
        @param object_pose: 4,4 torch.tensor, float, cuda
        @return: results: dict, 包含 'color', 'mask', 'depth'
        """
        device = object_pose.device
        self.opencv2blender = self.opencv2blender.to(device=device)
        
        proj = K_to_projection(K, self.H, self.W, device=device)

        # 1. 坐标变换 (世界 -> 相机 -> 裁剪空间)
        # T_cam_obj = self.opencv2blender @ object_pose
        pose = self.opencv2blender @ object_pose
        
        # pos_clip 形状: 1, N, 4
        # Note: transform_pos 内部应该处理 pose 矩阵乘法的顺序和齐次坐标转换
        pos_clip = transform_pos(proj @ pose, verts, device=device) 
        
        # 2. 光栅化 (Rasterization)
        # rast_out: (1, H, W, 4) - 包含 (x, y, z/w, primitive_id)
        rast_out, _ = dr.rasterize(self.glctx, pos_clip, faces, resolution=self.resolution)
        
        # 3. 渲染 Color (使用随机颜色)
        # ------------------------------------------------------------------------
        if mask_color is None:
            color_value = torch.rand(3, device=device).view(1,3)
        else:
            color_value = torch.tensor(mask_color, device=device).view(1,3)
        vtx_color = color_value.expand(verts.shape[0], 3) # N,3
        vtx_color = vtx_color.to(dtype=torch.float, device=verts.device)
        vtx_color = vtx_color.contiguous()
        
        color, _ = dr.interpolate(vtx_color[None, ...], rast_out, faces)
        color = dr.antialias(color, rast_out, pos_clip, faces)
        
        # 4. 渲染 Mask (使用单通道 1.0 作为顶点颜色)
        # ------------------------------------------------------------------------
        # N, 1 的张量，所有值设为 1.0
        mask_color_value = torch.tensor([1.0,1.0,1.0], device=device).view(1,3)
        mask_vtx_color = mask_color_value.expand(verts.shape[0], 3) # N,3
        mask_vtx_color = mask_vtx_color.to(dtype=torch.float, device=verts.device)
        mask_vtx_color = mask_vtx_color.contiguous()
        mask_render, _ = dr.interpolate(mask_vtx_color[None, ...], rast_out, faces)
        mask_render = dr.antialias(mask_render, rast_out, pos_clip, faces)
        mask = mask_render[0, :, :, 0]
        mask = torch.flip(mask, dims=[0])
        
        # 5. 渲染 Depth (使用 z/w 值作为深度信息)
        # ------------------------------------------------------------------------
        depth = rast_out[..., 2]

        # 6. 后处理和返回
        # ------------------------------------------------------------------------
        # 移除批次维度 [0] 并应用翻转 (flip)
        color = color[0]
        color = torch.flip(color, dims=[0])
        
        depth_final = depth[0]
        # 深度图也需要翻转
        depth_final = torch.flip(depth_final, dims=[0])
        
        # 原始的 z/w 值是 [-1, 1] 范围，但只有靠近 1 的值是可见的。
        # 0.0 表示背景（在光栅化中 primitive_id=0 的区域）
        # 我们可以通过 Mask 排除背景
        # depth_final[mask_final == 0] = 0.0 # 将背景深度设为 0.0

        return {
            'color': color,
            'mask': mask,
            'depth': depth_final # 0.0 到 1.0 的线性深度 (z/w)
        }
        
    def batch_render_mask(self, verts, faces, K, anti_aliasing=True):
        """
        @param batch_verts: N,3, torch.tensor, float, cuda
        @param batch_faces: M,3, torch.tensor, int32, cuda
        @param K: 3,3 torch.tensor, float ,cuda
        # @param batch_object_poses: N,4,4 torch.tensor, float, cuda
        @return: mask: 0 to 1, HxW torch.cuda.FloatTensor
        """
        device = verts.device
        self.opencv2blender = self.opencv2blender.to(device)
        proj = K_to_projection(K, self.H, self.W, device=device)

        pose = self.opencv2blender

        pos_clip = transform_pos(proj @ pose, verts, device=device)

        rast_out, _ = dr.rasterize(self.glctx, pos_clip, faces, resolution=self.resolution)
        if anti_aliasing:
            vtx_color = torch.ones(verts.shape, dtype=torch.float, device=verts.device)
            color, _ = dr.interpolate(vtx_color[None, ...], rast_out, faces)
            color = dr.antialias(color, rast_out, pos_clip, faces)
            mask = color[0, :, :, 0]
        else:
            mask = rast_out[0, :, :, 2] > 0
        mask = torch.flip(mask, dims=[0])
        return mask


def main():
    pose = np.array([[0.99638397, -0.0846324, 0.00750877, -0.20668708],
                     [-0.00875172, -0.19013488, -0.9817189, 0.08405855],
                     [0.0845129, 0.97810328, -0.19018805, 0.77892876],
                     [0., 0., 0., 1.]]).astype(np.float32)
    pose = torch.from_numpy(pose).cuda()
    pose.requires_grad = True
    mesh = trimesh.load_mesh("third_party/xarm_ros/xarm_description/meshes/xarm7/visual/link_base.stl")
    # K = np.loadtxt("data/realsense/20230124_092547/K.txt")
    K = np.array([
        [914.589,0.,651.099],
        [0,914.438,371.634],
        [0,0,1]
    ])
    H, W = 720, 1280
    renderer = NVDiffrastRenderer([H, W])
    mask = renderer.render_mask(torch.from_numpy(mesh.vertices).cuda().float(),
                                torch.from_numpy(mesh.faces).cuda().int(),
                                torch.from_numpy(K).cuda().float(),
                                pose)
    import pdb; pdb.set_trace()
    mask = mask.detach().cpu()
    image_np = (mask.numpy() * 255).astype(np.uint8)  # 转为 uint8 类型
    # 保存为 PNG 图像
    from PIL import Image
    image_bw = Image.fromarray(image_np)
    image_bw.save("binary_image.png")
    
    plt.imshow(mask.detach().cpu())
    plt.show()


if __name__ == '__main__':
    main()
