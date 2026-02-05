import numpy as np
import torch


def K_to_projection(K, H, W, n=0.001, f=10.0, device = "cuda"):
    fu, fv, cu, cv = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    proj = torch.tensor([[2 * fu / W, 0, -2 * cu / W + 1, 0],
                         [0, 2 * fv / H, 2 * cv / H - 1, 0],
                         [0, 0, -(f + n) / (f - n), -2 * f * n / (f - n)],
                         [0, 0, -1, 0]]).to(device=device).float()
    return proj


def transform_pos(mtx, pos, device = "cuda"):
    t_mtx = torch.from_numpy(mtx).to(device=device) if isinstance(mtx, np.ndarray) else mtx
    # (x,y,z) -> (x,y,z,1)
    posw = torch.cat([pos, torch.ones([pos.shape[0], 1]).to(device=device)], axis=1)
    return torch.matmul(posw, t_mtx.t())[None, ...]
