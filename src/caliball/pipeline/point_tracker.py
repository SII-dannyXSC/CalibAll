import torch
import time

from cotracker.predictor import CoTrackerPredictor
from cotracker.utils.visualizer import Visualizer

class Tracker:
    def __init__(self, repo_dir="facebookresearch/co-tracker", model_id = "cotracker3_offline",local_ckpt_path = None , device=None) -> None:
        self.device = device
        
        if local_ckpt_path is not None:
            self.cotracker = CoTrackerPredictor(checkpoint=local_ckpt_path, window_len=60, v2=False)
        else:
            self.cotracker = torch.hub.load(repo_dir, model_id)
        self.cotracker.eval()
        
        if device is not None:
            self.cotracker.to(device)
            
    def to(self, device):
        self.device = device
        self.cotracker.to(device)
        return self

    def track(self, video, uv, img_idx=0):
        video = torch.tensor(video).to(device = self.device, dtype=torch.float32)
        video = video.permute(0, 3, 1, 2) # T C H W 
        video = video.unsqueeze(0)        # B T C H W
        
        u,v = uv
        queries = [[img_idx, u, v]]
        queries = torch.tensor(queries).to(self.device, dtype=torch.float32)
        
        pred_tracks, pred_visibility = self.cotracker(video, queries=queries[None]) # B T N 2,  B T N 1
        
        points_2d = pred_tracks[0].permute(1,0,2)
        points_2d = points_2d.detach().cpu().numpy()
        points_2d = points_2d[0]    # only one query here
        
        vis = Visualizer(save_dir=".", pad_value = 100)
        vis.visualize(video=video,tracks=pred_tracks,visibility=pred_visibility,filename=f"point_tracking_{time.time()}")

        return points_2d, pred_tracks, pred_visibility
    
def build_tracker(config, device=None):
    tracker = Tracker(repo_dir=config.tracker_repo_dir, model_id=config.tracker_id, local_ckpt_path=config.tracker_ckpt_path, device=device)
    return tracker