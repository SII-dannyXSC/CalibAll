from PIL import Image
import cv2
import time


from caliball.utils.feature_extractor import build_feature_extractor
from caliball.pipeline.recognition import Recognizer
from caliball.pipeline.point_tracker import build_tracker
from caliball.pipeline.temporal_pnp import solve_pnp
from caliball.utils.intrinsic_estimator import build_intrinsic_estimator
from caliball.robot import build_robot

class CoarseInit:
    def __init__(self, config):
        self.config = config
        
        feature_extractor = build_feature_extractor(config)
        self.recognizer = Recognizer(feature_extractor)
        self.point_tracker = build_tracker(config)
        self.pnp_solver = solve_pnp
        self.robot_tf = build_robot()
        
        self._intrinsic = None
        self.intrinsic_estimator = None
        
        self._init_intrinsic()

    def to(self, device):
        self.recognizer.to(device)
        self.point_tracker.to(device)
        if self.intrinsic_estimator is not None:
            self.intrinsic_estimator.to(device)

    # TODO: check init
    def _init_recognizer(self, given_img_pil, given_p):
        self.recognizer.reset(img_pil=given_img_pil, p=given_p)
        
    def _init_intrinsic(self, intrinsic=None):
        self._intrinsic = intrinsic
        if intrinsic is None and self.intrinsic_estimator is None:
            self.intrinsic_estimator = build_intrinsic_estimator() 

    def _get_intrinsic(self, img_pil):
        # use vggt to init int intrinsic
        if self._intrinsic is None:
            self._intrinsic, origin_width, origin_height = self.intrinsic_estimator.estimate(img_pil=img_pil)
            width, height = img_pil.size
            
            self._intrinsic[0, :3] *= 1.0 * width / origin_width
            self._intrinsic[1, :3] *= 1.0 * height / origin_height
        
        return self._intrinsic

    def get_extrinsic(self, video, joint_angles, img_idx = 0, method=cv2.SOLVEPNP_EPNP):
        img_pil = Image.fromarray(video[img_idx])

        u, v = self.recognizer.get_uv(target_img_pil=img_pil)
        points_2d, pred_tracks, pred_visibility = self.point_tracker.track(video=video, uv=(u,v), img_idx=img_idx)

        K = self._get_intrinsic(img_pil)
        print(f"K: {K}")
        points_3d = self.robot_tf.fkine(joint_angles)[:,:3,3]
        extrinsic = self.pnp_solver(points_3d=points_3d, points_2d=points_2d, camera_matrix=K, method=method)
        return extrinsic, K