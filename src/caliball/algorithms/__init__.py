"""Algorithm components with swappable Protocol interfaces.

Protocols
---------
- ``FeatureRecognizer`` — visual keypoint recognizer
- ``PointTracker`` — 2-D point tracker across frames
- ``PoseEstimator`` — PnP solver (callable)
- ``MaskExtractor`` — segmentation mask extractor
- ``IntrinsicEstimator`` — monocular intrinsic estimator

Concrete implementations and factory helpers are available in sub-modules:
    ``caliball.algorithms.recognizer``
    ``caliball.algorithms.tracker``
    ``caliball.algorithms.pose_estimator``
    ``caliball.algorithms.mask_extractor``
    ``caliball.algorithms.intrinsic_estimator``
    ``caliball.algorithms.rendering_optimizer``
"""

from caliball.algorithms._protocols import (
    FeatureRecognizer,
    PointTracker,
    PoseEstimator,
    MaskExtractor,
    IntrinsicEstimator,
)

__all__ = [
    # Protocols
    "FeatureRecognizer",
    "PointTracker",
    "PoseEstimator",
    "MaskExtractor",
    "IntrinsicEstimator",
]
