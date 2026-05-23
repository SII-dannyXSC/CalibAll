"""Protocol interfaces for swappable algorithm components."""

from __future__ import annotations

from typing import Any, Optional, Protocol, Tuple, runtime_checkable

import numpy as np


@runtime_checkable
class FeatureRecognizer(Protocol):
    """Feature-based visual recognizer (e.g. DINOv2-backed Recognizer)."""

    def reset(self, img_pil: Any, p: Tuple[int, int]) -> None:
        """Set a reference image and keypoint for matching."""
        ...

    def get_uv(self, target_img_pil: Any) -> Tuple[float, float]:
        """Find the best-matching (u, v) in *target_img_pil*."""
        ...

    def to(self, device: str) -> None:
        """Move the underlying model to *device*."""
        ...


@runtime_checkable
class PointTracker(Protocol):
    """2-D point tracker across video frames (e.g. CoTracker)."""

    def track(
        self, video: np.ndarray, uv: Tuple[float, float], img_idx: int = 0
    ) -> Tuple[np.ndarray, Any, Any]:
        """Track a single query point through *video*.

        Returns:
            points_2d: (T, 2) array of tracked 2-D positions.
            pred_tracks: raw tracker output (for visualisation).
            pred_visibility: per-frame visibility flags.
        """
        ...

    def to(self, device: str) -> "PointTracker":
        """Move the model to *device*."""
        ...

    def visualize(
        self, video: np.ndarray, pred_tracks: Any, pred_visibility: Any, path: str, pad_value: int = 100
    ) -> None:
        """Save a tracking visualisation to *path*."""
        ...


class PoseEstimator(Protocol):
    """Callable that solves PnP (function signature, not a class)."""

    def __call__(
        self,
        points_3d: np.ndarray,
        points_2d: np.ndarray,
        camera_matrix: np.ndarray,
        method: int = ...,
        init_w2c: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Return a 4x4 world-to-camera matrix."""
        ...


@runtime_checkable
class MaskExtractor(Protocol):
    """Segment masks from an image (e.g. SAM3)."""

    def extract_masks(self, img_pil: Any, prompt: str = "robotic arm") -> Optional[np.ndarray]:
        """Return the best mask or ``None``."""
        ...


@runtime_checkable
class IntrinsicEstimator(Protocol):
    """Monocular intrinsic-parameter estimator (e.g. MoGe)."""

    def estimate(self, img_pil: Any) -> Tuple[np.ndarray, float, float]:
        """Return (intrinsic_3x3, width_scale, height_scale)."""
        ...

    def to(self, device: str) -> None:
        """Move the model to *device*."""
        ...
