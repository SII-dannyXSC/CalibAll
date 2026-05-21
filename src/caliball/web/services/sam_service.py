"""Per-frame SAM annotation state manager.

Encapsulates the ``per_frame_sam`` dict, ``active_frame_idx``, and
predict / set-image lifecycle that was previously scattered across the
main loop in ``web_interaction.py``.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
from PIL import Image


class SamService:
    """Manages per-frame SAM annotation state."""

    def __init__(self, predict_fn: Callable, set_image_fn: Callable):
        self._predict_fn = predict_fn
        self._set_image_fn = set_image_fn
        self._per_frame: dict[int, dict] = {}  # fi -> {points, labels, mask}
        self._active_frame: int = 0

    # ── Properties ───────────────────────────────────────────────────

    @property
    def active_frame(self) -> int:
        return self._active_frame

    # ── Frame state helpers ──────────────────────────────────────────

    def get_frame_state(self, fi: int) -> dict:
        """Return the SAM state for frame *fi*, creating it if absent."""
        if fi not in self._per_frame:
            self._per_frame[fi] = {"points": [], "labels": [], "mask": None}
        return self._per_frame[fi]

    # ── Image management ─────────────────────────────────────────────

    def set_image(self, fi: int, frame: np.ndarray):
        """Set *frame* as the active SAM image and update the active index."""
        self._set_image_fn(Image.fromarray(frame))
        self._active_frame = fi

    def switch_frame(self, fi: int, frame: np.ndarray):
        """Switch to frame *fi* without clearing its existing state."""
        self.set_image(fi, frame)

    # ── Annotation actions ───────────────────────────────────────────

    def add_point(
        self,
        fi: int,
        x: float,
        y: float,
        label: int,
        frame: np.ndarray,
    ) -> Optional[np.ndarray]:
        """Add a foreground/background point and return the updated mask."""
        if fi != self._active_frame:
            self.set_image(fi, frame)
        fs = self.get_frame_state(fi)
        fs["points"].append([x, y])
        fs["labels"].append(label)
        pts = np.array(fs["points"], dtype=np.float32)
        lbs = np.array(fs["labels"], dtype=np.int32)
        fs["mask"] = self._predict_fn(pts, lbs)
        return fs["mask"]

    def undo(self, fi: int) -> Optional[np.ndarray]:
        """Remove the last point and re-predict (or clear the mask)."""
        fs = self.get_frame_state(fi)
        if fs["points"]:
            fs["points"].pop()
            fs["labels"].pop()
        if fs["points"]:
            pts = np.array(fs["points"], dtype=np.float32)
            lbs = np.array(fs["labels"], dtype=np.int32)
            fs["mask"] = self._predict_fn(pts, lbs)
        else:
            fs["mask"] = None
        return fs["mask"]

    # ── Queries ──────────────────────────────────────────────────────

    def get_masks_for_refs(self, mask_refs: list[int]) -> list:
        """Return a list of masks for the given frame indices."""
        return [self.get_frame_state(fi)["mask"] for fi in mask_refs]

    def get_point_count(self, fi: int) -> int:
        """Return the number of annotation points on frame *fi*."""
        return len(self.get_frame_state(fi)["points"])

    # ── Lifecycle ────────────────────────────────────────────────────

    def clear(self):
        """Discard all per-frame state (e.g. on restart)."""
        self._per_frame.clear()
