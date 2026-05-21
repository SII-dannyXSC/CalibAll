"""Thread-safe shared state for the web annotation UI.

Replaces the raw ``shared`` dict + ``lock`` pattern used in
``web_interaction.py`` with an encapsulated, lock-protected object.
"""

import threading
from typing import Any, Optional


class SharedState:
    """Thread-safe session state for the web annotation UI."""

    def __init__(self):
        self._lock = threading.Lock()
        self._data: dict = {
            "overlay": "",
            "status": "",
            "pts": 0,
        }
        self._pipeline: dict = {}

    # ── SAM overlay state ────────────────────────────────────────────

    def update_overlay(self, overlay_url: str, status: str = "", pts: int = 0):
        """Set the current overlay image URL and optional status / point count."""
        with self._lock:
            self._data["overlay"] = overlay_url
            if status:
                self._data["status"] = status
            self._data["pts"] = pts

    def get_sam_state(self) -> dict:
        """Return a snapshot of ``{overlay, status, pts}``."""
        with self._lock:
            return dict(self._data)

    # ── Pipeline state ───────────────────────────────────────────────

    def update_pipeline(self, **kwargs):
        """Merge *kwargs* into the pipeline sub-dict."""
        with self._lock:
            self._pipeline.update(kwargs)

    def get_pipeline_state(self) -> dict:
        """Return a shallow copy of the pipeline state."""
        with self._lock:
            return dict(self._pipeline)

    def clear_pipeline(self):
        """Remove all pipeline state."""
        with self._lock:
            self._pipeline.clear()

    # ── Bulk helpers ─────────────────────────────────────────────────

    def reset(self, overlay: str = "", status: str = "\u5de6\u952e=\u524d\u666f \u53f3\u952e=\u80cc\u666f"):
        """Reset both SAM and pipeline state to initial values."""
        with self._lock:
            self._data = {"overlay": overlay, "status": status, "pts": 0}
            self._pipeline.clear()

    # ── Generic key access (for ad-hoc keys like _ds_resp_q) ─────

    def set(self, key: str, value: Any):
        """Set an arbitrary key in the data dict."""
        with self._lock:
            self._data[key] = value

    def get(self, key: str, default=None):
        """Get an arbitrary key from the data dict."""
        with self._lock:
            return self._data.get(key, default)

    def pop(self, key: str, default=None):
        """Pop an arbitrary key from the data dict."""
        with self._lock:
            return self._data.pop(key, default)
