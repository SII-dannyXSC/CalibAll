"""Pipeline API routes: progress polling and stop control."""

from __future__ import annotations

import threading

from flask import Blueprint, jsonify, current_app

from caliball.web.state import SharedState

pipeline_bp = Blueprint("pipeline", __name__)


@pipeline_bp.route("/state")
def pipeline_state():
    """Get current pipeline execution state."""
    state: SharedState = current_app.config["shared_state"]
    ps = state.get_pipeline_state()
    # Don't send large image data in done stage (blocks fetch)
    if ps.get("stage") == "done":
        ps = {k: v for k, v in ps.items() if k not in ("overlays", "image")}
    return jsonify(ps)


@pipeline_bp.route("/stop", methods=["POST"])
def stop_pipeline():
    """Request pipeline early termination."""
    stop_event: threading.Event = current_app.config.get("pipeline_stop_event")
    if stop_event is not None:
        stop_event.set()
    return jsonify({"ok": True})
