"""Annotation page routes: SAM interaction, tracking, frame management."""

from __future__ import annotations

import json
import os
import queue

import cv2
import numpy as np
from flask import Blueprint, render_template, request, jsonify, current_app, send_file
from PIL import Image

from src.caliball.web.state import SharedState
from src.caliball.web.services.image_utils import image_to_data_url, image_to_thumb_url

annotate_bp = Blueprint("annotate", __name__)


def _overlay_vis(img, msk, pts, labs):
    """Generate overlay visualization with mask + points."""
    v = np.asarray(img).copy()
    if v.dtype != np.uint8:
        v = (v * 255).astype(np.uint8) if v.max() <= 1.0 else v.astype(np.uint8)
    b = v.copy()
    if msk is not None and msk.size:
        mm = msk > 0
        gg = np.zeros_like(v)
        gg[..., 1] = 255
        v = (b.astype(np.float32) * 0.55 + gg.astype(np.float32) * 0.45).astype(np.uint8)
        v = np.where(mm[..., None], v, b)
    for (px, py), lb in zip(pts, labs):
        c = (255, 0, 0) if lb == 1 else (0, 0, 255)
        cv2.circle(v, (int(px), int(py)), 6, c, -1)
        cv2.circle(v, (int(px), int(py)), 6, (255, 255, 255), 1)
    return image_to_data_url(v)


@annotate_bp.route("/annotate")
def annotate_page():
    """Render the unified annotation page."""
    ctx = current_app.config.get("annotate_context", {})
    return render_template("annotate.html", **ctx)


@annotate_bp.route("/api/frame/<int:idx>")
def get_frame(idx):
    """Get a single frame as data URL."""
    frames = current_app.config.get("frames")
    if frames is None or not (0 <= idx < len(frames)):
        return jsonify({"error": "invalid index"}), 404
    return jsonify({"url": image_to_data_url(frames[idx])})


@annotate_bp.route("/api/sam/state")
def sam_state():
    """Get current SAM overlay state."""
    state: SharedState = current_app.config["shared_state"]
    return jsonify(state.get_sam_state())


@annotate_bp.route("/api/sam/click", methods=["POST"])
def sam_click():
    """Handle SAM point click."""
    d = request.get_json(force=True)
    cmd_q: queue.Queue = current_app.config["cmd_q"]
    cmd_q.put(("/api/sam/click", json.dumps(d).encode()))
    return jsonify({})


@annotate_bp.route("/api/sam/undo", methods=["POST"])
def sam_undo():
    """Undo last SAM point."""
    d = request.get_json(force=True)
    cmd_q: queue.Queue = current_app.config["cmd_q"]
    cmd_q.put(("/api/sam/undo", json.dumps(d).encode()))
    return jsonify({})


@annotate_bp.route("/api/sam/set_image", methods=["POST"])
def sam_set_image():
    """Set SAM image to a specific frame."""
    d = request.get_json(force=True)
    cmd_q: queue.Queue = current_app.config["cmd_q"]
    cmd_q.put(("/api/sam/set_image", json.dumps(d).encode()))
    return jsonify({})


@annotate_bp.route("/api/sam/switch_frame", methods=["POST"])
def sam_switch_frame():
    """Switch active SAM frame."""
    d = request.get_json(force=True)
    cmd_q: queue.Queue = current_app.config["cmd_q"]
    cmd_q.put(("/api/sam/switch_frame", json.dumps(d).encode()))
    return jsonify({})


@annotate_bp.route("/api/done", methods=["POST"])
def annotation_done():
    """Submit annotation result."""
    d = request.get_json(force=True)
    cmd_q: queue.Queue = current_app.config["cmd_q"]
    cmd_q.put(("/api/done", json.dumps(d).encode()))
    return jsonify({})


@annotate_bp.route("/api/finish", methods=["POST"])
def finish():
    """Finish the session."""
    cmd_q: queue.Queue = current_app.config["cmd_q"]
    cmd_q.put(("/api/finish", b"{}"))
    return jsonify({})


@annotate_bp.route("/api/restart", methods=["POST"])
def restart():
    """Restart annotation (keep same dataset)."""
    cmd_q: queue.Queue = current_app.config["cmd_q"]
    cmd_q.put(("/api/restart", b"{}"))
    return jsonify({})


@annotate_bp.route("/api/reconfig", methods=["POST"])
def reconfig():
    """Go back to dataset config page."""
    cmd_q: queue.Queue = current_app.config["cmd_q"]
    cmd_q.put(("/api/reconfig", b"{}"))
    return jsonify({})


@annotate_bp.route("/api/dataset/load", methods=["POST"])
def dataset_load():
    """Load a different dataset (dynamic switch)."""
    d = request.get_json(force=True)
    cmd_q: queue.Queue = current_app.config["cmd_q"]
    cmd_q.put(("/api/dataset/load", json.dumps(d).encode()))
    # Wait for response from main thread
    resp_q = current_app.config.get("ds_resp_q")
    if resp_q:
        try:
            resp = resp_q.get(timeout=60)
        except queue.Empty:
            resp = {"ok": False, "error": "timeout"}
        return jsonify(resp)
    return jsonify({"ok": False, "error": "not supported"})


@annotate_bp.route("/api/auto_detect", methods=["POST"])
def auto_detect():
    """Auto-detect mask using SAM3 text prompt."""
    d = request.get_json(force=True)
    cmd_q: queue.Queue = current_app.config["cmd_q"]
    resp_q = queue.Queue()
    current_app.config["auto_detect_resp_q"] = resp_q
    cmd_q.put(("/api/auto_detect", json.dumps(d).encode()))
    try:
        resp = resp_q.get(timeout=30)
    except queue.Empty:
        resp = {"ok": False, "error": "timeout"}
    return jsonify(resp)


@annotate_bp.route("/api/auto_tracking", methods=["POST"])
def auto_tracking():
    """Auto-generate tracking point using DINOv2 Recognizer."""
    d = request.get_json(force=True)
    cmd_q: queue.Queue = current_app.config["cmd_q"]
    resp_q = queue.Queue()
    current_app.config["auto_tracking_resp_q"] = resp_q
    cmd_q.put(("/api/auto_tracking", json.dumps(d).encode()))
    try:
        resp = resp_q.get(timeout=30)
    except queue.Empty:
        resp = {"ok": False, "error": "timeout"}
    return jsonify(resp)


@annotate_bp.route("/api/dataset/cameras")
def dataset_cameras():
    """Get cameras for a specific dataset index."""
    datasets_info = current_app.config.get("datasets_info", [])
    try:
        idx = int(request.args.get("idx", "0"))
        cams = datasets_info[idx].get("cameras", []) if idx < len(datasets_info) else []
    except Exception:
        cams = []
    return jsonify({"cameras": cams})


@annotate_bp.route("/api/video")
def serve_video():
    """Serve the annotation overlay video."""
    state: SharedState = current_app.config["shared_state"]
    ps = state.get_pipeline_state()
    vpath = ps.get("video_path")
    if vpath and os.path.isfile(vpath):
        return send_file(vpath, mimetype="video/mp4")
    return "", 404


@annotate_bp.route("/api/tracking_video")
def serve_tracking_video():
    """Serve the tracking video."""
    state: SharedState = current_app.config["shared_state"]
    ps = state.get_pipeline_state()
    vpath = ps.get("tracking_video_path")
    if vpath and os.path.isfile(vpath):
        return send_file(vpath, mimetype="video/mp4")
    return "", 404


@annotate_bp.route("/api/download/<key>")
def download_npy(key):
    """Download npy files (intrinsic, extrinsic_coarse, extrinsic_refined)."""
    state: SharedState = current_app.config["shared_state"]
    ps = state.get_pipeline_state()
    fpath = ps.get(f"{key}_path")
    if fpath and os.path.isfile(fpath):
        return send_file(fpath, as_attachment=True, download_name=os.path.basename(fpath))
    return "", 404
