"""Config page routes: dataset selection, scanning, loading."""

from __future__ import annotations

from flask import Blueprint, render_template, request, jsonify, current_app
from caliball.web.state import SharedState
from caliball.web.services.dataset_builder import DatasetBuilder

config_bp = Blueprint("config", __name__)


@config_bp.route("/")
def config_page():
    """Render the dataset configuration page."""
    ctx = current_app.config.get("config_context", {})
    yaml_configs = DatasetBuilder.list_configs()
    return render_template(
        "config.html",
        yaml_configs=yaml_configs,
        robot_types=ctx.get("robot_types", []),
        default_robot_type=ctx.get("default_robot_type", "ur5e"),
        default_task_path=ctx.get("default_task_path", ""),
        default_dataset_name=ctx.get("default_dataset_name", ""),
        default_camera_name=ctx.get("default_camera_name", ""),
        default_episode_idx=ctx.get("default_episode_idx", 0),
        default_strike=ctx.get("default_strike", 4),
    )


@config_bp.route("/api/scan", methods=["POST"])
def scan_path():
    """Scan a dataset path for cameras and state_keys."""
    from caliball.web.services.scan_service import scan_cameras

    d = request.get_json(force=True)
    task_path = d.get("task_path", "")
    result = scan_cameras(task_path)
    return jsonify(result)


@config_bp.route("/api/config/yaml_info", methods=["POST"])
def yaml_info():
    """Return parsed YAML config info for frontend rendering."""
    d = request.get_json(force=True)
    filename = d.get("filename", "")
    if not filename:
        return jsonify({"error": "filename is required"})
    try:
        info = DatasetBuilder.parse_config(filename)
        return jsonify(info)
    except FileNotFoundError:
        return jsonify({"error": f"YAML not found: {filename}"})
    except Exception as e:
        return jsonify({"error": str(e)})


@config_bp.route("/api/config/submit", methods=["POST"])
def config_submit():
    """Accept dataset configuration and signal ready."""
    from pathlib import Path

    d = request.get_json(force=True)
    tp = Path(d.get("task_path", ""))
    if not tp.is_dir():
        return jsonify({"ok": False, "error": f"路径不存在: {tp}"})
    if not (tp / "data").is_dir() and not (tp / "meta").is_dir() and not (tp / "videos").is_dir():
        return jsonify({"ok": False, "error": "不是合法的 LeRobot 数据集"})

    state: SharedState = current_app.config["shared_state"]
    state.set("config_result", d)
    state.set("config_done", True)
    return jsonify({"ok": True})


@config_bp.route("/api/loading_status")
def loading_status():
    """Poll loading progress."""
    state: SharedState = current_app.config["shared_state"]
    status = state.get("loading_status", {
        "message": "准备中…", "progress": 0, "done": False
    })
    return jsonify(status)
