"""Dataset and camera scanning utilities.

Extracts the scanning logic previously embedded in
``web_interaction._scan_cameras`` and ``extrinsic_detection.scan_datasets``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List


def scan_cameras(task_path: str) -> dict:
    """Scan *task_path* for available cameras, state keys, and dataset name.

    Returns a dict with keys:

    * ``cameras``      -- list of camera name strings
    * ``state_keys``   -- list of numeric parquet column names
    * ``dataset_name`` -- inferred from the parent directory name
    * ``error``        -- present only when something went wrong
    """
    tp = Path(task_path)
    if not tp.is_dir():
        return {"error": f"\u8def\u5f84\u4e0d\u5b58\u5728: {task_path}"}

    cameras: list[str] = []

    # Method 1: read meta/info.json (LeRobot 2.1 format)
    info_json = tp / "meta" / "info.json"
    if info_json.is_file():
        try:
            info = json.loads(info_json.read_text())
            cameras = info.get("camera_keys", info.get("video_keys", []))
            if not cameras and "features" in info:
                cameras = [
                    k
                    for k, v in info["features"].items()
                    if isinstance(v, dict) and v.get("dtype") == "video"
                ]
        except Exception:
            pass

    # Method 2: infer from videos/ directory
    if not cameras:
        videos_dir = tp / "videos"
        if videos_dir.is_dir():
            seen: set[str] = set()
            # Flat layout: videos/*.mp4
            for vf in sorted(videos_dir.iterdir()):
                if vf.suffix == ".mp4":
                    cam = (
                        vf.stem.rsplit("_episode_", 1)[0]
                        if "_episode_" in vf.stem
                        else vf.stem
                    )
                    if cam not in seen:
                        cameras.append(cam)
                        seen.add(cam)
            # Chunk layout: videos/chunk-*/camera_name/*.mp4
            if not cameras:
                for chunk_dir in sorted(videos_dir.glob("chunk-*")):
                    if chunk_dir.is_dir():
                        for cam_dir in sorted(chunk_dir.iterdir()):
                            if cam_dir.is_dir() and cam_dir.name not in seen:
                                cameras.append(cam_dir.name)
                                seen.add(cam_dir.name)

    # Infer dataset_name from parent directory
    dataset_name = tp.parent.name

    result: dict = {"dataset_name": dataset_name}
    if cameras:
        result["cameras"] = cameras
    else:
        result["cameras"] = []
        result["error"] = "\u672a\u627e\u5230 camera"

    # Read parquet columns as candidate state_keys (numeric arrays only)
    # Returns list of {name, dim} dicts
    state_keys: list[dict] = []
    data_dir = tp / "data"
    if data_dir.is_dir():
        parquet_files = sorted(data_dir.rglob("*.parquet"))
        if parquet_files:
            try:
                import numpy as np
                import pandas as pd

                df = pd.read_parquet(parquet_files[0])
                skip = {
                    "episode_index",
                    "frame_index",
                    "index",
                    "timestamp",
                    "task_index",
                }
                video_cols = set(cameras) if cameras else set()
                for c in df.columns:
                    if c in skip or c in video_cols:
                        continue
                    try:
                        arr = np.stack(df[c].values[:3])
                        if np.issubdtype(arr.dtype, np.number) and arr.ndim >= 1:
                            dim = arr.shape[-1] if arr.ndim == 2 else 1
                            state_keys.append({"name": c, "dim": int(dim)})
                    except Exception:
                        pass
            except Exception:
                pass

    result["state_keys"] = state_keys
    return result


def scan_datasets(data_root: str) -> List[dict]:
    """Scan *data_root* for available LeRobot datasets.

    Walks two directory levels (``data_root / dataset_name / task_name``)
    looking for directories that contain a ``data/`` or ``meta/`` sub-dir.

    Returns a list of dicts, each with:

    * ``task_path``     -- absolute path string
    * ``task_name``     -- leaf directory name
    * ``dataset_name``  -- parent directory name
    * ``display_name``  -- ``dataset_name/task_name``
    * ``cameras``       -- list of camera name strings
    """
    root = Path(data_root)
    if not root.is_dir():
        print(f"[scan] \u6570\u636e\u6839\u76ee\u5f55\u4e0d\u5b58\u5728: {root}")
        return []

    results: list[dict] = []

    for sub in sorted(root.iterdir()):
        if not sub.is_dir():
            continue
        for task_dir in sorted(sub.iterdir()):
            if not task_dir.is_dir():
                continue
            has_data = (task_dir / "data").is_dir() or (task_dir / "meta").is_dir()
            if not has_data:
                continue

            # Scan cameras via meta/info.json first
            cameras: list[str] = []
            meta_dir = task_dir / "meta"
            if meta_dir.is_dir():
                info_json = meta_dir / "info.json"
                if info_json.is_file():
                    try:
                        info = json.loads(info_json.read_text())
                        cameras = info.get("camera_keys", info.get("video_keys", []))
                    except Exception:
                        pass

            # Fallback: infer from videos/ directory
            if not cameras:
                videos_dir = task_dir / "videos"
                if videos_dir.is_dir():
                    for vf in videos_dir.iterdir():
                        if vf.suffix == ".mp4":
                            cam = (
                                vf.stem.rsplit("_episode_", 1)[0]
                                if "_episode_" in vf.stem
                                else vf.stem
                            )
                            if cam not in cameras:
                                cameras.append(cam)
                    cameras.sort()

            results.append(
                {
                    "task_path": str(task_dir),
                    "task_name": task_dir.name,
                    "dataset_name": sub.name,
                    "display_name": f"{sub.name}/{task_dir.name}",
                    "cameras": cameras,
                }
            )

    print(f"[scan] \u626b\u63cf\u5230 {len(results)} \u4e2a\u6570\u636e\u96c6\uff08{data_root}\uff09")
    return results
