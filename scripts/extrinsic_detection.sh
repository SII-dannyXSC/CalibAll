#!/bin/bash
# ── Web 模式（完整交互 + pipeline） ──
PYTHONPATH=. python scripts/extrinsic_detection.py \
  --task-path /cpfs02/user/xiesicheng.xsc/project/CalibAll/data/RoboMIND_lerobot_v2.1_sl/benchmark1_0_compressed/franka_3rgb/close_cap_trash_can_1 \
  --dataset-name robomind.franka_3rgb \
  --camera-name observation.images.camera_top \
  --robot-type franka --host 0.0.0.0

# ── Config 模式（跳过交互，直接 pipeline） ──
# PYTHONPATH=. python scripts/extrinsic_detection.py \
#   --config manual_label/robomind.franka_3rgb.close_cap_trash_can_1.observation.images.camera_top.0.config.json \
#   --robot-type franka
