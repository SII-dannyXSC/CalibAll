#!/usr/bin/env bash
# CARDS_PER_ROW：一页每行几个任务卡片，例：export CARDS_PER_ROW=2
# IMAGES_PER_ROW：单个卡片内每行几张相机图（可选）
cd /cpfs02/user/xiesicheng.xsc/project/CalibAll
EXTRA=()
[[ -n "${CARDS_PER_ROW:-}" ]] && EXTRA+=(--cards_per_row "${CARDS_PER_ROW}")
[[ -n "${IMAGES_PER_ROW:-}" ]] && EXTRA+=(--images_per_row "${IMAGES_PER_ROW}")
PYTHONPATH=. python scripts/review_calibration.py \
  --base_path /cpfs02/user/xiesicheng.xsc/project/CalibAll/label_out/robomind_ur \
  --port 8765 \
  "${EXTRA[@]}"