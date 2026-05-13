#!/usr/bin/env bash
# 启动 extrinsic_detection.py（Web 选 tracking + SAM3 mask → manual_label）
#
# 两趟流程：第一趟只导出 PNG 后退出；帧齐全后第二趟再跑选点/SAM。
#
# 必设环境变量（或改下面默认值）：
#   TASK_PATH   LeRobot 任务根目录，例如 .../benchmark1_1_compressed/ur_1rgb/bread_in_basket_1
#
# 常用可选：
#   DATASET_NAME       默认 robomind.ur_1rgb（写入 config json）
#   TASK_NAME          默认取 basename(TASK_PATH)
#   CAMERA_NAME        默认 observation.images.camera_top
#   ROBOT_TYPE         默认 ur5e
#   STATE_KEY          默认 actions.joint_position（与 robomind_ur5e_1rgb.yaml 一致）
#   EPISODE_IDX STRIKE START_IDX END_IDX MASK_FRAME_IDX
#   HOST               默认 127.0.0.1；远程可 0.0.0.0 + SSH 转发
#   TRACKING_PORT SAM_PORT
#   NO_BROWSER=1       不自动打开浏览器
#   TRACKING_X + TRACKING_Y  同时设置则跳过网页选点
#   MASK_NPY           指定则跳过 SAM 网页
#
# 示例：
#   export TASK_PATH="$PWD/data/RoboMIND_lerobot_v2.1/benchmark1_0_compressed/ur_1rgb/bread_in_basket_1"
#   bash scripts/run_extrinsic_detection.sh
#   bash scripts/run_extrinsic_detection.sh --device cuda
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# shellcheck source=/dev/null
[[ -f .venv/bin/activate ]] && source .venv/bin/activate

export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

: "${TASK_PATH:?请设置 TASK_PATH（LeRobot 任务目录绝对或相对路径）}"

DATASET_NAME="${DATASET_NAME:-robomind.ur_1rgb}"
TASK_NAME="${TASK_NAME:-$(basename "${TASK_PATH}")}"
CAMERA_NAME="${CAMERA_NAME:-observation.images.camera_top}"
ROBOT_TYPE="${ROBOT_TYPE:-ur5e}"
STATE_KEY="${STATE_KEY:-actions.joint_position}"
EPISODE_IDX="${EPISODE_IDX:-0}"
STRIKE="${STRIKE:-4}"
START_IDX="${START_IDX:-0}"
END_IDX="${END_IDX:-39}"
MASK_FRAME_IDX="${MASK_FRAME_IDX:-35}"
HOST="${HOST:-127.0.0.1}"
TRACKING_PORT="${TRACKING_PORT:-8765}"
SAM_PORT="${SAM_PORT:-8766}"

EXTRA=()
[[ "${NO_BROWSER:-0}" == "1" ]] && EXTRA+=(--no-browser)
[[ -n "${TRACKING_X:-}" && -n "${TRACKING_Y:-}" ]] && EXTRA+=(--tracking-x "${TRACKING_X}" --tracking-y "${TRACKING_Y}")
[[ -n "${MASK_NPY:-}" ]] && EXTRA+=(--mask-npy "${MASK_NPY}")

exec python scripts/extrinsic_detection.py \
  --task-path "${TASK_PATH}" \
  --task-name "${TASK_NAME}" \
  --dataset-name "${DATASET_NAME}" \
  --camera-name "${CAMERA_NAME}" \
  --robot-type "${ROBOT_TYPE}" \
  --state-key "${STATE_KEY}" \
  --episode-idx "${EPISODE_IDX}" \
  --strike "${STRIKE}" \
  --start-idx "${START_IDX}" \
  --end-idx "${END_IDX}" \
  --mask-frame-idx "${MASK_FRAME_IDX}" \
  --host "${HOST}" \
  --tracking-port "${TRACKING_PORT}" \
  --sam-port "${SAM_PORT}" \
  "${EXTRA[@]}" \
  "$@"
