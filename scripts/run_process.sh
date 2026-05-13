#!/usr/bin/env bash
# ============================================================
# run_process.sh
#
# 用法（三种模式）：
#
#   模式 1：处理单个 task（自动 fallback 查找有效路径）
#     bash scripts/run_process.sh
#
#   模式 2：枚举所有有效 task 并批量处理
#     bash scripts/run_process.sh --all
#
#   模式 3：从任务列表文件批量处理
#     python scripts/find_valid_tasks.py --output label_result/valid_tasks.json
#     bash scripts/run_process.sh --tasks-file label_result/valid_tasks.json
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

[ -f .venv/bin/activate ] && source .venv/bin/activate

# ─── 可修改的变量 ─────────────────────────────────────────────────────────────

# 默认 task 名称（用于单 task 模式）
TASK_NAME="1_potatooven"

# 数据集 benchmark 根目录（脚本会按此顺序 fallback）
DATASET_ROOT="${PROJECT_ROOT}/data/RoboMIND_lerobot_v2.1/benchmark1_0_compressed/agilex_3rgb"

# 标注结果输出根目录（每个 task 会在其下创建子目录）
OUTPUT_ROOT="${PROJECT_ROOT}/label_result"

# 数据集名称（用于查询内外参）
DATASET_NAME="robomind/agilex_3rgb"

# 相机列表
CAMERA_NAMES="camera_front"

# 机器人类型（franka / ur5e / aloha_cobot_magic / arx5_robotwin）
ROBOT_TYPE="arx5_robotwin"

# torch 设备
DEVICE="cuda"

# 调试时限制 episode 数量（设为空字符串表示处理全部）
MAX_EPISODES="1"

# 起始 episode（断点续处理）
EPISODE_START=0

# EEF 旋转表示方式
EEF_ROTATION_TYPE="euler_xyz"

# 机械臂 mesh 数量
ARM_MESH_NUM=8

# 是否跳过 mask/bbox 渲染（1=跳过，方便快速调试）
SKIP_MASK=0

# 最大并行任务数（1=串行，>1=多进程并行）
MAX_PARALLEL=1

# ─── 脚本正文 ────────────────────────────────────────────────────────────────

echo "=== 项目根目录: ${PROJECT_ROOT} ==="

build_python_args() {
    local TASK_PATH="$1"
    local OUTPUT_DIR="$2"
    local ARGS=(
        "scripts/process_dataset.py"
        "--task_path"         "${TASK_PATH}"
        "--output_dir"        "${OUTPUT_DIR}"
        "--dataset_name"     "${DATASET_NAME}"
        "--camera_names"      ${CAMERA_NAMES}
        "--robot_type"       "${ROBOT_TYPE}"
        "--device"           "${DEVICE}"
        "--episode_start"     "${EPISODE_START}"
        "--eef_rotation_type" "${EEF_ROTATION_TYPE}"
        "--arm_mesh_num"      "${ARM_MESH_NUM}"
    )
    if [ -n "${MAX_EPISODES}" ]; then
        ARGS+=("--max_episodes" "${MAX_EPISODES}")
    fi
    if [ "${SKIP_MASK}" = "1" ]; then
        ARGS+=("--skip_mask")
    fi
    echo "${ARGS[@]}"
}

process_single_task() {
    local TASK_NAME="$1"
    local TASK_PATH="${DATASET_ROOT}/${TASK_NAME}"
    local OUTPUT_DIR="${OUTPUT_ROOT}/${TASK_NAME}"
    mkdir -p "${OUTPUT_DIR}"
    echo ""
    echo ">>> 任务: ${TASK_NAME}"
    echo "    路径: ${TASK_PATH}  （若为空，脚本会自动 fallback）"
    echo "    输出: ${OUTPUT_DIR}"
    # shellcheck disable=SC2046
    python $(build_python_args "${TASK_PATH}" "${OUTPUT_DIR}")
}

run_task_list() {
    local TASKS_FILE="$1"
    local _failed=0

    while IFS='::' read -r tname tpath; do
        local OUTPUT_DIR="${OUTPUT_ROOT}/${tname}"
        mkdir -p "${OUTPUT_DIR}"
        echo ""
        echo ">>> 任务: ${tname}  输出: ${OUTPUT_DIR}"

        if [ "${MAX_PARALLEL}" -gt 1 ]; then
            while [ "$(jobs -rp | wc -l)" -ge "${MAX_PARALLEL}" ]; do
                wait -n 2>/dev/null || sleep 0.2
            done
            # shellcheck disable=SC2046
            python $(build_python_args "${tpath}" "${OUTPUT_DIR}") \
                > "${OUTPUT_DIR}/process.log" 2>&1 &
            echo "    [后台 PID $!] 日志: ${OUTPUT_DIR}/process.log"
        else
            # shellcheck disable=SC2046
            python $(build_python_args "${tpath}" "${OUTPUT_DIR}")
        fi
    done < <(python - "${TASKS_FILE}" <<'PYEOF'
import json, sys
with open(sys.argv[1]) as f:
    tasks = json.load(f)
for name, path in tasks.items():
    print(f"{name}::{path}")
PYEOF
)

    if [ "${MAX_PARALLEL}" -gt 1 ]; then
        echo ""
        echo "=== 等待所有并行任务完成 ==="
        for pid in $(jobs -rp); do
            wait "${pid}" || { echo "[WARN] PID ${pid} 以非零状态退出"; _failed=$((_failed + 1)); }
        done
        wait
    fi

    if [ "${_failed}" -gt 0 ]; then
        echo "[ERROR] 有 ${_failed} 个任务失败，请检查各任务目录下的 process.log"
        return 1
    fi
}

# ─── 解析命令行 ───────────────────────────────────────────────────────────────

MODE="single"
TASKS_FILE=""

for arg in "$@"; do
    case "$arg" in
        --all) MODE="all" ;;
        --tasks-file=*) MODE="file"; TASKS_FILE="${arg#*=}" ;;
        --tasks-file) MODE="file" ;;
    esac
done

if [ "${MODE}" = "file" ] && [ -z "${TASKS_FILE}" ]; then
    for i in "$@"; do
        if [ -n "${PREV_WAS_TASKS_FILE:-}" ]; then
            TASKS_FILE="$i"
            unset PREV_WAS_TASKS_FILE
        fi
        [ "$i" = "--tasks-file" ] && PREV_WAS_TASKS_FILE=1
    done
fi

# ─── 执行 ─────────────────────────────────────────────────────────────────────

if [ "${MODE}" = "single" ]; then
    process_single_task "${TASK_NAME}"

elif [ "${MODE}" = "all" ]; then
    echo "=== 枚举所有有效任务 ==="
    TASKS_FILE="${OUTPUT_ROOT}/valid_tasks.json"
    python scripts/find_valid_tasks.py --output "${TASKS_FILE}"

    echo "=== 批量处理（MAX_PARALLEL=${MAX_PARALLEL}）==="
    run_task_list "${TASKS_FILE}"

elif [ "${MODE}" = "file" ]; then
    if [ -z "${TASKS_FILE}" ] || [ ! -f "${TASKS_FILE}" ]; then
        echo "[ERROR] --tasks-file 指定的文件不存在: ${TASKS_FILE}"
        exit 1
    fi
    echo "=== 从文件加载任务列表: ${TASKS_FILE}（MAX_PARALLEL=${MAX_PARALLEL}）==="
    run_task_list "${TASKS_FILE}"
fi

echo ""
echo "=== 全部完成 ==="
