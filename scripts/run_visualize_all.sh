#!/usr/bin/env bash
# ============================================================
# run_visualize_all.sh
#
# 批量可视化标注 JSON：默认扫描 label_result/ 与 label_out/ 下
# 各子目录（每个子目录 = 一个 task / 数据集名，内含 episode_*.json）。
#
# task_path 解析顺序：
#   1) 当前标注根下的 valid_tasks.json（若存在）
#   2) label_result/valid_tasks.json
#   3) BENCHMARK_ROOTS/<task_name>（RoboMIND）
#   4) LEROBOT_OXE_ROOT/<task_name>（OXE 转 LeRobot，如 berkeley_autolab_ur5）
#
# 用法：
#   bash scripts/run_visualize_all.sh [--episode i]
#
# 环境变量（可选）：
#   LEROBOT_OXE_ROOT   OXE LeRobot 2.1 父目录（默认见下方，与 berkeley_autolab_ur5.yaml 一致）
#   FIRST_FRAME_ONLY=1 只导出每 episode 第一帧 JPG（调试快）
#   SKIP_EXISTING_EPISODE_VIS=0  与 --episode 联用时：即使该 episode 已有 mp4/jpg 也强制重跑
#
# 选项：
#   --episode i  仅处理第 i 个 episode
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

[ -f .venv/bin/activate ] && source .venv/bin/activate

# ─── 可修改的变量 ─────────────────────────────────────────────────────────────

# 多个标注根（均会扫描；不存在则跳过）
LABEL_ROOTS=(
    "${PROJECT_ROOT}/label_result"
    "${PROJECT_ROOT}/label_out"
)

# OXE → LeRobot 数据集父目录（子目录名须与 JSON 所在文件夹名一致）
LEROBOT_OXE_ROOT="${LEROBOT_OXE_ROOT:-/cpfs02/user/xiesicheng.xsc/convert/oxe/lerobot_2.1}"

# 若 valid_tasks.json 中找不到 task，按此顺序在 RoboMIND benchmark 下找 <task_name>
BENCHMARK_ROOTS=(
    "${PROJECT_ROOT}/data/RoboMIND_lerobot_v2.1/benchmark1_0_compressed/franka_3rgb"
    "${PROJECT_ROOT}/data/RoboMIND_lerobot_v2.1/benchmark1_1_compressed/franka_3rgb"
    "${PROJECT_ROOT}/data/RoboMIND_lerobot_v2.1/benchmark1_2_compressed/franka_3rgb"
)

# 相机：留空则**不传** --cameras，由 visualize_labels.py 从 JSON 推断（Berkeley 单相机 image 等）
# RoboMIND 三相机请设为：CAMERAS="camera_left camera_right camera_top"
CAMERAS=""

FPS=15
ALPHA=0.45
MAX_PARALLEL=1
EPISODE_INDEX=""

# NO_MASK="--no_mask"
# NO_BBOX="--no_bbox"
# NO_EEF="--no_eef"
# NO_GRIP="--no_grip"

while [ $# -gt 0 ]; do
    case "$1" in
        --episode)
            if [ -n "${2:-}" ] && [[ "${2}" =~ ^[0-9]+$ ]]; then
                EPISODE_INDEX="$2"
                shift 2
            else
                echo "[ERROR] --episode 需要数字参数，如: --episode 0"
                exit 1
            fi
            ;;
        *)
            shift
            ;;
    esac
done

# ─── 正文 ────────────────────────────────────────────────────────────────────

EXTRA_ARGS=""
[ -n "${NO_MASK:-}" ]       && EXTRA_ARGS="${EXTRA_ARGS} --no_mask"
[ -n "${NO_BBOX:-}" ]       && EXTRA_ARGS="${EXTRA_ARGS} --no_bbox"
[ -n "${NO_EEF:-}" ]        && EXTRA_ARGS="${EXTRA_ARGS} --no_eef"
[ -n "${NO_GRIP:-}" ]       && EXTRA_ARGS="${EXTRA_ARGS} --no_grip"
[ -n "${EPISODE_INDEX}" ]   && EXTRA_ARGS="${EXTRA_ARGS} --episode ${EPISODE_INDEX}"
[ "${FIRST_FRAME_ONLY:-0}" = "1" ] && EXTRA_ARGS="${EXTRA_ARGS} --first_frame_only"

if [ -n "${CAMERAS}" ]; then
    # shellcheck disable=SC2206
    CAM_ARGS=(--cameras ${CAMERAS})
else
    CAM_ARGS=()
fi

resolve_task_path() {
    local tname="$1"
    local vjson="$2"

    if [ -f "${vjson}" ]; then
        local path
        path=$(python - "${vjson}" "${tname}" <<'PYEOF'
import json, sys
with open(sys.argv[1]) as f:
    tasks = json.load(f)
print(tasks.get(sys.argv[2], ""))
PYEOF
)
        if [ -n "${path}" ] && [ -d "${path}" ]; then
            echo "${path}"
            return
        fi
    fi

    local fallback_json="${PROJECT_ROOT}/label_result/valid_tasks.json"
    if [ "${vjson}" != "${fallback_json}" ] && [ -f "${fallback_json}" ]; then
        path=$(python - "${fallback_json}" "${tname}" <<'PYEOF'
import json, sys
with open(sys.argv[1]) as f:
    tasks = json.load(f)
print(tasks.get(sys.argv[2], ""))
PYEOF
)
        if [ -n "${path}" ] && [ -d "${path}" ]; then
            echo "${path}"
            return
        fi
    fi

    for root in "${BENCHMARK_ROOTS[@]}"; do
        if [ -d "${root}/${tname}" ]; then
            echo "${root}/${tname}"
            return
        fi
    done

    if [ -n "${LEROBOT_OXE_ROOT}" ] && [ -d "${LEROBOT_OXE_ROOT}/${tname}" ]; then
        echo "${LEROBOT_OXE_ROOT}/${tname}"
        return
    fi

    echo ""
}

visualize_task() {
    local tname="$1"
    local json_dir="${LABEL_ROOT}/${tname}"
    local out_dir="${VIS_ROOT}/${tname}"

    if ! ls "${json_dir}"/*.json &>/dev/null; then
        echo "[SKIP] ${LABEL_ROOT##*/}/${tname}：目录中无 JSON 文件"
        return
    fi

    # 仅在使用 --episode 时跳过「该 episode 子目录里已有 mp4/jpg」；不再因其它 episode 有输出就跳过整 task
    if [ -n "${EPISODE_INDEX}" ] && [ "${SKIP_EXISTING_EPISODE_VIS:-1}" = "1" ]; then
        ep_dir="${out_dir}/episode_$(printf '%06d' "${EPISODE_INDEX}")"
        if ls "${ep_dir}"/*.mp4 &>/dev/null 2>&1 || ls "${ep_dir}"/*.jpg &>/dev/null 2>&1; then
            echo "[SKIP] ${tname}：episode ${EPISODE_INDEX} 已有可视化（${ep_dir}），跳过"
            return
        fi
    fi

    local task_path
    task_path=$(resolve_task_path "${tname}" "${VALID_TASKS_JSON}")
    if [ -z "${task_path}" ]; then
        echo "[WARN] ${tname}：找不到原始数据集路径（可设置 LEROBOT_OXE_ROOT 或 valid_tasks.json），跳过"
        return
    fi

    mkdir -p "${out_dir}"
    echo ">>> [${LABEL_ROOT##*/}] 任务: ${tname}"
    echo "    数据: ${task_path}"
    echo "    输出: ${out_dir}"

    python scripts/visualize_labels.py \
        --json_dir   "${json_dir}" \
        --task_path  "${task_path}" \
        --output_dir "${out_dir}" \
        "${CAM_ARGS[@]}" \
        --fps        "${FPS}" \
        --alpha      "${ALPHA}" \
        ${EXTRA_ARGS}
}

echo "=== 批量可视化（MAX_PARALLEL=${MAX_PARALLEL}）==="
[ -n "${EPISODE_INDEX}" ] && echo "    [模式] 仅处理 episode ${EPISODE_INDEX}"
[ -z "${CAMERAS}" ] && echo "    [相机] 未设置 CAMERAS → 由 visualize_labels 从 JSON 自动推断"
[ -n "${CAMERAS}" ] && echo "    [相机] ${CAMERAS}"
echo "    标注根:      ${LABEL_ROOTS[*]}"
echo "    OXE 根目录:  ${LEROBOT_OXE_ROOT:-<未设置>}"
echo ""

_failed=0
_any_root=0

shopt -s nullglob 2>/dev/null || true

for LABEL_ROOT in "${LABEL_ROOTS[@]}"; do
    [ -d "${LABEL_ROOT}" ] || continue
    _any_root=1

    VIS_ROOT="${LABEL_ROOT}/vis"
    if [ -f "${LABEL_ROOT}/valid_tasks.json" ]; then
        VALID_TASKS_JSON="${LABEL_ROOT}/valid_tasks.json"
    else
        VALID_TASKS_JSON="${PROJECT_ROOT}/label_result/valid_tasks.json"
    fi

    echo "--- 扫描: ${LABEL_ROOT} → 输出 ${VIS_ROOT} ---"

    for task_dir in "${LABEL_ROOT}"/*/; do
        [ ! -d "${task_dir}" ] && continue
        tname="$(basename "${task_dir}")"
        [ "${tname}" = "vis" ] && continue

        if [ "${MAX_PARALLEL}" -gt 1 ]; then
            while [ "$(jobs -rp | wc -l)" -ge "${MAX_PARALLEL}" ]; do
                wait -n 2>/dev/null || sleep 0.2
            done
            mkdir -p "${VIS_ROOT}/${tname}"
            # 并行时 LABEL_ROOT/VIS_ROOT 必须固定到当前迭代（子 shell 继承当前值）
            (
                LABEL_ROOT="${LABEL_ROOT}"
                VIS_ROOT="${VIS_ROOT}"
                VALID_TASKS_JSON="${VALID_TASKS_JSON}"
                visualize_task "${tname}" \
                    > "${VIS_ROOT}/${tname}/vis.log" 2>&1
            ) &
            echo "[后台 PID $!] ${LABEL_ROOT##*/}/${tname}  日志: ${VIS_ROOT}/${tname}/vis.log"
        else
            visualize_task "${tname}"
        fi
    done
done

if [ "${_any_root}" = 0 ]; then
    echo "[WARN] 以下目录均不存在，未处理任何任务: ${LABEL_ROOTS[*]}"
    exit 0
fi

if [ "${MAX_PARALLEL}" -gt 1 ]; then
    echo ""
    echo "=== 等待所有并行任务完成 ==="
    for pid in $(jobs -rp); do
        wait "${pid}" || { echo "[WARN] PID ${pid} 失败"; _failed=$((_failed + 1)); }
    done
    wait
fi

if [ "${_failed}" -gt 0 ]; then
    echo "[ERROR] 有 ${_failed} 个任务失败，请检查各 task 下 vis.log"
    exit 1
fi

echo ""
echo "=== 全部完成（已扫描的标注根下 vis/ 子目录）==="
