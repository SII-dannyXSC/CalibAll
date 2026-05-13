#!/usr/bin/env bash
# 单条 episode 快速标注 + 可视化（默认 Berkeley OXE 配置，与 label_from_config_test.sh 一致）
#
#   bash scripts/label_from_config_test_visual.sh
#
# 环境变量（可选）：
#   CONFIG              任务 YAML（默认 src/caliball/config/berkeley_autolab_ur5.yaml）
#   LABEL_TEST_OUT      覆盖标注输出目录；未设置时取 CONFIG 的 label.output_dir
#   TASK_PATH           覆盖 LeRobot 数据根；未设置时取 CONFIG 的 dataset.repo_id（相对路径相对项目根）
#   FIRST_FRAME_ONLY=1  可视化只导出首帧 JPG，加快调试
#   SKIP_LABEL=1        跳过标注，仅做可视化（假定 JSON 已存在）
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

CONFIG="${CONFIG:-src/caliball/config/rdt_aloha.yaml}"

_resolve_from_config() {
  python - "${PROJECT_ROOT}" "${CONFIG}" <<'PY'
import sys
from pathlib import Path

sys.path.insert(0, sys.argv[1])
from src.caliball.config import compose_job_config_from_path

root = Path(sys.argv[1])
cfg_path = sys.argv[2]
cfg = compose_job_config_from_path(cfg_path, project_root=root)

lab = cfg.get("label")
out = None
if lab is not None and lab.get("output_dir") is not None:
    out = str(lab.output_dir)
if not out:
    out = "./label_out/" + str(cfg.get("dataset_name", "task"))

repo = None
ds = cfg.get("dataset")
if ds is not None and ds.get("repo_id") is not None:
    repo = str(ds.repo_id)
if not repo:
    bp, dn = cfg.get("base_path"), cfg.get("dataset_name")
    if bp is not None and dn is not None:
        repo = str(Path(str(bp)) / str(dn))
if repo:
    p = Path(repo)
    if not p.is_absolute():
        p = (root / p).resolve()
    else:
        p = p.resolve()
    repo = str(p)

print(out)
print(repo or "")
PY
}

mapfile -t _cfg_paths < <(_resolve_from_config)
LABEL_TEST_OUT="${LABEL_TEST_OUT:-${_cfg_paths[0]:-}}"
TASK_PATH="${TASK_PATH:-${_cfg_paths[1]:-}}"
VIS_PARENT="${VIS_PARENT:-${LABEL_TEST_OUT}/vis}"

[[ -n "${TASK_PATH}" ]] || {
  echo "[ERROR] 无法从 CONFIG 解析 LeRobot 数据路径（请检查 dataset.repo_id / base_path+dataset_name，或设置 TASK_PATH）"
  exit 1
}

JSON_EP0="${LABEL_TEST_OUT}/episode_000000.json"

if [[ "${SKIP_LABEL:-0}" != "1" ]]; then
  LABEL_EXTRA=()
  [[ "${SKIP_MASK:-0}" == "1" ]] && LABEL_EXTRA+=(--skip_mask)
  [[ -n "${BASE_PATH:-}" ]] && LABEL_EXTRA+=(--base_path "${BASE_PATH}")
  python scripts/label_from_config.py \
    --config "${CONFIG}" \
    --output_dir "${LABEL_TEST_OUT}" \
    --episode_start 0 \
    --max_episodes 1 \
    "${LABEL_EXTRA[@]}"
fi

[[ -f "${JSON_EP0}" ]] || {
  echo "[ERROR] 未找到 ${JSON_EP0}（先跑标注或检查 LABEL_TEST_OUT）"
  exit 1
}

VIS_EXTRA=()
[[ "${FIRST_FRAME_ONLY:-0}" == "1" ]] && VIS_EXTRA+=(--first_frame_only)

python scripts/visualize_labels.py \
  --json_path "${JSON_EP0}" \
  --task_path "${TASK_PATH}" \
  --output_dir "${VIS_PARENT}" \
  --no_eef \
  "${VIS_EXTRA[@]}"

echo "[OK] 标注: ${JSON_EP0}"
echo "[OK] 可视化目录: ${VIS_PARENT}/episode_000000/"