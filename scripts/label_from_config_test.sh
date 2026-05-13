#!/usr/bin/env bash
# 快速测试：只标注 episode 0，输出到单独目录，便于验证管线是否正常。
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck source=/dev/null
[[ -f .venv/bin/activate ]] && source .venv/bin/activate

export PYTHONPATH=.

python scripts/label_from_config.py \
  --config src/caliball/config/berkeley_autolab_ur5.yaml \
  --output_dir ./label_out/berkeley_autolab_ur5_test_one \
  --episode_start 0 \
  --max_episodes 1

echo "[OK] 仅 1 条 episode，结果见 ./label_out/berkeley_autolab_ur5_test_one/episode_000000.json"
