#!/bin/bash
# test_pipeline.sh — End-to-end test of CalibAll pipeline scripts
#
# Tests:
#   1. caliball_web.py --config (pipeline from saved config)
#   2. label.py --format json (batch labeling, JSON output)
#   3. label.py --format lerobot (batch labeling, LeRobot output)
#   4. visualize.py on JSON output (full video)
#   5. visualize.py on LeRobot output (full video)
#
# Usage:
#   PYTHONPATH=src bash scripts/test_pipeline.sh

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

OUTPUT_DIR="test_output"
rm -rf "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

CONFIG_YAML="src/caliball/config/robomind_franka_1rgb.yaml"
SAVED_CONFIG="manual_label/franka_1rgb.bread_on_table.observation.images.camera_top.0.config.json"
TASK_PATH="data/caliball_data/RoboMIND_lerobot_v2.1/benchmark1_0_compressed/franka_1rgb/bread_in_basket"

echo "============================================"
echo "CalibAll Pipeline Test"
echo "============================================"

# --- Test 1: caliball_web.py --config ---
echo ""
echo "[1/5] Testing caliball_web.py --config ..."
if [ -f "$SAVED_CONFIG" ]; then
    python scripts/caliball_web.py \
        --config "$SAVED_CONFIG" \
        --device cuda \
        --result-dir "$OUTPUT_DIR/web_pipeline" \
        --manual-label-dir "$OUTPUT_DIR/web_manual_label"
    echo "[1/5] PASSED — output: $OUTPUT_DIR/web_pipeline/"
else
    echo "[1/5] SKIPPED — no saved config at $SAVED_CONFIG"
fi

# --- Test 2: label.py --format json ---
echo ""
echo "[2/5] Testing label.py --format json ..."
python scripts/label.py \
    --config "$CONFIG_YAML" \
    --output_dir "$OUTPUT_DIR/label_json" \
    --format json \
    --max_episodes 1 \
    --device cuda
echo "[2/5] PASSED — output: $OUTPUT_DIR/label_json/"

# --- Test 3: label.py --format lerobot ---
echo ""
echo "[3/5] Testing label.py --format lerobot ..."
python scripts/label.py \
    --config "$CONFIG_YAML" \
    --output_dir "$OUTPUT_DIR/label_lerobot" \
    --format lerobot \
    --max_episodes 1 \
    --device cuda
echo "[3/5] PASSED — output: $OUTPUT_DIR/label_lerobot/"

# --- Test 4: visualize.py on JSON output (full video) ---
echo ""
echo "[4/5] Testing visualize.py (JSON mode, full video) ..."
JSON_FILE="$OUTPUT_DIR/label_json/episode_000000.json"
if [ -f "$JSON_FILE" ]; then
    python scripts/visualize.py \
        --json_path "$JSON_FILE" \
        --task_path "$TASK_PATH" \
        --output_dir "$OUTPUT_DIR/vis_json"
    echo "[4/5] PASSED — output: $OUTPUT_DIR/vis_json/"
else
    echo "[4/5] FAILED — JSON file not found: $JSON_FILE"
    exit 1
fi

# --- Test 5: visualize.py on LeRobot output (full video) ---
echo ""
echo "[5/5] Testing visualize.py (LeRobot mode, full video) ..."
LEROBOT_DIR="$OUTPUT_DIR/label_lerobot"
if [ -d "$LEROBOT_DIR/data" ]; then
    python scripts/visualize.py \
        --input_format lerobot \
        --dataset_dir "$LEROBOT_DIR" \
        --episodes 0 \
        --output_dir "$OUTPUT_DIR/vis_lerobot"
    echo "[5/5] PASSED — output: $OUTPUT_DIR/vis_lerobot/"
else
    echo "[5/5] SKIPPED — LeRobot dataset not found at $LEROBOT_DIR"
fi

# --- Summary ---
echo ""
echo "============================================"
echo "All tests completed. Output files:"
echo "============================================"
find "$OUTPUT_DIR" -type f \( -name "*.mp4" -o -name "*.json" -o -name "*.jpg" -o -name "*.npy" -o -name "*.yaml" \) | sort
echo ""
echo "Videos:"
find "$OUTPUT_DIR" -name "*.mp4" | sort
