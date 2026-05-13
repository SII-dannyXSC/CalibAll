"""
find_valid_tasks.py

枚举 RoboMIND franka_3rgb 数据集中所有可用的 task 路径，
输出格式为一个 JSON 文件，方便 process_dataset.py 批量处理。

用法：
    python scripts/find_valid_tasks.py
    python scripts/find_valid_tasks.py --output ./label_result/valid_tasks.json
    python scripts/find_valid_tasks.py --print
"""

import argparse
import json
import os

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

_BENCHMARK_ROOTS = [
    os.path.join(_ROOT, "data/RoboMIND_lerobot_v2.1/benchmark1_0_compressed/franka_3rgb"),
    os.path.join(_ROOT, "data/RoboMIND_lerobot_v2.1/benchmark1_1_compressed/franka_3rgb"),
    os.path.join(_ROOT, "data/RoboMIND_lerobot_v2.1/benchmark1_2_compressed/franka_3rgb"),
]

_REQUIRED_DIRS = {"data", "meta", "videos"}


def is_valid_lerobot_task(path):
    if not os.path.isdir(path):
        return False
    entries = set(os.listdir(path))
    return bool(_REQUIRED_DIRS & entries)


def find_all_valid_tasks(benchmark_roots=None):
    """
    遍历所有 benchmark 根目录，返回 {task_name: path} 字典。
    同名任务若出现在多个 benchmark 中，优先选优先级靠前的。
    """
    roots = benchmark_roots or _BENCHMARK_ROOTS
    task_map = {}
    for root in roots:
        if not os.path.isdir(root):
            print(f"[SKIP] 目录不存在: {root}")
            continue
        benchmark_tag = os.path.basename(os.path.dirname(root))
        for task_name in sorted(os.listdir(root)):
            task_path = os.path.join(root, task_name)
            if not is_valid_lerobot_task(task_path):
                continue
            if task_name not in task_map:
                task_map[task_name] = task_path
                print(f"[OK] {benchmark_tag}/{task_name}")
            else:
                print(f"[DUP] {benchmark_tag}/{task_name}  (已使用 {task_map[task_name]})")
    return task_map


def main():
    parser = argparse.ArgumentParser(description="枚举所有合法的 LeRobot franka_3rgb 任务路径")
    parser.add_argument(
        "--output", type=str, default=None,
        help="输出 JSON 文件路径（默认 label_result/valid_tasks.json）"
    )
    parser.add_argument(
        "--print", action="store_true", dest="print_only",
        help="仅打印，不写文件"
    )
    parser.add_argument(
        "--benchmark_roots", nargs="+", default=None,
        help="benchmark 根目录列表（覆盖默认）"
    )
    args = parser.parse_args()

    print("=== 搜索所有有效任务 ===")
    task_map = find_all_valid_tasks(args.benchmark_roots)
    print(f"\n共找到 {len(task_map)} 个有效任务。")

    if args.print_only:
        print(json.dumps(task_map, indent=2, ensure_ascii=False))
        return

    output_path = args.output or os.path.join(_ROOT, "label_result", "valid_tasks.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(task_map, f, indent=2, ensure_ascii=False)
    print(f"\n任务路径已保存到: {output_path}")


if __name__ == "__main__":
    main()
