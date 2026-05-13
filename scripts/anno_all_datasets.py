#!/usr/bin/env python3
"""
anno_all_datasets.py

对 data/caliball_data 下所有有 config 的数据集运行 create_lerobot_with_anno.py，
生成带 annotation 列的 lerobot 数据集，输出到 data/caliball_anno_data。

支持：
  - 多线程并发（--workers）+ GPU 轮询分配
  - 外部多进程：--shard i N  （第 i 个进程处理第 i 片任务，0-indexed）
  - 断点续跑（内部 --resume，跳过已全部完成的 task）

用法：
    # 单进程，自动检测 GPU，每 GPU 2 个并发
    python scripts/anno_all_datasets.py

    # 指定 GPU 和并发数
    python scripts/anno_all_datasets.py --gpus 0,1,2,3 --workers 8

    # 外部 4 进程分片（在 4 台机器或 4 个 screen 里分别运行）
    python scripts/anno_all_datasets.py --gpus 0,1 --workers 4 --shard 0 4
    python scripts/anno_all_datasets.py --gpus 0,1 --workers 4 --shard 1 4
    python scripts/anno_all_datasets.py --gpus 0,1 --workers 4 --shard 2 4
    python scripts/anno_all_datasets.py --gpus 0,1 --workers 4 --shard 3 4
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

# ── paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT  = Path(__file__).resolve().parent.parent
DATA_BASE     = PROJECT_ROOT / "data" / "caliball_data"
OUT_BASE      = PROJECT_ROOT / "data" / "caliball_anno_data"
LOG_DIR       = OUT_BASE / "_logs"
DEFAULT_WORKERS_PER_GPU = 2


# ── task spec ─────────────────────────────────────────────────────────────────
@dataclass
class TaskSpec:
    group:        str                   # 输出分组（用于 out_dir）
    config:       str                   # 相对 PROJECT_ROOT 的 YAML 路径
    src_dir:      str                   # 源 lerobot 数据集目录（绝对路径）
    base_path:    Optional[str] = None  # --base_path 覆盖
    dataset_name: Optional[str] = None # --dataset_name 覆盖
    out_sub:      Optional[str] = None  # out_dir 中间层（benchmark 名）

    @property
    def short_name(self) -> str:
        return self.dataset_name or Path(self.src_dir).name

    @property
    def out_dir(self) -> Path:
        parts = [OUT_BASE, self.group]
        if self.out_sub:
            parts.append(self.out_sub)
        parts.append(self.short_name)
        return Path(*parts)

    @property
    def src_total_episodes(self) -> int:
        """读源数据集 info.json 的 total_episodes，失败返回 -1。"""
        try:
            info = json.loads((Path(self.src_dir) / "meta" / "info.json").read_text())
            return int(info.get("total_episodes", -1))
        except Exception:
            return -1

    @property
    def done(self) -> bool:
        """输出 meta/info.json 存在且 total_episodes 与源一致则视为完成。"""
        info_path = self.out_dir / "meta" / "info.json"
        if not info_path.exists():
            return False
        try:
            out_info = json.loads(info_path.read_text())
            src_ep = self.src_total_episodes
            if src_ep < 0:
                return False
            return int(out_info.get("total_episodes", 0)) >= src_ep
        except Exception:
            return False


# ── task list ─────────────────────────────────────────────────────────────────
def build_tasks() -> List[TaskSpec]:
    tasks: List[TaskSpec] = []

    # ── 1. OXE 扁平数据集 ────────────────────────────────────────────────────
    OXE = [
        ("berkeley_autolab_ur5",
         "berkeley_autolab_ur5.yaml",
         DATA_BASE / "berkeley_autolab_ur5"),
        ("kaist_nonprehensile",
         "nonprehensile.yaml",
         DATA_BASE / "kaist_nonprehensile_converted_externally_to_rlds"),
        ("nyu_franka",
         "nyu_franka.yaml",
         DATA_BASE / "nyu_franka_play_dataset_converted_externally_to_rlds"),
        ("toto",
         "toto.yaml",
         DATA_BASE / "toto"),
        ("ucsd_kitchen",
         "ucsd_kitchen.yaml",
         DATA_BASE / "ucsd_kitchen_dataset_converted_externally_to_rlds"),
    ]
    for group, cfg_file, data_path in OXE:
        if data_path.exists():
            tasks.append(TaskSpec(
                group=group,
                config=f"src/caliball/config/{cfg_file}",
                src_dir=str(data_path),
            ))

    robomind_root = DATA_BASE / "RoboMIND_lerobot_v2.1"

    # ── 2. RoboMIND ur_1rgb ──────────────────────────────────────────────────
    for bench_dir in sorted(robomind_root.glob("*/ur_1rgb")):
        bench_name = bench_dir.parent.name
        if not bench_name.startswith("benchmark"):
            continue
        for task_dir in sorted(d for d in bench_dir.iterdir() if d.is_dir()):
            tasks.append(TaskSpec(
                group="ur_1rgb",
                config="src/caliball/config/robomind_ur5e_1rgb.yaml",
                src_dir=str(task_dir),
                base_path=str(bench_dir),
                dataset_name=task_dir.name,
                out_sub=bench_name,
            ))

    # ── 3. RoboMIND agilex_3rgb ──────────────────────────────────────────────
    for bench_dir in sorted(robomind_root.glob("*/agilex_3rgb")):
        bench_name = bench_dir.parent.name
        if not bench_name.startswith("benchmark"):
            continue
        for task_dir in sorted(d for d in bench_dir.iterdir() if d.is_dir()):
            tasks.append(TaskSpec(
                group="agilex_3rgb",
                config="src/caliball/config/robomind_aloha.yaml",
                src_dir=str(task_dir),
                base_path=str(bench_dir),
                dataset_name=task_dir.name,
                out_sub=bench_name,
            ))

    # ── 4. RoboMIND franka_3rgb ──────────────────────────────────────────────
    for bench_dir in sorted(robomind_root.glob("*/franka_3rgb")):
        bench_name = bench_dir.parent.name
        if not bench_name.startswith("benchmark"):
            continue
        for task_dir in sorted(d for d in bench_dir.iterdir() if d.is_dir()):
            tasks.append(TaskSpec(
                group="franka_3rgb",
                config="src/caliball/config/robomind_franka.yaml",
                src_dir=str(task_dir),
                base_path=str(bench_dir),
                dataset_name=task_dir.name,
                out_sub=bench_name,
            ))

    # ── 5. RDT-ALOHA ─────────────────────────────────────────────────────────
    rdt_base = DATA_BASE / "rdt_aloha_lerobot2.1"
    if rdt_base.exists():
        for task_dir in sorted(d for d in rdt_base.iterdir() if d.is_dir()):
            tasks.append(TaskSpec(
                group="rdt_aloha",
                config="src/caliball/config/rdt_aloha.yaml",
                src_dir=str(task_dir),
                base_path=str(rdt_base),
                dataset_name=task_dir.name,
            ))

    return tasks


# ── worker ────────────────────────────────────────────────────────────────────
def run_one(spec: TaskSpec, gpu_id: int) -> tuple[bool, str]:
    spec.out_dir.mkdir(parents=True, exist_ok=True)
    log_file = LOG_DIR / f"{spec.group}__{spec.short_name}.log"

    cmd = [
        sys.executable, "scripts/create_lerobot_with_anno.py",
        "--config",     spec.config,
        "--output_dir", str(spec.out_dir),
        "--resume",
    ]
    if spec.base_path:
        cmd += ["--base_path", spec.base_path]
    if spec.dataset_name:
        cmd += ["--dataset_name", spec.dataset_name]

    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu_id)}

    with open(log_file, "w") as lf:
        rc = subprocess.run(
            cmd, cwd=str(PROJECT_ROOT), env=env,
            stdout=lf, stderr=subprocess.STDOUT,
        ).returncode

    if rc != 0:
        return False, f"FAIL(rc={rc})"
    if not (spec.out_dir / "meta" / "info.json").exists():
        return False, "FAIL(no meta/info.json)"
    return True, "OK"


# ── GPU detection ─────────────────────────────────────────────────────────────
def detect_gpus() -> List[int]:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"], text=True,
        )
        return [int(x) for x in out.strip().split() if x.strip().isdigit()]
    except Exception:
        return [0]


# ── main ──────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--gpus",      type=str, default=None,
                   help="逗号分隔 GPU 编号（默认自动检测）")
    p.add_argument("--workers",   type=int, default=None,
                   help=f"并发任务数（默认 n_gpus × {DEFAULT_WORKERS_PER_GPU}）")
    p.add_argument("--shard",     type=int, nargs=2, default=None,
                   metavar=("I", "N"),
                   help="外部多进程分片：处理第 I 片（共 N 片），0-indexed")
    p.add_argument("--dry-run",   action="store_true",
                   help="只打印任务列表，不执行")
    p.add_argument("--regen-meta", action="store_true",
                   help="强制对所有 task（含已完成）重跑 --resume，仅更新 meta（不重算已有 parquet）")
    p.add_argument("--output_dir", type=str, default=None,
                   help=f"输出根目录（默认 {OUT_BASE}）")
    return p.parse_args()


def main():
    global OUT_BASE, LOG_DIR
    args = parse_args()

    if args.output_dir:
        OUT_BASE = Path(args.output_dir).resolve()
        LOG_DIR  = OUT_BASE / "_logs"

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(LOG_DIR / "run.log"),
        ],
    )

    gpus = ([int(g) for g in args.gpus.split(",")]
            if args.gpus else detect_gpus())
    n_workers = args.workers or len(gpus) * DEFAULT_WORKERS_PER_GPU
    logging.info(f"GPUs: {gpus}  workers: {n_workers}")

    all_tasks = build_tasks()

    # 分片
    if args.shard:
        shard_i, shard_n = args.shard
        all_tasks = [t for i, t in enumerate(all_tasks) if i % shard_n == shard_i]
        logging.info(f"分片 {shard_i}/{shard_n}：{len(all_tasks)} 个任务")

    if args.regen_meta:
        pending = all_tasks
        skipped = 0
        logging.info(f"--regen-meta 模式：强制对所有 {len(pending)} 个 task 重跑（--resume）")
    else:
        pending  = [t for t in all_tasks if not t.done]
        skipped  = len(all_tasks) - len(pending)
        logging.info(f"总任务: {len(all_tasks)}  待处理: {len(pending)}  已跳过: {skipped}")

    if args.dry_run:
        for i, t in enumerate(pending[:30]):
            print(f"  [{i:4d}] {t.group}/{t.out_sub or ''}/{t.short_name}")
        if len(pending) > 30:
            print(f"  ... 共 {len(pending)} 条")
        return

    results = {"ok": 0, "skip": skipped, "fail": 0}
    lock = threading.Lock()
    running: set = set()
    task_times: list = []   # 已完成任务的耗时（秒）
    start_wall = time.time()

    def fmt_eta(seconds: float) -> str:
        seconds = int(seconds)
        h, m, s = seconds // 3600, (seconds % 3600) // 60, seconds % 60
        return f"{h}h{m:02d}m{s:02d}s" if h else f"{m}m{s:02d}s"

    def worker(idx_spec):
        idx, spec = idx_spec
        gpu_id = gpus[idx % len(gpus)]
        tag = f"{spec.group}/{spec.out_sub+'/' if spec.out_sub else ''}{spec.short_name}"
        t0 = time.time()
        with lock:
            running.add(tag)
            logging.info(f"[START] gpu={gpu_id} {tag}  (running: {len(running)})")
        ok, msg = run_one(spec, gpu_id)
        elapsed = time.time() - t0
        with lock:
            running.discard(tag)
            task_times.append(elapsed)
            done_n = results["ok"] + results["fail"] + 1
            remaining = len(pending) - done_n
            avg = sum(task_times) / len(task_times)
            # ETA 考虑并发：remaining 个任务 / workers 并行
            eta = avg * remaining / max(n_workers, 1)
            progress = f"{done_n}/{len(pending)}"
            if ok:
                results["ok"] += 1
                logging.info(
                    f"[OK   ] {tag}  {elapsed:.0f}s  进度={progress}  ETA={fmt_eta(eta)}"
                )
            else:
                results["fail"] += 1
                logging.warning(
                    f"[FAIL ] {tag}  {msg}  {elapsed:.0f}s  进度={progress}  ETA={fmt_eta(eta)}"
                )

    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        futs = {ex.submit(worker, (i, t)): t for i, t in enumerate(pending)}
        done = 0
        for _ in as_completed(futs):
            done += 1
            if done % 20 == 0:
                with lock:
                    r = dict(results)
                    cur = list(running)
                logging.info(f"进度 {done}/{len(pending)} | {r} | 运行中: {cur}")

    logging.info(f"完成: {results}  总耗时={fmt_eta(time.time()-start_wall)}")
    if results["fail"]:
        logging.warning(f"失败任务日志: {LOG_DIR}/")


if __name__ == "__main__":
    main()
