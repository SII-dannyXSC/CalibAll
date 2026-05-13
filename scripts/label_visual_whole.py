#!/usr/bin/env python3
"""
Batch label + visualize: one episode per task × camera for all 9 datasets.
Output: label_out/visual_whole/{group}/{task}/episode_000000.json + vis/

Resume-safe: skips tasks where episode_000000.json already exists.
GPU parallelism: tasks are assigned to GPUs round-robin;
  max_workers = n_gpus × MAX_PER_GPU concurrent subprocesses.

Usage:
    cd /cpfs02/user/xiesicheng.xsc/project/CalibAll
    python scripts/label_visual_whole.py [--dry-run] [--no-vis] [--gpus 0,1,2,3]
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

# ─────────────────────────────── paths ───────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_BASE     = PROJECT_ROOT / "label_out" / "visual_whole"   # overridable via --output_dir
DATA_BASE    = PROJECT_ROOT / "data" / "caliball_data"
LOG_DIR      = OUT_BASE / "_logs"
MAX_PER_GPU  = 10


# ─────────────────────────────── task spec ───────────────────────────────────
@dataclass
class TaskSpec:
    group:        str
    config:       str            # relative to PROJECT_ROOT
    task_path:    str            # actual LeRobot data dir (for visualize_labels)
    base_path:    Optional[str] = None   # --base_path override
    dataset_name: Optional[str] = None  # --dataset_name override
    # sub-directory under group in output (used to distinguish benchmark variants)
    out_sub:      Optional[str] = None

    @property
    def short_name(self) -> str:
        return self.dataset_name or Path(self.task_path).name

    @property
    def out_dir(self) -> Path:
        if self.out_sub:
            return OUT_BASE / self.group / self.out_sub / self.short_name
        return OUT_BASE / self.group / self.short_name

    @property
    def json_path(self) -> Path:
        return self.out_dir / "episode_000000.json"

    @property
    def done(self) -> bool:
        return self.json_path.exists()


# ─────────────────────────────── task list ───────────────────────────────────
def build_tasks() -> List[TaskSpec]:
    tasks: List[TaskSpec] = []

    # ── 1–5. OXE flat datasets (no task iteration needed) ────────────────────
    OXE = [
        ("berkeley_autolab_ur5", "berkeley_autolab_ur5.yaml",
         DATA_BASE / "berkeley_autolab_ur5"),
        ("kaist_nonprehensile",  "nonprehensile.yaml",
         DATA_BASE / "kaist_nonprehensile_converted_externally_to_rlds"),
        ("nyu_franka",           "nyu_franka.yaml",
         DATA_BASE / "nyu_franka_play_dataset_converted_externally_to_rlds"),
        ("toto",                 "toto.yaml",
         DATA_BASE / "toto"),
        ("ucsd_kitchen",         "ucsd_kitchen.yaml",
         DATA_BASE / "ucsd_kitchen_dataset_converted_externally_to_rlds"),
    ]
    for group, cfg_file, data_path in OXE:
        tasks.append(TaskSpec(
            group=group,
            config=f"src/caliball/config/{cfg_file}",
            task_path=str(data_path),
        ))

    robomind_root = DATA_BASE / "RoboMIND_lerobot_v2.1"

    # ── 6. robomind ur_1rgb ───────────────────────────────────────────────────
    for bench_dir in sorted(robomind_root.glob("*/ur_1rgb")):
        bench_name = bench_dir.parent.name
        if not bench_name.startswith("benchmark"):
            continue
        for task_dir in sorted(d for d in bench_dir.iterdir() if d.is_dir()):
            tasks.append(TaskSpec(
                group="ur_1rgb",
                config="src/caliball/config/robomind_ur5e_1rgb.yaml",
                task_path=str(task_dir),
                base_path=str(bench_dir),
                dataset_name=task_dir.name,
                out_sub=bench_name,
            ))

    # ── 7. robomind agilex_3rgb ───────────────────────────────────────────────
    for bench_dir in sorted(robomind_root.glob("*/agilex_3rgb")):
        bench_name = bench_dir.parent.name
        if not bench_name.startswith("benchmark"):
            continue
        for task_dir in sorted(d for d in bench_dir.iterdir() if d.is_dir()):
            tasks.append(TaskSpec(
                group="agilex_3rgb",
                config="src/caliball/config/robomind_aloha.yaml",
                task_path=str(task_dir),
                base_path=str(bench_dir),
                dataset_name=task_dir.name,
                out_sub=bench_name,
            ))

    # ── 8. robomind franka_3rgb ───────────────────────────────────────────────
    for bench_dir in sorted(robomind_root.glob("*/franka_3rgb")):
        bench_name = bench_dir.parent.name
        if not bench_name.startswith("benchmark"):
            continue
        for task_dir in sorted(d for d in bench_dir.iterdir() if d.is_dir()):
            tasks.append(TaskSpec(
                group="franka_3rgb",
                config="src/caliball/config/robomind_franka.yaml",
                task_path=str(task_dir),
                base_path=str(bench_dir),
                dataset_name=task_dir.name,
                out_sub=bench_name,
            ))

    # ── 9. rdt_aloha ──────────────────────────────────────────────────────────
    rdt_base = DATA_BASE / "rdt_aloha_lerobot2.1"
    for task_dir in sorted(d for d in rdt_base.iterdir() if d.is_dir()):
        tasks.append(TaskSpec(
            group="rdt_aloha",
            config="src/caliball/config/rdt_aloha.yaml",
            task_path=str(task_dir),
            base_path=str(rdt_base),
            dataset_name=task_dir.name,
        ))

    return tasks


# ─────────────────────────────── worker ──────────────────────────────────────
def run_one(spec: TaskSpec, gpu_id: int, do_vis: bool) -> tuple[bool, str]:
    """Label episode 0 then optionally visualize. Returns (ok, status_str)."""
    if spec.done:
        return True, "SKIP"

    spec.out_dir.mkdir(parents=True, exist_ok=True)
    log_file = LOG_DIR / f"{spec.group}__{spec.short_name}.log"
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu_id)}

    # ── label ─────────────────────────────────────────────────────────────────
    label_cmd = [
        sys.executable, "scripts/label_from_config.py",
        "--config",      spec.config,
        "--output_dir",  str(spec.out_dir),
        "--max_episodes", "1",
    ]
    if spec.base_path:
        label_cmd += ["--base_path", spec.base_path]
    if spec.dataset_name:
        label_cmd += ["--dataset_name", spec.dataset_name]

    with open(log_file, "w") as lf:
        rc = subprocess.run(
            label_cmd, cwd=str(PROJECT_ROOT), env=env,
            stdout=lf, stderr=subprocess.STDOUT,
        ).returncode

    if rc != 0 or not spec.json_path.exists():
        return False, f"FAIL:label(rc={rc})"

    if not do_vis:
        return True, "OK"

    # ── visualize ─────────────────────────────────────────────────────────────
    vis_cmd = [
        sys.executable, "scripts/visualize_labels.py",
        "--json_path",  str(spec.json_path),
        "--task_path",  spec.task_path,
        "--output_dir", str(spec.out_dir / "vis"),
        "--no_eef",
    ]
    with open(log_file, "a") as lf:
        rc = subprocess.run(
            vis_cmd, cwd=str(PROJECT_ROOT), env=env,
            stdout=lf, stderr=subprocess.STDOUT,
        ).returncode

    if rc != 0:
        return False, f"FAIL:vis(rc={rc})"

    return True, "OK"


# ─────────────────────────────── main ────────────────────────────────────────
def detect_gpus() -> List[int]:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            text=True,
        )
        return [int(x) for x in out.strip().split() if x.strip().isdigit()]
    except Exception:
        return [0]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run",  action="store_true", help="只打印任务列表，不执行")
    p.add_argument("--no-vis",   action="store_true", help="跳过可视化，只生成 JSON")
    p.add_argument("--gpus",     type=str, default=None,
                   help="逗号分隔 GPU 编号（默认自动检测，如 0,1,2,3）")
    p.add_argument("--workers",  type=int, default=None,
                   help="总并发数（覆盖 --max-per-gpu；默认 n_gpus × max-per-gpu）")
    p.add_argument("--max-per-gpu", type=int, default=MAX_PER_GPU,
                   help=f"每 GPU 最大并发数（默认 {MAX_PER_GPU}）")
    p.add_argument("--output_dir", type=str, default=None,
                   help="输出根目录（覆盖 OUT_BASE，默认 label_out/visual_whole）")
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

    gpus = (
        [int(g) for g in args.gpus.split(",")]
        if args.gpus else detect_gpus()
    )
    n_workers = args.workers if args.workers else len(gpus) * args.max_per_gpu
    logging.info(f"GPUs: {gpus}  workers: {n_workers}")

    tasks   = build_tasks()
    pending = [t for t in tasks if not t.done]
    skipped = len(tasks) - len(pending)
    logging.info(f"总任务: {len(tasks)}  待处理: {len(pending)}  已跳过: {skipped}")

    if args.dry_run:
        for i, t in enumerate(pending[:20]):
            rel = t.out_dir.relative_to(OUT_BASE)
            print(f"  [{i:3d}] {rel}")
        if len(pending) > 20:
            print(f"  ... 共 {len(pending)} 条")
        return

    results = {"ok": 0, "skip": skipped, "fail": 0}
    lock    = threading.Lock()

    def worker(idx_spec):
        idx, spec = idx_spec
        gpu_id   = gpus[idx % len(gpus)]
        ok, msg  = run_one(spec, gpu_id, do_vis=not args.no_vis)
        tag = str(spec.out_dir.relative_to(OUT_BASE))
        with lock:
            if msg == "SKIP":
                results["skip"] += 1
            elif ok:
                results["ok"] += 1
                logging.info(f"[OK  ] {tag}")
            else:
                results["fail"] += 1
                logging.warning(f"[FAIL] {tag}  {msg}")

    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        futs = {ex.submit(worker, (i, t)): t for i, t in enumerate(pending)}
        done = 0
        for _ in as_completed(futs):
            done += 1
            if done % 20 == 0:
                with lock:
                    r = dict(results)
                logging.info(f"进度 {done}/{len(pending)} | {r}")

    logging.info(f"完成: {results}")
    if results["fail"]:
        logging.warning(f"失败任务日志: {LOG_DIR}/")


if __name__ == "__main__":
    main()
