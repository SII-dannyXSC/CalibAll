#!/usr/bin/env python3
"""
从 RoboMIND LeRobot v2.1 的 benchmark*/*/ur_1rgb 各任务中，截取 episode 0 第一帧，
保存为 PNG，并生成 index.html 浏览。

视频多为 AV1，需 imageio[ffmpeg]（见项目 requirements）。
"""

from __future__ import annotations

import argparse
import html
import json
import re
import sys
from pathlib import Path


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _safe_slug(s: str) -> str:
    s = re.sub(r"[^\w.\-]+", "_", s)
    return s.strip("_") or "unnamed"


def _video_keys_from_info(info: dict) -> list[str]:
    feats = info.get("features") or {}
    return [k for k, v in feats.items() if isinstance(v, dict) and v.get("dtype") == "video"]


def _episode_video_path(task_root: Path, info: dict, episode_index: int, video_key: str) -> Path:
    chunks_size = int(info.get("chunks_size") or 1000)
    chunk = episode_index // chunks_size
    rel = (
        f"videos/chunk-{chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
    )
    return task_root / rel


def _save_first_frame_mp4(mp4: Path, out_png: Path) -> bool:
    try:
        import imageio.v3 as iio
    except ImportError:
        print("需要安装 imageio：pip install 'imageio[ffmpeg]'", file=sys.stderr)
        return False
    if not mp4.is_file():
        return False
    try:
        frame = iio.imread(str(mp4), index=0)
    except Exception as e:
        print(f"[WARN] 无法解码 {mp4}: {e}", file=sys.stderr)
        return False
    out_png.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(str(out_png), frame)
    return True


def _iter_ur1rgb_tasks(benchmark_root: Path) -> list[tuple[str, Path]]:
    ur = benchmark_root / "ur_1rgb"
    if not ur.is_dir():
        return []
    out = []
    for p in sorted(ur.iterdir()):
        if p.is_dir() and (p / "meta" / "info.json").is_file():
            out.append((p.name, p))
    return out


def main() -> int:
    root = _project_root()
    default_roots = [
        root
        / "data"
        / "RoboMIND_lerobot_v2.1"
        / "benchmark1_0_compressed",
        root
        / "data"
        / "RoboMIND_lerobot_v2.1"
        / "benchmark1_1_compressed",
    ]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "roots",
        nargs="*",
        type=Path,
        default=default_roots,
        help="benchmark 根目录（内含 ur_1rgb/）；默认使用项目下 data/RoboMIND.../benchmark1_{0,1}_compressed",
    )
    ap.add_argument(
        "-o",
        "--output",
        type=Path,
        default=root / "label_out" / "robomind_ur1rgb_ep0_gallery",
        help="输出目录（images/ 与 index.html）",
    )
    ap.add_argument(
        "--episode",
        type=int,
        default=0,
        help="episode 索引（默认 0）",
    )
    args = ap.parse_args()
    out_dir = args.output.resolve()
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    rows: list[tuple[str, str, str, str]] = []
    # (benchmark_name, task_name, rel_img_for_html, status_note)

    for bench_root in args.roots:
        bench_root = bench_root.expanduser().resolve()
        bench_name = bench_root.name
        for task_name, task_path in _iter_ur1rgb_tasks(bench_root):
            info_path = task_path / "meta" / "info.json"
            try:
                info = json.loads(info_path.read_text(encoding="utf-8"))
            except Exception as e:
                print(f"[WARN] 跳过 {task_path}（读取 info.json 失败）: {e}", file=sys.stderr)
                continue
            keys = _video_keys_from_info(info)
            if not keys:
                print(f"[WARN] {task_path} 无 video feature，跳过", file=sys.stderr)
                continue
            vk = keys[0]
            if len(keys) > 1:
                print(
                    f"[INFO] {bench_name}/{task_name} 多路视频，使用第一路: {vk}",
                    file=sys.stderr,
                )
            mp4 = _episode_video_path(task_path, info, args.episode, vk)
            slug = _safe_slug(f"{bench_name}__{task_name}")
            png = img_dir / f"{slug}.png"
            ok = _save_first_frame_mp4(mp4, png)
            rel = f"images/{png.name}"
            if ok:
                rows.append((bench_name, task_name, rel, vk))
            else:
                rows.append((bench_name, task_name, "", f"失败（{mp4.name}）"))

    # HTML
    parts = [
        "<!DOCTYPE html>",
        '<html lang="zh-CN">',
        "<head>",
        '<meta charset="utf-8"/>',
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\"/>",
        "<title>RoboMIND ur_1rgb — episode 0 首帧</title>",
        "<style>",
        "body { font-family: system-ui, sans-serif; margin: 24px; background: #111; color: #e8e8e8; }",
        "h1 { font-weight: 600; }",
        ".grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); gap: 20px; }",
        "figure { margin: 0; background: #1a1a1a; border-radius: 8px; overflow: hidden; border: 1px solid #333; }",
        "figure img { width: 100%; height: auto; display: block; background: #000; }",
        "figcaption { padding: 10px 12px; font-size: 13px; line-height: 1.45; }",
        ".bench { color: #7cb7ff; }",
        ".task { color: #cfd4dc; font-weight: 500; }",
        ".vk { color: #888; font-size: 11px; }",
        ".err { color: #f66; }",
        "</style>",
        "</head>",
        "<body>",
        "<h1>RoboMIND ur_1rgb — episode 0 第一帧</h1>",
        "<p>生成目录：<code>"
        + html.escape(str(out_dir))
        + "</code></p>",
        '<div class="grid">',
    ]
    for bench_name, task_name, rel, note in rows:
        parts.append("<figure>")
        if rel:
            parts.append(
                f'<img loading="lazy" src="{html.escape(rel)}" alt="{html.escape(task_name)}"/>'
            )
        else:
            parts.append(f'<div class="err" style="padding:24px;">{html.escape(note)}</div>')
        parts.append("<figcaption>")
        parts.append(f'<span class="bench">{html.escape(bench_name)}</span>')
        parts.append(" / ")
        parts.append(f'<span class="task">{html.escape(task_name)}</span>')
        if rel:
            parts.append(f'<div class="vk">{html.escape(note)}</div>')
        parts.append("</figcaption>")
        parts.append("</figure>")
    parts.extend(["</div>", "</body>", "</html>"])
    index_path = out_dir / "index.html"
    index_path.write_text("\n".join(parts), encoding="utf-8")

    ok_n = sum(1 for r in rows if r[2])
    print(f"完成：{ok_n}/{len(rows)} 张图 → {img_dir}")
    print(f"打开浏览：file://{index_path}")
    return 0 if ok_n == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
