#!/usr/bin/env python3
"""
review_calibration.py

读取标注可视化结果目录下各 task 的预览视频（中间帧或首帧 JPG），启动本地 HTTP 服务，
在浏览器中展示交互式审核页面，让用户逐一判断是否需要重新标定。结果保存为 JSON。

支持三种目录布局：

1. 扁平 / Berkeley：
   ``<base_path>/vis/<task>/episode_000000/ep000000_<cam>_full.mp4``

2. RoboMIND 旧批量：
   ``<base_path>/<b1_x>/<task>/vis/robomind/episode_000000/...``
   （对 ``label_out/robomind_ur`` 传 ``--base_path`` 指向该目录即可；会递归查找所有 ``episode_000000``）

3. label_visual_whole 新批量（推荐）：
   ``<base_path>/{group}/[{benchmark}/]{task}/vis/episode_000000/ep000000_<cam>_full.mp4``
   示例：
     python scripts/review_calibration.py \\
       --base_path ./label_out/visual_whole \\
       --port 8765 --cards_per_row 3

用法：
    python scripts/review_calibration.py --base_path ./label_out/robomind_ur --port 8765
    python scripts/review_calibration.py --base_path ./label_out/visual_whole --cards_per_row 3
    python scripts/review_calibration.py ... --cards_per_row 2    # 一页上每行 2 个任务卡片
    python scripts/review_calibration.py ... --images_per_row 3   # 卡片内相机图每行 3 张（多相机时换行）
    # 浏览器打开 http://localhost:8765 ；远端：ssh -L 8765:localhost:8765 <server>
"""

import argparse
import base64
import glob
import json
import os
import re
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse

import cv2

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def parse_args():
    parser = argparse.ArgumentParser(description="标定质量审核 - 交互式 Web 页面")
    parser.add_argument(
        "--base_path", type=str,
        default=os.path.join(_ROOT, "label_result"),
        help="标注结果根目录：可为 label_result、label_out/robomind_ur 等；将递归查找 episode_000000",
    )
    parser.add_argument(
        "--output_json", type=str, default=None,
        help="审核结果 JSON 输出路径（默认 base_path/recalib_review.json）"
    )
    parser.add_argument(
        "--port", type=int, default=8765,
        help="HTTP 服务端口"
    )
    parser.add_argument(
        "--thumb_w", type=int, default=320,
        help="缩略图宽度（像素）"
    )
    parser.add_argument(
        "--cameras",
        nargs="*",
        default=None,
        metavar="NAME",
        help="相机名列表（与文件名 ep000000_<NAME>_full 一致）。"
        " 不传则从数据目录自动推断（适合 observation.images.camera_top 等）",
    )
    parser.add_argument(
        "--cards_per_row",
        type=int,
        default=1,
        metavar="N",
        help="页面布局：一行并排展示几个任务卡片（默认 1，纵向单列）。",
    )
    parser.add_argument(
        "--images_per_row",
        type=int,
        default=None,
        metavar="N",
        help="单个卡片内部：相机缩略图一行放几张（多相机时自动换行）。"
        " 默认不限制，即该卡片内所有相机同一行。",
    )
    return parser.parse_args()


def _frame_to_b64(frame, thumb_w=320) -> str:
    h, w = frame.shape[:2]
    if w > thumb_w:
        frame = cv2.resize(frame, (thumb_w, int(h * thumb_w / w)))
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
    return base64.b64encode(buf).decode()


def read_preview_b64(base_path: str, thumb_w=320) -> str | None:
    """
    优先读 base_path + '.mp4' 的中间帧；
    若不存在则尝试 base_path + '.jpg'（first_frame_only 模式的输出）。
    """
    mp4 = base_path + ".mp4"
    jpg = base_path + ".jpg"

    if os.path.exists(mp4):
        cap = cv2.VideoCapture(mp4)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, total // 2))
        ok, frame = cap.read()
        cap.release()
        return _frame_to_b64(frame, thumb_w) if ok else None

    if os.path.exists(jpg):
        frame = cv2.imread(jpg)
        return _frame_to_b64(frame, thumb_w) if frame is not None else None

    return None


def _find_episode_000000_dirs(root: str) -> list[str]:
    root = os.path.abspath(root)
    found: list[str] = []
    for dirpath, _dirnames, _filenames in os.walk(root):
        if os.path.basename(dirpath) == "episode_000000":
            found.append(dirpath)
    return sorted(found)


def _cameras_in_episode_dir(ep_dir: str) -> list[str]:
    names: set[str] = set()
    for p in glob.glob(os.path.join(ep_dir, "ep000000_*_full.mp4")) + glob.glob(
        os.path.join(ep_dir, "ep000000_*_full.jpg")
    ):
        m = re.match(r"ep000000_(.+)_full\.(mp4|jpg)$", os.path.basename(p), re.I)
        if m:
            names.add(m.group(1))
    return sorted(names)


def _task_name_for_review(ep_dir: str, base_path: str) -> str:
    """
    根据 episode_000000 目录的路径推断任务名（用于页面显示 & JSON 键）。

    布局 1 — 扁平/Berkeley：
      .../vis/<task>/episode_000000  →  <task>

    布局 2 — RoboMIND 旧批量：
      .../<tag>/<task>/vis/robomind/episode_000000  →  relpath(<tag>/<task>, base_path)

    布局 3 — label_visual_whole 新批量：
      .../{group}/[{bench}/]{task}/vis/episode_000000  →  relpath({group}/[{bench}/]{task}, base_path)
    """
    base_path = os.path.abspath(base_path)
    parent = os.path.dirname(ep_dir)
    parent_name = os.path.basename(parent)

    # 布局 2：.../vis/robomind/episode_000000
    if parent_name == "robomind":
        vis = os.path.dirname(parent)
        if os.path.basename(vis) == "vis":
            task_dir = os.path.dirname(vis)
            try:
                return os.path.relpath(task_dir, base_path)
            except ValueError:
                return os.path.basename(task_dir)

    # 布局 3：.../{task}/vis/episode_000000
    if parent_name == "vis":
        task_dir = os.path.dirname(parent)
        try:
            return os.path.relpath(task_dir, base_path)
        except ValueError:
            return os.path.basename(task_dir)

    # 布局 1：.../vis/{task}/episode_000000
    return parent_name


def _default_cam_labels(cameras: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for c in cameras:
        if "." in c:
            out[c] = c.rsplit(".", 1)[-1].replace("_", " ")
        else:
            out[c] = c.replace("_", " ")
    return out


def load_tasks(base_path: str, cameras: list[str] | None, thumb_w: int) -> tuple[list, list[str]]:
    """
    在 base_path 下递归查找 episode_000000，组装 tasks。
    cameras 为 None 或空列表时，对所有 episode 目录中的文件取相机名并集并排序。
    返回 (tasks, cam_order)。
    """
    base_path = os.path.abspath(base_path)
    ep_dirs = _find_episode_000000_dirs(base_path)
    if not ep_dirs:
        return [], []

    cam_order: list[str]
    if cameras:
        cam_order = list(cameras)
    else:
        union: set[str] = set()
        for d in ep_dirs:
            union.update(_cameras_in_episode_dir(d))
        cam_order = sorted(union)
        if not cam_order:
            print("[WARN] 未能从文件中推断相机名，回退 Berkeley 三相机", file=sys.stderr)
            cam_order = ["camera_left", "camera_right", "camera_top"]

    tasks: list[dict] = []
    seen: set[str] = set()
    for ep_dir in ep_dirs:
        tname = _task_name_for_review(ep_dir, base_path)
        if tname in seen:
            print(f"[WARN] 重复 task 键，跳过: {tname} ({ep_dir})", file=sys.stderr)
            continue
        seen.add(tname)
        frames: dict[str, str | None] = {}
        for cam in cam_order:
            stem = os.path.join(ep_dir, f"ep000000_{cam}_full")
            b64 = read_preview_b64(stem, thumb_w)
            if b64 is None:
                print(f"[WARN] 无法读取预览: {stem}.(mp4|jpg)", file=sys.stderr)
            frames[cam] = b64
        tasks.append({"name": tname, "frames": frames})

    return tasks, cam_order


def load_existing(output_json: str) -> dict:
    if os.path.exists(output_json):
        with open(output_json, encoding="utf-8") as f:
            return json.load(f)
    return {}


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>标定质量审核</title>
<style>
  :root {{
    --ok:     #22c55e;
    --bad:    #ef4444;
    --skip:   #94a3b8;
    --bg:     #0f172a;
    --card:   #1e293b;
    --border: #334155;
    --text:   #e2e8f0;
    --sub:    #94a3b8;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    background: var(--bg); color: var(--text);
    font-family: 'Segoe UI', system-ui, sans-serif;
    min-height: 100vh;
  }}
  header {{
    position: sticky; top: 0; z-index: 100;
    background: #0f172aee; backdrop-filter: blur(8px);
    border-bottom: 1px solid var(--border);
    padding: 14px 24px;
    display: flex; align-items: center; gap: 20px; flex-wrap: wrap;
  }}
  header h1 {{ font-size: 1.1rem; font-weight: 600; flex: 1; }}
  #progress {{ font-size: .85rem; color: var(--sub); }}
  #save-btn {{
    padding: 8px 22px; border-radius: 8px; border: none; cursor: pointer;
    background: var(--ok); color: #fff; font-weight: 600; font-size: .9rem;
    transition: opacity .2s;
  }}
  #save-btn:hover {{ opacity: .85; }}
  #save-btn:disabled {{ background: var(--border); cursor: default; }}
  #status {{ font-size: .85rem; color: var(--ok); min-width: 120px; }}

  main#grid {{
    padding: 24px;
    display: grid;
    grid-template-columns: repeat({cards_per_row}, minmax(0, 1fr));
    gap: 18px;
    align-items: start;
  }}
  .card {{
    background: var(--card);
    border: 2px solid var(--border);
    border-radius: 12px;
    overflow: hidden;
    transition: border-color .2s, box-shadow .2s;
  }}
  .card.ok   {{ border-color: var(--ok);  box-shadow: 0 0 16px #22c55e33; }}
  .card.bad  {{ border-color: var(--bad); box-shadow: 0 0 16px #ef444433; }}
  .cam-row {{
    display: grid;
    /* columns set dynamically per card based on actual cameras available */
    gap: 2px;
    background: var(--border);
  }}
  .cam-cell {{ display: flex; flex-direction: column; background: var(--card); }}
  .cam-cell img {{ width: 100%; display: block; }}
  .cam-label {{
    font-size: .68rem; color: var(--sub); text-align: center;
    padding: 3px 0 4px; background: #0f172a88;
  }}
  .card-body {{ padding: 10px 14px 12px; display: flex; align-items: center; gap: 12px; }}
  .task-name {{
    flex: 1; font-size: .85rem; font-weight: 600; color: var(--text);
    overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
  }}
  .btns {{ display: flex; gap: 8px; flex-shrink: 0; }}
  .btn {{
    padding: 7px 20px; border-radius: 7px; border: none; cursor: pointer;
    font-weight: 600; font-size: .82rem; transition: opacity .15s, transform .1s;
  }}
  .btn:active {{ transform: scale(.96); }}
  .btn-ok  {{ background: #16803444; color: var(--ok);  border: 1px solid var(--ok); }}
  .btn-bad {{ background: #b91c1c44; color: var(--bad); border: 1px solid var(--bad); }}
  .card.ok  .btn-ok  {{ background: var(--ok);  color: #fff; }}
  .card.bad .btn-bad {{ background: var(--bad); color: #fff; }}

  .legend {{ display: flex; gap: 16px; align-items: center; font-size: .8rem; }}
  .dot {{ width:10px; height:10px; border-radius:50%; display:inline-block; margin-right:4px; }}
</style>
</head>
<body>
<header>
  <h1>📷 标定质量审核</h1>
  <div class="legend">
    <span><span class="dot" style="background:var(--ok)"></span>标定 OK</span>
    <span><span class="dot" style="background:var(--bad)"></span>需重新标定</span>
    <span><span class="dot" style="background:var(--border)"></span>未判断</span>
  </div>
  <div id="progress">0 / {total} 已判断</div>
  <div id="status"></div>
  <button id="save-btn" onclick="saveResults()">💾 保存结果</button>
</header>
<main id="grid"></main>

<script>
const TASKS   = {tasks_json};
const EXISTING = {existing_json};
const decisions = {{}};
TASKS.forEach(t => {{ decisions[t.name] = 'recalibrate'; }});
Object.assign(decisions, EXISTING);

function updateProgress() {{
  const n = Object.keys(decisions).length;
  document.getElementById('progress').textContent = n + ' / ' + TASKS.length + ' 已判断';
}}

function escHtml(s) {{
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/"/g,'&quot;');
}}

function setDecision(idx, val) {{
  const name = TASKS[idx].name;
  decisions[name] = val;
  const card = document.getElementById('card-' + idx);
  if (card) card.className = 'card ' + val;
  updateProgress();
}}

const CAM_LABELS    = {cam_labels_json};
const CAM_ORDER     = {cam_order_json};
const IMAGES_PER_ROW = {images_per_row_json};  // null = no limit

function buildGrid() {{
  const grid = document.getElementById('grid');
  TASKS.forEach((t, idx) => {{
    const val = decisions[t.name] || '';
    const card = document.createElement('div');
    card.className = 'card ' + val;
    card.id = 'card-' + idx;

    // Only render cameras that actually have image data for this task
    const availCams = CAM_ORDER.filter(cam => t.frames[cam] != null);
    const nCols = (IMAGES_PER_ROW && availCams.length > IMAGES_PER_ROW)
      ? IMAGES_PER_ROW : availCams.length || 1;

    const camCells = availCams.map(cam => {{
      const b64 = t.frames[cam];
      return `<div class="cam-cell">` +
        `<img src="data:image/jpeg;base64,${{b64}}" loading="lazy">` +
        `<div class="cam-label">${{CAM_LABELS[cam] || cam}}</div>` +
        `</div>`;
    }}).join('');

    const camRow = availCams.length
      ? `<div class="cam-row" style="grid-template-columns:repeat(${{nCols}},minmax(0,1fr))">${{camCells}}</div>`
      : `<div style="aspect-ratio:16/9;background:#0f172a;display:flex;align-items:center;` +
        `justify-content:center;color:var(--sub);font-size:.75rem">无相机预览</div>`;

    card.innerHTML = camRow + `
      <div class="card-body">
        <div class="task-name" title="${{escHtml(t.name)}}">${{escHtml(t.name)}}</div>
        <div class="btns">
          <button class="btn btn-ok"  onclick="setDecision(${{idx}},'ok')">✓ 标定 OK</button>
          <button class="btn btn-bad" onclick="setDecision(${{idx}},'recalibrate')">✗ 重新标定</button>
        </div>
      </div>`;
    grid.appendChild(card);
  }});
  updateProgress();
}}

function saveResults() {{
  const btn = document.getElementById('save-btn');
  const status = document.getElementById('status');
  btn.disabled = true;
  status.textContent = '保存中…';
  fetch('/save', {{
    method: 'POST',
    headers: {{'Content-Type': 'application/json'}},
    body: JSON.stringify(decisions)
  }})
  .then(r => r.json())
  .then(d => {{
    status.textContent = '✅ 已保存 ' + d.path;
    btn.disabled = false;
  }})
  .catch(e => {{
    status.textContent = '❌ 保存失败: ' + e;
    btn.disabled = false;
  }});
}}

buildGrid();
</script>
</body>
</html>
"""


def build_html(
    tasks: list,
    existing: dict,
    cam_order: list[str],
    cam_labels: dict[str, str],
    *,
    cards_per_row: int,
    images_per_row: int | None,
) -> str:
    tasks_for_js = [{"name": t["name"], "frames": t["frames"]} for t in tasks]
    cr = max(1, int(cards_per_row))
    return HTML_TEMPLATE.format(
        total=len(tasks),
        cards_per_row=cr,
        images_per_row_json=json.dumps(images_per_row),   # null or int → JS constant
        cam_order_json=json.dumps(cam_order, ensure_ascii=False),
        cam_labels_json=json.dumps(cam_labels, ensure_ascii=False),
        tasks_json=json.dumps(tasks_for_js, ensure_ascii=False),
        existing_json=json.dumps(existing, ensure_ascii=False),
    )


_html_content = ""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):  # noqa: A002
        pass

    def do_GET(self):
        if urlparse(self.path).path == "/":
            body = _html_content.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_error(404)

    def do_POST(self):
        if urlparse(self.path).path == "/save":
            output_json = self.server.output_json
            length = int(self.headers.get("Content-Length", 0))
            data = json.loads(self.rfile.read(length))
            od = os.path.dirname(output_json)
            if od:
                os.makedirs(od, exist_ok=True)
            merged = load_existing(output_json)
            merged.update(data)
            with open(output_json, "w", encoding="utf-8") as f:
                json.dump(merged, f, ensure_ascii=False, indent=2)
            print(f"\n[OK] 结果已保存 → {output_json}  ({len(data)} 条)")
            resp = json.dumps({"path": output_json}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(resp)))
            self.end_headers()
            self.wfile.write(resp)
        else:
            self.send_error(404)


def main():
    global _html_content

    args = parse_args()
    base_path = os.path.abspath(os.path.expanduser(args.base_path))
    output_json = args.output_json or os.path.join(base_path, "recalib_review.json")

    print("正在提取视频中间帧，请稍候…")
    cameras_arg = args.cameras if args.cameras else None
    tasks, cam_order = load_tasks(base_path, cameras_arg, args.thumb_w)
    if not tasks:
        print(f"[ERROR] 在 {base_path} 下未找到 episode_000000 或无可读预览", file=sys.stderr)
        sys.exit(1)
    if cameras_arg is None:
        print(f"[INFO] 自动推断相机: {cam_order}")
    if args.cards_per_row < 1:
        print("[ERROR] --cards_per_row 须为 >= 1 的整数", file=sys.stderr)
        sys.exit(1)
    if args.cards_per_row > 1:
        print(f"[INFO] 每行任务卡片数: {args.cards_per_row}")
    if args.images_per_row is not None:
        if args.images_per_row < 1:
            print("[ERROR] --images_per_row 须为 >= 1 的整数", file=sys.stderr)
            sys.exit(1)
        print(f"[INFO] 卡片内每行相机缩略图数: {args.images_per_row}")
    existing = load_existing(output_json)
    print(f"共找到 {len(tasks)} 个 task，已有审核结果 {len(existing)} 条")

    cam_labels = _default_cam_labels(cam_order)
    _html_content = build_html(
        tasks,
        existing,
        cam_order,
        cam_labels,
        cards_per_row=args.cards_per_row,
        images_per_row=args.images_per_row,
    )

    server = HTTPServer(("0.0.0.0", args.port), Handler)
    server.output_json = output_json
    url = f"http://localhost:{args.port}"
    print(f"\n>>> 审核页面已就绪：{url}")
    print(f"    远端服务器请先做端口转发：ssh -L {args.port}:localhost:{args.port} <server>")
    print("    审核完成后在页面点击「保存结果」，再按 Ctrl+C 退出\n")

    threading.Timer(0.5, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n服务已停止。")
        if os.path.exists(output_json):
            print(f"结果文件：{output_json}")


if __name__ == "__main__":
    main()
