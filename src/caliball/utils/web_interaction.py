"""
浏览器内点击交互：本地 HTTP 替代 Matplotlib。
SAM3 推理在主线程消费队列时执行，避免在 HTTP 线程调用 CUDA。
"""

from __future__ import annotations

import base64
import io
import json
import queue
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Callable, Optional, Tuple

HTTPServer.allow_reuse_address = True

import numpy as np
from PIL import Image


def _image_to_data_url(image_rgb: np.ndarray) -> str:
    arr = np.asarray(image_rgb)
    if arr.dtype != np.uint8:
        arr = (arr * 255).astype(np.uint8) if arr.max() <= 1.0 else arr.astype(np.uint8)
    if arr.ndim == 2:
        pil = Image.fromarray(arr, mode="L").convert("RGB")
    else:
        pil = Image.fromarray(arr)
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    b64 = base64.standard_b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def pick_tracking_point_web(
    image_rgb: np.ndarray,
    host: str = "127.0.0.1",
    port: int = 8765,
    open_browser: bool = True,
) -> Tuple[float, float]:
    """浏览器点击图片选 tracking point，确认后返回 (x, y)。"""
    data_url = _image_to_data_url(image_rgb)
    result: dict = {"xy": None, "err": None}
    done = threading.Event()

    html = (
        "<!DOCTYPE html><html lang=zh-CN><head><meta charset=UTF-8><title>tracking point</title>"
        "<style>body{font-family:system-ui;max-width:960px;margin:1rem auto;padding:0 1rem}"
        "#w{position:relative;display:inline-block;border:2px solid #334;cursor:crosshair}"
        "#w img{display:block;max-width:100%;height:auto}"
        "#d{position:absolute;width:14px;height:14px;margin:-7px 0 0 -7px;border:2px solid #fff;"
        "border-radius:50%;background:#e11;box-shadow:0 0 0 1px #0008;display:none;pointer-events:none}"
        "button{margin-top:.75rem;padding:.5rem 1.2rem;cursor:pointer}</style></head><body>"
        "<h1>选择 tracking point</h1><p>单击图像选点，再点确认。</p>"
        f'<div id=w><img id=i src="{data_url}"/><div id=d></div></div>'
        "<p id=c></p><button id=b disabled>确认并关闭</button>"
        "<script>"
        "const i=document.getElementById('i'),d=document.getElementById('d'),b=document.getElementById('b'),c=document.getElementById('c');"
        "let x=null,y=null;"
        "i.onclick=function(ev){const r=i.getBoundingClientRect();const nx=ev.clientX-r.left,ny=ev.clientY-r.top;"
        "const sx=i.naturalWidth/i.clientWidth,sy=i.naturalHeight/i.clientHeight;x=nx*sx;y=ny*sy;"
        "d.style.left=nx+'px';d.style.top=ny+'px';d.style.display='block';c.textContent='('+x.toFixed(1)+', '+y.toFixed(1)+')';b.disabled=false;};"
        "b.onclick=function(){if(x===null)return;fetch('/done',{method:'POST',headers:{'Content-Type':'application/json'},"
        "body:JSON.stringify({x:x,y:y})}).then(()=>{document.body.innerHTML='<p>已提交</p>';});};"
        "</script></body></html>"
    )

    class H(BaseHTTPRequestHandler):
        def log_message(self, format, *args):  # noqa: A002
            pass

        def do_GET(self):
            if self.path in ("/", "/?"):
                b = html.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(b)))
                self.end_headers()
                self.wfile.write(b)
            else:
                self.send_error(404)

        def do_POST(self):
            if self.path != "/done":
                self.send_error(404)
                return
            n = int(self.headers.get("Content-Length", "0"))
            try:
                d = json.loads(self.rfile.read(n).decode("utf-8"))
                result["xy"] = (float(d["x"]), float(d["y"]))
            except Exception as e:
                result["err"] = str(e)
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"{}")
            done.set()

    srv = HTTPServer((host, port), H)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    url = f"http://{host}:{port}/"
    print(f"[web] tracking point: {url}")
    if open_browser:
        webbrowser.open(url)
    done.wait(timeout=3600)
    srv.shutdown()
    th.join(timeout=2)
    if result.get("err"):
        raise RuntimeError(result["err"])
    if result["xy"] is None:
        raise RuntimeError("未收到坐标（超时或未确认）")
    return result["xy"]


def run_sam3_points_web(
    image_rgb: np.ndarray,
    predict_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    host: str = "127.0.0.1",
    port: int = 8766,
    open_browser: bool = True,
) -> Optional[np.ndarray]:
    """
    左键 FG、右键 BG、撤销、保存。predict_fn(pts (N,2) float32, labels (N,) int32) -> mask uint8.
    """
    import cv2

    base_url = _image_to_data_url(image_rgb)
    cmd_q: queue.Queue = queue.Queue()
    lock = threading.Lock()
    state = {"overlay_b64": base_url, "status": "左=前景 右=背景", "done": False, "saved": False}

    html = (
        "<!DOCTYPE html><html lang=zh-CN><head><meta charset=UTF-8><title>SAM</title>"
        "<style>body{font-family:system-ui;max-width:1000px;margin:1rem auto}"
        "#v{display:inline-block;border:2px solid #334}#v img{max-width:100%;display:block}"
        ".t{margin:.75rem 0}button{margin-right:.5rem;padding:.4rem 1rem;cursor:pointer}</style></head><body>"
        "<h1>SAM 点选</h1><div class=t><button id=u>撤销</button><button id=s>保存并关闭</button>"
        "<button id=x>取消</button></div><p id=st></p><div id=v><img id=im src='"
        + base_url
        + "'/></div>"
        "<script>"
        "const im=document.getElementById('im'),st=document.getElementById('st');"
        "function poll(){fetch('/api/state').then(r=>r.json()).then(j=>{im.src=j.overlay_b64;st.textContent=j.status||'';"
        "if(j.done)document.body.innerHTML='<p>'+(j.saved?'已保存':'已取消')+'</p>';});}"
        "setInterval(poll,250);poll();"
        "im.oncontextmenu=e=>e.preventDefault();"
        "im.onmousedown=function(ev){if(ev.button!==0&&ev.button!==2)return;const r=im.getBoundingClientRect();"
        "const nx=ev.clientX-r.left,ny=ev.clientY-r.top;const sx=im.naturalWidth/im.clientWidth,sy=im.naturalHeight/im.clientHeight;"
        "fetch('/api/click',{method:'POST',headers:{'Content-Type':'application/json'},"
        "body:JSON.stringify({x:nx*sx,y:ny*sy,label:ev.button===2?0:1})});};"
        "document.getElementById('u').onclick=()=>fetch('/api/undo',{method:'POST'});"
        "document.getElementById('s').onclick=()=>fetch('/api/save',{method:'POST'});"
        "document.getElementById('x').onclick=()=>fetch('/api/cancel',{method:'POST'});"
        "</script></body></html>"
    )

    class H(BaseHTTPRequestHandler):
        def log_message(self, format, *args):  # noqa: A002
            pass

        def do_GET(self):
            if self.path in ("/", "/?"):
                b = html.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(b)))
                self.end_headers()
                self.wfile.write(b)
            elif self.path == "/api/state":
                with lock:
                    out = json.dumps(
                        {
                            "overlay_b64": state["overlay_b64"],
                            "status": state["status"],
                            "done": state["done"],
                            "saved": state["saved"],
                        },
                        ensure_ascii=False,
                    ).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(out)))
                self.end_headers()
                self.wfile.write(out)
            else:
                self.send_error(404)

        def do_POST(self):
            n = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(n) if n else b"{}"
            cmd_q.put((self.path.split("?")[0], raw))
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"{}")

    def overlay_vis(img: np.ndarray, mask: Optional[np.ndarray], pts: list, labs: list) -> str:
        vis = np.asarray(img).copy()
        if vis.dtype != np.uint8:
            vis = (vis * 255).astype(np.uint8) if vis.max() <= 1.0 else vis.astype(np.uint8)
        base = vis.copy()
        if mask is not None and mask.size:
            m = mask > 0
            g = np.zeros_like(vis)
            g[..., 1] = 255
            vis = (base.astype(np.float32) * 0.55 + g.astype(np.float32) * 0.45).astype(np.uint8)
            vis = np.where(m[..., None], vis, base)
        for (px, py), lb in zip(pts, labs):
            col = (255, 0, 0) if lb == 1 else (0, 0, 255)
            cv2.circle(vis, (int(px), int(py)), 6, col, -1)
            cv2.circle(vis, (int(px), int(py)), 6, (255, 255, 255), 1)
        return _image_to_data_url(vis)

    srv = HTTPServer((host, port), H)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    url = f"http://{host}:{port}/"
    print(f"[web] SAM 交互: {url}")
    if open_browser:
        webbrowser.open(url)

    points: list = []
    labels: list = []
    mask_out: Optional[np.ndarray] = None
    finished = False

    while not finished:
        try:
            path, raw = cmd_q.get(timeout=0.15)
        except queue.Empty:
            continue
        if path == "/api/click":
            d = json.loads(raw.decode("utf-8"))
            points.append([float(d["x"]), float(d["y"])])
            labels.append(int(d.get("label", 1)))
            pts = np.array(points, dtype=np.float32)
            lbs = np.array(labels, dtype=np.int32)
            m = predict_fn(pts, lbs) if points else None
            with lock:
                state["overlay_b64"] = overlay_vis(image_rgb, m, points, labels)
                state["status"] = f"点数 {len(points)}"
        elif path == "/api/undo":
            if points:
                points.pop()
                labels.pop()
            pts = np.array(points, dtype=np.float32)
            lbs = np.array(labels, dtype=np.int32)
            m = predict_fn(pts, lbs) if points else None
            with lock:
                state["overlay_b64"] = overlay_vis(image_rgb, m, points, labels)
                state["status"] = f"撤销后 {len(points)} 点"
        elif path == "/api/save":
            if not points:
                with lock:
                    state["status"] = "请先点击"
                continue
            pts = np.array(points, dtype=np.float32)
            lbs = np.array(labels, dtype=np.int32)
            mask_out = predict_fn(pts, lbs)
            with lock:
                state["overlay_b64"] = overlay_vis(image_rgb, mask_out, points, labels)
                state["done"] = True
                state["saved"] = True
                state["status"] = "已保存"
            finished = True
        elif path == "/api/cancel":
            with lock:
                state["done"] = True
                state["saved"] = False
            mask_out = None
            finished = True

    srv.shutdown()
    th.join(timeout=2)
    return mask_out


def _image_to_thumb_url(image_rgb: np.ndarray, max_w: int = 120) -> str:
    """生成紧凑 JPEG 缩略图 data URL。"""
    arr = np.asarray(image_rgb)
    if arr.dtype != np.uint8:
        arr = (arr * 255).astype(np.uint8) if arr.max() <= 1.0 else arr.astype(np.uint8)
    if arr.ndim == 2:
        pil = Image.fromarray(arr, mode="L").convert("RGB")
    else:
        pil = Image.fromarray(arr)
    w, h = pil.size
    if w > max_w:
        pil = pil.resize((max_w, int(h * max_w / w)), Image.LANCZOS)
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=55)
    b64 = base64.standard_b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def pick_frames_web(
    frames: np.ndarray,
    default_start: int = 0,
    default_end: Optional[int] = None,
    default_mask_ref: Optional[int] = None,
    host: str = "127.0.0.1",
    port: int = 8767,
    open_browser: bool = True,
) -> Tuple[int, int, int]:
    """浏览器帧选择页面：选 start / end / mask 参考帧，返回 (start, end, mask_ref)。"""
    n = len(frames)
    if default_end is None:
        default_end = n - 1
    if default_mask_ref is None:
        default_mask_ref = default_start

    print(f"[web] 生成 {n} 帧缩略图 …")
    thumbs_json = json.dumps([_image_to_thumb_url(f) for f in frames])

    result: dict = {"val": None, "err": None}
    done = threading.Event()

    html = (
        "<!DOCTYPE html><html lang=zh-CN><head><meta charset=UTF-8><title>帧选择</title>"
        "<style>"
        "body{font-family:system-ui;max-width:1200px;margin:1rem auto;padding:0 1rem}"
        ".bar{display:flex;align-items:center;gap:8px;margin:10px 0;flex-wrap:wrap}"
        ".bar button{padding:6px 16px;border:2px solid #999;border-radius:4px;cursor:pointer;background:#f5f5f5}"
        ".bar button.a{background:#334;color:#fff;border-color:#334}"
        ".grid{display:flex;flex-wrap:wrap;gap:4px;max-height:60vh;overflow-y:auto;padding:4px;"
        "border:1px solid #ddd;border-radius:4px}"
        ".t{position:relative;cursor:pointer;border:3px solid transparent;border-radius:4px}"
        ".t:hover{opacity:.85}.t img{display:block}"
        ".t span{position:absolute;bottom:0;right:0;background:rgba(0,0,0,.65);color:#fff;"
        "font-size:10px;padding:1px 4px;border-radius:2px 0 0 0}"
        ".t.ir{background:rgba(100,200,100,.15)}"
        ".t.ss{border-color:#0a0}.t.se{border-color:#c00}.t.sm{border-color:#06f}"
        "#info{font-size:15px;font-weight:bold;margin:8px 0}"
        "#ok{padding:8px 24px;font-size:15px;cursor:pointer;margin-top:8px}"
        ".lg{display:flex;gap:12px;font-size:13px;margin:4px 0}"
        ".lg span{display:inline-flex;align-items:center;gap:4px}"
        ".lg i{display:inline-block;width:14px;height:14px;border:2px solid;border-radius:3px}"
        "</style></head><body>"
        "<h2>帧选择</h2><p>点击缩略图选择帧。先选模式，再点击对应帧。</p>"
        "<div class=bar><span>选择模式：</span>"
        "<button class=a data-m=start onclick=\"sm('start')\">起始帧</button>"
        "<button data-m=end onclick=\"sm('end')\">结束帧</button>"
        "<button data-m=mask onclick=\"sm('mask')\">mask参考帧</button></div>"
        "<div class=lg>"
        "<span><i style='border-color:#0a0'></i>起始帧</span>"
        "<span><i style='border-color:#c00'></i>结束帧</span>"
        "<span><i style='border-color:#06f'></i>mask参考帧</span>"
        "<span><i style='border-color:transparent;background:rgba(100,200,100,.25)'></i>范围内</span>"
        "</div>"
        "<p id=info></p><div class=grid id=grid></div>"
        "<button id=ok onclick=submit()>确认并关闭</button>"
        "<script>"
        "var T=" + thumbs_json + ";"
        "var M='start',S={start:" + str(default_start) + ",end:" + str(default_end)
        + ",mask:" + str(default_mask_ref) + "};"
        "var G=document.getElementById('grid');"
        "function render(){"
        "G.innerHTML='';"
        "for(var i=0;i<T.length;i++){"
        "var d=document.createElement('div');d.className='t';"
        "if(i>=S.start&&i<=S.end)d.classList.add('ir');"
        "if(i===S.start)d.classList.add('ss');"
        "if(i===S.end)d.classList.add('se');"
        "if(i===S.mask)d.classList.add('sm');"
        "var im=document.createElement('img');im.src=T[i];d.appendChild(im);"
        "var sp=document.createElement('span');sp.textContent=i;d.appendChild(sp);"
        "d.dataset.idx=i;d.onclick=function(){pick(parseInt(this.dataset.idx))};"
        "G.appendChild(d);}"
        "document.getElementById('info').textContent="
        "'起始帧: '+S.start+' | 结束帧: '+S.end+' | mask参考帧: '+S.mask"
        "+' | 片段长度: '+(S.end-S.start+1);}"
        "function pick(i){"
        "if(M==='start'){S.start=i;if(S.end<i)S.end=i;"
        "if(S.mask<S.start||S.mask>S.end)S.mask=S.start;}"
        "else if(M==='end'){S.end=i;if(S.start>i)S.start=i;"
        "if(S.mask<S.start||S.mask>S.end)S.mask=S.start;}"
        "else{if(i<S.start||i>S.end){alert('mask参考帧须在起始帧与结束帧之间');return;}"
        "S.mask=i;}"
        "render();}"
        "function sm(m){M=m;document.querySelectorAll('.bar button[data-m]')"
        ".forEach(function(b){b.classList.toggle('a',b.dataset.m===m)});}"
        "function submit(){"
        "fetch('/done',{method:'POST',headers:{'Content-Type':'application/json'},"
        "body:JSON.stringify(S)}).then(function(){"
        "document.body.innerHTML='<p>已提交，可关闭此页面。</p>';});}"
        "render();"
        "</script></body></html>"
    )

    class H(BaseHTTPRequestHandler):
        def log_message(self, fmt, *a):
            pass

        def do_GET(self):
            if self.path in ("/", "/?"):
                b = html.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(b)))
                self.end_headers()
                self.wfile.write(b)
            else:
                self.send_error(404)

        def do_POST(self):
            if self.path != "/done":
                self.send_error(404)
                return
            ln = int(self.headers.get("Content-Length", "0"))
            try:
                d = json.loads(self.rfile.read(ln).decode("utf-8"))
                result["val"] = (int(d["start"]), int(d["end"]), int(d["mask"]))
            except Exception as e:
                result["err"] = str(e)
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"{}")
            done.set()

    srv = HTTPServer((host, port), H)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    url = f"http://{host}:{port}/"
    print(f"[web] 帧选择: {url}")
    if open_browser:
        webbrowser.open(url)
    done.wait(timeout=3600)
    srv.shutdown()
    th.join(timeout=2)
    if result.get("err"):
        raise RuntimeError(result["err"])
    if result["val"] is None:
        raise RuntimeError("未收到帧选择结果（超时或未确认）")
    return result["val"]


def run_unified_web(
    frames: np.ndarray,
    predict_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    set_image_fn: Callable[[Image.Image], None],
    default_start: int = 0,
    default_end: Optional[int] = None,
    default_mask_ref: Optional[int] = None,
    default_mask_refs: Optional[list] = None,
    default_tracking: Optional[Tuple[float, float]] = None,
    default_mask: Optional[np.ndarray] = None,
    default_masks: Optional[list] = None,
    host: str = "127.0.0.1",
    port: int = 8765,
    open_browser: bool = True,
    pipeline_fn: Optional[Callable] = None,
    robot_types: Optional[list] = None,
    default_robot_type: str = "",
    datasets_info: Optional[list] = None,
    default_dataset_idx: int = 0,
) -> dict:
    """
    统一 Web 交互页面：选帧 + 选追踪点 + 选 mask，三步同页。
    支持多帧 mask（多个 mask 参考帧，每帧独立 SAM 交互）。
    支持数据集选择（datasets_info 提供可选数据集列表）。

    predict_fn(pts (N,2) float32, labels (N,) int32) -> mask uint8
    set_image_fn(PIL.Image) -> None  更新 SAM 当前图片

    Returns dict: start, end, mask_refs, tracking_point, masks
    """
    import cv2
    import os

    n = len(frames)
    if default_end is None:
        default_end = n - 1

    # ── 兼容旧接口：单帧 → list ──
    if default_mask_refs is None:
        if default_mask_ref is not None:
            default_mask_refs = [default_mask_ref]
        else:
            default_mask_refs = [default_start]
    if default_masks is None and default_mask is not None:
        default_masks = [default_mask]

    has_def_t = default_tracking is not None
    has_def_m = default_masks is not None and len(default_masks) > 0

    # ── 缩略图 ──
    print(f"[web] 生成 {n} 帧缩略图 …")
    thumbs_json = json.dumps([_image_to_thumb_url(f) for f in frames])

    # ── 初始图片 ──
    tracking_url = _image_to_data_url(frames[default_start])
    first_mask_ref = default_mask_refs[0]
    if has_def_m:
        vis = frames[first_mask_ref].copy()
        base = vis.copy()
        m = default_masks[0] > 0
        g = np.zeros_like(vis)
        g[..., 1] = 255
        bl = (base.astype(np.float32) * 0.55 + g.astype(np.float32) * 0.45).astype(np.uint8)
        vis = np.where(m[..., None], bl, base)
        mask_url = _image_to_data_url(vis)
    else:
        mask_url = _image_to_data_url(frames[first_mask_ref])

    # ── 预设 SAM ──
    set_image_fn(Image.fromarray(frames[first_mask_ref]))
    print("[web] SAM image 已预设")

    # ── 多帧 SAM 状态 (per-frame) ──
    lock = threading.Lock()
    shared: dict = {"overlay": mask_url, "status": "左键=前景 右键=背景", "pts": 0, "cur_mask_idx": 0}
    cmd_q: queue.Queue = queue.Queue()
    result: dict = {"val": None}
    finished = threading.Event()

    # 每帧独立的 SAM 状态
    per_frame_sam: dict = {}  # {frame_idx: {"points": [], "labels": [], "mask": None}}

    def _ov(img, msk, pts, labs):
        v = np.asarray(img).copy()
        if v.dtype != np.uint8:
            v = (v * 255).astype(np.uint8) if v.max() <= 1.0 else v.astype(np.uint8)
        b = v.copy()
        if msk is not None and msk.size:
            mm = msk > 0
            gg = np.zeros_like(v)
            gg[..., 1] = 255
            v = (b.astype(np.float32) * 0.55 + gg.astype(np.float32) * 0.45).astype(np.uint8)
            v = np.where(mm[..., None], v, b)
        for (px, py), lb in zip(pts, labs):
            c = (255, 0, 0) if lb == 1 else (0, 0, 255)
            cv2.circle(v, (int(px), int(py)), 6, c, -1)
            cv2.circle(v, (int(px), int(py)), 6, (255, 255, 255), 1)
        return _image_to_data_url(v)

    # ── HTML ──
    dt_x = str(default_tracking[0]) if has_def_t else "null"
    dt_y = str(default_tracking[1]) if has_def_t else "null"
    hdt = "true" if has_def_t else "false"
    hdm = "true" if has_def_m else "false"
    hp = "true" if pipeline_fn is not None else "false"
    init_ms_json = json.dumps(default_mask_refs)

    if robot_types:
        _opts = "".join(
            "<option value='" + rt + "'" + (" selected" if rt == default_robot_type else "") + ">" + rt + "</option>"
            for rt in robot_types
        )
        robot_html = ("<div style='margin-bottom:12px'>"
                       "<label style='font-size:14px'>机型: "
                       "<select id=rt style='padding:4px 8px;font-size:14px'>" + _opts + "</select>"
                       "</label></div>")
    else:
        robot_html = ""

    # 数据集选择 HTML
    if datasets_info:
        ds_opts = "".join(
            "<option value='" + str(i) + "'" + (" selected" if i == default_dataset_idx else "") + ">"
            + d.get("display_name", d.get("task_name", str(i))) + "</option>"
            for i, d in enumerate(datasets_info)
        )
        cam_opts = ""
        if datasets_info and default_dataset_idx < len(datasets_info):
            cams = datasets_info[default_dataset_idx].get("cameras", [])
            cam_opts = "".join("<option value='" + c + "'>" + c + "</option>" for c in cams)
        dataset_html = (
            "<div style='margin-bottom:12px;display:flex;gap:16px;align-items:center;flex-wrap:wrap'>"
            "<label style='font-size:14px'>数据集: "
            "<select id=ds style='padding:4px 8px;font-size:14px' onchange='switchDs()'>" + ds_opts + "</select>"
            "</label>"
            "<label style='font-size:14px'>Camera: "
            "<select id=cam style='padding:4px 8px;font-size:14px'>" + cam_opts + "</select>"
            "</label>"
            "<button onclick='loadDs()' style='padding:4px 12px;font-size:14px'>加载</button>"
            "<span id=ds-status style='font-size:13px;color:#666'></span>"
            "</div>"
        )
    else:
        dataset_html = ""

    html = (
        "<!DOCTYPE html><html lang=zh-CN><head><meta charset=UTF-8>"
        "<title>外参检测 - 交互标注</title><style>"
        "*{box-sizing:border-box}"
        "body{font-family:system-ui;max-width:1100px;margin:0 auto;padding:1rem}"
        ".step{border:1px solid #ddd;border-radius:6px;margin-bottom:12px;overflow:hidden}"
        ".sh{display:flex;justify-content:space-between;align-items:center;padding:8px 12px;"
        "background:#f5f5f5;border-bottom:1px solid #ddd}"
        ".sh h3{margin:0;font-size:15px}"
        ".tg{font-size:13px;cursor:pointer;user-select:none}"
        ".sb{padding:10px 12px}.sb.off{opacity:.35;pointer-events:none}"
        ".si{padding:4px 12px;font-size:13px;color:#555;background:#fafafa}"
        ".grid{display:flex;flex-wrap:wrap;gap:3px;max-height:40vh;overflow-y:auto}"
        ".t{position:relative;cursor:pointer;border:3px solid transparent;border-radius:3px}"
        ".t img{display:block}.t span{position:absolute;bottom:0;right:0;"
        "background:rgba(0,0,0,.6);color:#fff;font-size:9px;padding:1px 3px}"
        ".t.ir{background:rgba(100,200,100,.15)}"
        ".t.ss{border-color:#0a0}.t.se{border-color:#c00}.t.sm{border-color:#06f}"
        ".mb{margin-bottom:6px}.mb button{padding:4px 12px;border:1px solid #999;"
        "border-radius:3px;cursor:pointer;background:#fff;margin-right:4px}"
        ".mb button.a{background:#334;color:#fff}"
        ".lg{display:flex;gap:10px;font-size:12px;margin-bottom:4px}"
        ".lg span{display:inline-flex;align-items:center;gap:3px}"
        ".lg i{display:inline-block;width:12px;height:12px;border:2px solid;border-radius:2px}"
        "#tw{position:relative;display:inline-block;cursor:crosshair}"
        "#tw img{display:block;max-width:100%;max-height:40vh}"
        ".dot{position:absolute;width:14px;height:14px;margin:-7px 0 0 -7px;"
        "border:2px solid #fff;border-radius:50%;background:#e11;"
        "box-shadow:0 0 0 1px #0008;pointer-events:none}"
        "#mw{display:inline-block;cursor:crosshair}#mw img{display:block;max-width:100%;max-height:40vh}"
        ".sb2{margin-bottom:6px;font-size:13px}"
        ".sb2 button{padding:3px 10px;cursor:pointer;margin-right:4px}"
        "#sub{display:block;margin:16px auto;padding:10px 32px;font-size:16px;"
        "cursor:pointer;background:#334;color:#fff;border:none;border-radius:6px}"
        "#pipeline{display:none}"
        ".ps{padding:6px 10px;margin:2px 0;font-size:14px;border-radius:4px}"
        ".ps.active{background:#e8f4e8;font-weight:bold}"
        ".ps.done{color:#0a0}"
        "#fin-btn{display:none;margin:16px auto;padding:10px 32px;font-size:16px;"
        "cursor:pointer;background:#0a0;color:#fff;border:none;border-radius:6px}"
        ".mtabs{display:flex;gap:4px;margin-bottom:8px;flex-wrap:wrap}"
        ".mtab{padding:4px 12px;border:2px solid #06f;border-radius:4px;cursor:pointer;"
        "background:#fff;font-size:13px}"
        ".mtab.a{background:#06f;color:#fff}"
        ".mtab .x{margin-left:6px;color:#c00;font-weight:bold}"
        "</style></head><body>"
        "<h2>外参检测 - 交互标注</h2>"
        + dataset_html + robot_html +
        # ── Step 1 ──
        "<section class=step>"
        "<div class=sh><h3>1. 帧选择</h3>"
        "<label class=tg><input type=checkbox id=fa onchange=\"tog('f')\"> 使用默认</label></div>"
        "<div class=si id=fi></div>"
        "<div class='sb' id=fb>"
        "<div class=mb id=fmb>"
        "<button class=a data-m=start onclick=\"sfm('start')\">起始帧</button>"
        "<button data-m=end onclick=\"sfm('end')\">结束帧</button>"
        "<button data-m=mask onclick=\"sfm('mask')\">mask参考帧(多选)</button></div>"
        "<div class=lg>"
        "<span><i style='border-color:#0a0'></i>起始</span>"
        "<span><i style='border-color:#c00'></i>结束</span>"
        "<span><i style='border-color:#06f'></i>mask帧</span></div>"
        "<div class=grid id=fg></div></div></section>"
        # ── Step 2 ──
        "<section class=step>"
        "<div class=sh><h3>2. 追踪点</h3>"
        "<label class=tg id=tl><input type=checkbox id=ta "
        + ("checked" if has_def_t else "")
        + " onchange=\"tog('t')\"> 使用默认</label></div>"
        "<div class=si id=ti></div>"
        "<div class='sb" + (" off" if has_def_t else "") + "' id=tb>"
        "<div id=tw><img id=timg src='" + tracking_url + "'/>"
        "<div class=dot id=td style=display:none></div></div></div></section>"
        # ── Step 3: Multi-mask ──
        "<section class=step>"
        "<div class=sh><h3>3. Mask（多帧）</h3>"
        "<label class=tg id=ml><input type=checkbox id=mau "
        + ("checked" if has_def_m else "")
        + " onchange=\"tog('m')\"> 使用默认</label></div>"
        "<div class=si id=mi></div>"
        "<div class='sb" + (" off" if has_def_m else "") + "' id=mmb>"
        "<div class=mtabs id=mtabs></div>"
        "<div class=sb2><button onclick=samUndo()>撤销</button>"
        "<span id=ms>左键=前景 右键=背景</span></div>"
        "<div id=mw><img id=mimg src='" + mask_url + "'/></div></div></section>"
        "<button id=sub onclick=go()>确认并保存</button>"
        "<section class=step id=pipeline>"
        "<div class=sh><h3>Pipeline</h3></div>"
        "<div class=sb>"
        "<div id=p-stages>"
        "<div class=ps id=ps-tracking>&#9203; Tracking</div>"
        "<div class=ps id=ps-coarse>&#9203; Coarse Init (PnP)</div>"
        "<div class=ps id=ps-refine>&#9203; Refinement <span id=p-step></span></div>"
        "</div>"
        "<p id=p-msg style='margin:8px 0;font-size:14px'>等待标注完成...</p>"
        "<div class=mtabs id=p-mtabs style='display:none'></div>"
        "<img id=p-img style='max-width:100%;max-height:50vh;display:none;margin-top:8px'/>"
        "</div></section>"
        "<button id=fin-btn onclick=fin()>退出</button>"
        "<button id=restart-btn style='display:none;margin:16px auto;padding:10px 32px;font-size:16px;"
        "cursor:pointer;background:#06f;color:#fff;border:none;border-radius:6px' onclick=restart()>重新标注</button>"
        # ── JS ──
        "<script>"
        "var T=" + thumbs_json + ";"
        "var S={s:" + str(default_start) + ",e:" + str(default_end)
        + ",ms:" + init_ms_json
        + ",curMI:0"
        + ",tx:" + dt_x + ",ty:" + dt_y
        + ",fm:'start',fa:false"
        + ",ta:" + hdt + ",ma:" + hdm + "};"
        "var HDT=" + hdt + ",HDM=" + hdm + ",HP=" + hp + ";"
        # Per-frame SAM pts tracker (client side)
        "var samPtsMap={};"
        "S.ms.forEach(function(m){samPtsMap[m]=0;});"
        # Frame grid
        "function renderGrid(){"
        "var g=document.getElementById('fg');g.innerHTML='';"
        "for(var i=0;i<T.length;i++){"
        "var d=document.createElement('div');d.className='t';"
        "if(i>=S.s&&i<=S.e)d.classList.add('ir');"
        "if(i===S.s)d.classList.add('ss');"
        "if(i===S.e)d.classList.add('se');"
        "if(S.ms.indexOf(i)>=0)d.classList.add('sm');"
        "var im=document.createElement('img');im.src=T[i];d.appendChild(im);"
        "var sp=document.createElement('span');sp.textContent=i;d.appendChild(sp);"
        "d.dataset.idx=i;d.onclick=function(){pickF(parseInt(this.dataset.idx))};"
        "g.appendChild(d);}}"
        # Pick frame
        "function pickF(i){"
        "if(S.fm==='start'){S.s=i;if(S.e<i)S.e=i;"
        "S.ms=S.ms.filter(function(m){return m>=S.s&&m<=S.e;});"
        "if(S.ms.length===0)S.ms=[S.s];}"
        "else if(S.fm==='end'){S.e=i;if(S.s>i)S.s=i;"
        "S.ms=S.ms.filter(function(m){return m>=S.s&&m<=S.e;});"
        "if(S.ms.length===0)S.ms=[S.s];}"
        "else{"  # mask mode: toggle
        "if(i<S.s||i>S.e){alert('mask参考帧须在范围内');return;}"
        "var idx=S.ms.indexOf(i);"
        "if(idx>=0){if(S.ms.length>1)S.ms.splice(idx,1);}"
        "else{S.ms.push(i);S.ms.sort(function(a,b){return a-b;});}"
        "if(!(i in samPtsMap))samPtsMap[i]=0;}"
        "S.curMI=0;"
        "renderGrid();renderMTabs();upInfo();upTImg();"
        "if(!S.ma)switchMask(0);}"
        # Frame mode
        "function sfm(m){S.fm=m;"
        "document.querySelectorAll('#fmb button[data-m]')"
        ".forEach(function(b){b.classList.toggle('a',b.dataset.m===m)});}"
        # Mask tabs
        "function renderMTabs(){"
        "var c=document.getElementById('mtabs');c.innerHTML='';"
        "S.ms.forEach(function(m,i){"
        "var b=document.createElement('div');b.className='mtab'+(i===S.curMI?' a':'');"
        "b.textContent='帧 '+m;"
        "var pts=samPtsMap[m]||0;"
        "if(pts>0){var sp=document.createElement('span');sp.textContent=' ('+pts+'点)';sp.style.fontSize='11px';b.appendChild(sp);}"
        "b.onclick=function(){switchMask(i);};"
        "c.appendChild(b);});}"
        # Switch mask tab
        "function switchMask(idx){"
        "S.curMI=idx;"
        "renderMTabs();"
        "fetch('/api/sam/switch_frame',{method:'POST',headers:{'Content-Type':'application/json'},"
        "body:JSON.stringify({fi:S.ms[idx]})});}"
        # Toggle
        "function tog(w){"
        "if(w==='f')S.fa=document.getElementById('fa').checked;"
        "else if(w==='t')S.ta=document.getElementById('ta').checked;"
        "else{S.ma=document.getElementById('mau').checked;if(!S.ma)switchMask(S.curMI);}"
        "document.getElementById(w==='f'?'fb':w==='t'?'tb':'mmb')"
        ".classList.toggle('off',w==='f'?S.fa:w==='t'?S.ta:S.ma);upInfo();}"
        # Tracking click
        "document.getElementById('tw').onclick=function(ev){"
        "var img=document.getElementById('timg');"
        "var r=img.getBoundingClientRect();"
        "var sx=img.naturalWidth/img.clientWidth,sy=img.naturalHeight/img.clientHeight;"
        "S.tx=(ev.clientX-r.left)*sx;S.ty=(ev.clientY-r.top)*sy;"
        "upDot();upInfo();};"
        # Update dot
        "function upDot(){"
        "var img=document.getElementById('timg'),dot=document.getElementById('td');"
        "if(S.tx===null){dot.style.display='none';return;}"
        "dot.style.left=(S.tx*img.clientWidth/img.naturalWidth)+'px';"
        "dot.style.top=(S.ty*img.clientHeight/img.naturalHeight)+'px';"
        "dot.style.display='block';}"
        # Update tracking image
        "function upTImg(){"
        "fetch('/api/frame/'+S.s).then(function(r){return r.json()}).then(function(d){"
        "var img=document.getElementById('timg');"
        "img.onload=function(){upDot()};img.src=d.url;});}"
        # SAM click
        "document.getElementById('mw').oncontextmenu=function(e){e.preventDefault()};"
        "document.getElementById('mw').onmousedown=function(ev){"
        "if(ev.button!==0&&ev.button!==2)return;"
        "var img=document.getElementById('mimg');"
        "var r=img.getBoundingClientRect();"
        "var sx=img.naturalWidth/img.clientWidth,sy=img.naturalHeight/img.clientHeight;"
        "fetch('/api/sam/click',{method:'POST',headers:{'Content-Type':'application/json'},"
        "body:JSON.stringify({x:(ev.clientX-r.left)*sx,y:(ev.clientY-r.top)*sy,"
        "label:ev.button===2?0:1,fi:S.ms[S.curMI]})});};"
        # SAM undo
        "function samUndo(){"
        "fetch('/api/sam/undo',{method:'POST',headers:{'Content-Type':'application/json'},"
        "body:JSON.stringify({fi:S.ms[S.curMI]})});}"
        # Poll SAM
        "function pollSam(){"
        "if(S.ma)return;"
        "fetch('/api/sam/state').then(function(r){return r.json()}).then(function(d){"
        "document.getElementById('mimg').src=d.overlay;"
        "document.getElementById('ms').textContent=d.status;"
        "samPtsMap[S.ms[S.curMI]]=d.pts||0;renderMTabs();upInfo();});}"
        "var samTimer=setInterval(pollSam,300);"
        # Info
        "function upInfo(){"
        "var totalPts=0;S.ms.forEach(function(m){totalPts+=(samPtsMap[m]||0);});"
        "document.getElementById('fi').textContent="
        "'起始帧: '+S.s+' | 结束帧: '+S.e+' | mask帧: ['+S.ms.join(',')+'] | 长度: '+(S.e-S.s+1);"
        "document.getElementById('ti').textContent="
        "S.tx!==null?'追踪点: ('+S.tx.toFixed(1)+', '+S.ty.toFixed(1)+')':'追踪点: 未选择';"
        "document.getElementById('mi').textContent="
        "S.ma?'mask: 使用默认 ('+S.ms.length+'帧)':'mask: SAM '+S.ms.length+'帧, 共'+totalPts+'点';}"
        # Dataset switch
        "function switchDs(){"
        "var sel=document.getElementById('ds');"
        "fetch('/api/dataset/cameras?idx='+sel.value).then(function(r){return r.json()}).then(function(d){"
        "var c=document.getElementById('cam');c.innerHTML='';"
        "d.cameras.forEach(function(n){var o=document.createElement('option');o.value=n;o.textContent=n;c.appendChild(o);});});}"
        "function loadDs(){"
        "var di=document.getElementById('ds').value;"
        "var cn=document.getElementById('cam').value;"
        "document.getElementById('ds-status').textContent='加载中…';"
        "fetch('/api/dataset/load',{method:'POST',headers:{'Content-Type':'application/json'},"
        "body:JSON.stringify({dataset_idx:parseInt(di),camera:cn})})"
        ".then(function(r){return r.json()}).then(function(d){"
        "if(d.ok){document.getElementById('ds-status').textContent='已加载 '+d.n_frames+' 帧';"
        "T=d.thumbs;S.s=0;S.e=d.n_frames-1;S.ms=[0];S.curMI=0;samPtsMap={0:0};"
        "renderGrid();renderMTabs();upInfo();upTImg();}"
        "else{document.getElementById('ds-status').textContent='加载失败: '+(d.error||'');}});}"
        # Submit
        "function go(){"
        "if(S.tx===null||S.ty===null){alert('请选择追踪点');return;}"
        "if(!S.ma){"
        "var missing=[];S.ms.forEach(function(m){if(!samPtsMap[m]||samPtsMap[m]===0)missing.push(m);});"
        "if(missing.length>0){alert('以下mask帧尚未标注: '+missing.join(', '));return;}}"
        "fetch('/api/done',{method:'POST',headers:{'Content-Type':'application/json'},"
        "body:JSON.stringify({start:S.s,end:S.e,maskRefs:S.ms,"
        "trackingX:S.tx,trackingY:S.ty,maskAuto:S.ma,"
        "robotType:document.getElementById('rt')?document.getElementById('rt').value:''})})"
        ".then(function(){if(HP){"
        "clearInterval(samTimer);"
        "document.querySelectorAll('.step').forEach(function(s){s.style.display='none'});"
        "document.getElementById('sub').style.display='none';"
        "document.getElementById('pipeline').style.display='block';"
        "pipeTimer=setInterval(pollPipe,1000);"
        "}else{document.body.innerHTML="
        "'<p style=\"text-align:center;margin-top:3rem;font-size:18px\">已提交，可关闭此页面。</p>';}});}"
        # Init
        "renderGrid();renderMTabs();upInfo();upDot();"
        "if(!HDT){document.getElementById('tl').style.display='none';"
        "document.getElementById('tb').classList.remove('off');}"
        "if(!HDM){document.getElementById('ml').style.display='none';"
        "document.getElementById('mmb').classList.remove('off');switchMask(0);}"
        # Pipeline JS
        "var pipeTimer,pOverlays=[],pMaskIds=[],pCurOv=0;"
        "function renderPTabs(){"
        "var c=document.getElementById('p-mtabs');"
        "if(pMaskIds.length<=1){c.style.display='none';return;}"
        "c.style.display='flex';c.innerHTML='';"
        "pMaskIds.forEach(function(m,i){"
        "var b=document.createElement('div');b.className='mtab'+(i===pCurOv?' a':'');"
        "b.textContent='帧 '+m;"
        "b.onclick=function(){pCurOv=i;renderPTabs();"
        "if(pOverlays[i]){var img=document.getElementById('p-img');img.src=pOverlays[i];img.style.display='';}};"
        "c.appendChild(b);});}"
        "function pollPipe(){"
        "fetch('/api/pipeline/state').then(function(r){return r.json()}).then(function(d){"
        "if(!d||!d.stage)return;"
        "document.getElementById('p-msg').textContent=d.message||'';"
        "var ss=['tracking','coarse','refine'],ci=ss.indexOf(d.stage);"
        "ss.forEach(function(s,i){"
        "var el=document.getElementById('ps-'+s);"
        "if(d.stage==='done'||i<ci){el.className='ps done';}"
        "else if(i===ci){el.className='ps active';}"
        "else{el.className='ps';}});"
        "if(d.step!==undefined)document.getElementById('p-step').textContent='('+d.step+'/'+(d.max_steps||'?')+')';"
        "if(d.overlays&&d.overlays.length>0){"
        "pOverlays=d.overlays;"
        "if(d.mask_ids)pMaskIds=d.mask_ids;"
        "renderPTabs();"
        "var img=document.getElementById('p-img');img.src=pOverlays[pCurOv];img.style.display='';}"
        "else if(d.image){var img=document.getElementById('p-img');img.src=d.image;img.style.display='';}"
        "if(d.stage==='done'){clearInterval(pipeTimer);"
        "document.getElementById('fin-btn').style.display='block';"
        "document.getElementById('restart-btn').style.display='block';"
        "if(d.video_path){var v=document.createElement('video');"
        "v.src='/api/video';v.controls=true;v.autoplay=true;v.loop=true;"
        "v.style.maxWidth='100%';v.style.maxHeight='50vh';v.style.marginTop='8px';"
        "var c=document.getElementById('p-img').parentNode;c.appendChild(v);}}"
        "});}"
        "function fin(){"
        "fetch('/api/finish',{method:'POST'}).then(function(){"
        "document.body.innerHTML='<p style=\"text-align:center;margin-top:3rem;font-size:18px\">已完成</p>';});}"
        "function restart(){"
        "fetch('/api/restart',{method:'POST'}).then(function(){location.reload();});}"
        "</script></body></html>"
    )

    # ── HTTP handler ──
    class H(BaseHTTPRequestHandler):
        def log_message(self, fmt, *a):
            pass

        def do_GET(self):
            if self.path in ("/", "/?"):
                b = html.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(b)))
                self.end_headers()
                self.wfile.write(b)
            elif self.path.startswith("/api/frame/"):
                try:
                    idx = int(self.path.rsplit("/", 1)[-1])
                except ValueError:
                    self.send_error(400)
                    return
                if not (0 <= idx < n):
                    self.send_error(404)
                    return
                out = json.dumps({"url": _image_to_data_url(frames[idx])}).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(out)))
                self.end_headers()
                self.wfile.write(out)
            elif self.path == "/api/sam/state":
                with lock:
                    out = json.dumps({
                        "overlay": shared["overlay"],
                        "status": shared["status"],
                        "pts": shared["pts"],
                    }).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(out)))
                self.end_headers()
                self.wfile.write(out)
            elif self.path == "/api/pipeline/state":
                with lock:
                    ps = shared.get("pipeline", {})
                    out = json.dumps(ps, ensure_ascii=False).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(out)))
                self.end_headers()
                self.wfile.write(out)
            elif self.path == "/api/video":
                vpath = None
                with lock:
                    vpath = shared.get("pipeline", {}).get("video_path")
                if vpath and os.path.isfile(vpath):
                    sz = os.path.getsize(vpath)
                    self.send_response(200)
                    self.send_header("Content-Type", "video/mp4")
                    self.send_header("Content-Length", str(sz))
                    self.end_headers()
                    with open(vpath, "rb") as vf:
                        self.wfile.write(vf.read())
                else:
                    self.send_error(404)
            elif self.path.startswith("/api/dataset/cameras"):
                # 返回指定数据集的 camera 列表
                try:
                    qs = self.path.split("?", 1)[1] if "?" in self.path else ""
                    params = dict(p.split("=") for p in qs.split("&") if "=" in p)
                    idx = int(params.get("idx", "0"))
                    cams = datasets_info[idx].get("cameras", []) if datasets_info and idx < len(datasets_info) else []
                    out = json.dumps({"cameras": cams}).encode("utf-8")
                except Exception:
                    out = json.dumps({"cameras": []}).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(out)))
                self.end_headers()
                self.wfile.write(out)
            else:
                self.send_error(404)

        def do_POST(self):
            cl = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(cl) if cl else b"{}"
            p = self.path.split("?")[0]
            if p in ("/api/sam/click", "/api/sam/undo", "/api/sam/set_image",
                      "/api/sam/switch_frame", "/api/done", "/api/finish",
                      "/api/restart", "/api/dataset/load"):
                cmd_q.put((p, raw))
                if p == "/api/dataset/load":
                    # 数据集加载需要等待结果
                    try:
                        resp = shared.get("_ds_resp_q", queue.Queue()).get(timeout=60)
                    except queue.Empty:
                        resp = {"ok": False, "error": "timeout"}
                    out = json.dumps(resp).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(out)))
                    self.end_headers()
                    self.wfile.write(out)
                else:
                    self.send_response(200)
                    self.end_headers()
                    self.wfile.write(b"{}")
            else:
                self.send_error(404)

    # ── 当前活跃帧索引 ──
    active_frame_idx = first_mask_ref

    def _get_frame_sam(fi):
        """获取帧 fi 的 SAM 状态，不存在则初始化。"""
        if fi not in per_frame_sam:
            per_frame_sam[fi] = {"points": [], "labels": [], "mask": None}
        return per_frame_sam[fi]

    # ── Start server ──
    srv = HTTPServer((host, port), H)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    url = f"http://{host}:{port}/"
    print(f"[web] 统一交互页面: {url}")
    if open_browser:
        webbrowser.open(url)

    # ── Main loop (SAM 推理在主线程) ──
    session_done = False
    while not session_done:
        # Phase 1: 标注交互
        while True:
            try:
                path, raw = cmd_q.get(timeout=0.15)
            except queue.Empty:
                continue

            if path == "/api/sam/set_image":
                idx = int(json.loads(raw)["idx"])
                set_image_fn(Image.fromarray(frames[idx]))
                active_frame_idx = idx
                fs = _get_frame_sam(idx)
                fs["points"].clear()
                fs["labels"].clear()
                fs["mask"] = None
                with lock:
                    shared["overlay"] = _image_to_data_url(frames[idx])
                    shared["status"] = f"已切换到帧 {idx}"
                    shared["pts"] = 0

            elif path == "/api/sam/switch_frame":
                fi = int(json.loads(raw)["fi"])
                set_image_fn(Image.fromarray(frames[fi]))
                active_frame_idx = fi
                fs = _get_frame_sam(fi)
                with lock:
                    if fs["mask"] is not None:
                        shared["overlay"] = _ov(frames[fi], fs["mask"], fs["points"], fs["labels"])
                    else:
                        shared["overlay"] = _ov(frames[fi], None, fs["points"], fs["labels"])
                    shared["status"] = f"帧 {fi}: {len(fs['points'])} 点"
                    shared["pts"] = len(fs["points"])

            elif path == "/api/sam/click":
                d = json.loads(raw)
                fi = int(d.get("fi", active_frame_idx))
                if fi != active_frame_idx:
                    set_image_fn(Image.fromarray(frames[fi]))
                    active_frame_idx = fi
                fs = _get_frame_sam(fi)
                fs["points"].append([float(d["x"]), float(d["y"])])
                fs["labels"].append(int(d.get("label", 1)))
                pts = np.array(fs["points"], dtype=np.float32)
                lbs = np.array(fs["labels"], dtype=np.int32)
                fs["mask"] = predict_fn(pts, lbs)
                with lock:
                    shared["overlay"] = _ov(frames[fi], fs["mask"], fs["points"], fs["labels"])
                    shared["status"] = f"帧 {fi}: {len(fs['points'])} 点"
                    shared["pts"] = len(fs["points"])

            elif path == "/api/sam/undo":
                d = json.loads(raw) if raw else {}
                fi = int(d.get("fi", active_frame_idx))
                if fi != active_frame_idx:
                    set_image_fn(Image.fromarray(frames[fi]))
                    active_frame_idx = fi
                fs = _get_frame_sam(fi)
                if fs["points"]:
                    fs["points"].pop()
                    fs["labels"].pop()
                if fs["points"]:
                    pts = np.array(fs["points"], dtype=np.float32)
                    lbs = np.array(fs["labels"], dtype=np.int32)
                    fs["mask"] = predict_fn(pts, lbs)
                else:
                    fs["mask"] = None
                with lock:
                    shared["overlay"] = _ov(frames[fi], fs["mask"], fs["points"], fs["labels"])
                    shared["status"] = f"帧 {fi}: 撤销后 {len(fs['points'])} 点"
                    shared["pts"] = len(fs["points"])

            elif path == "/api/dataset/load":
                d = json.loads(raw)
                ds_resp_q = queue.Queue()
                with lock:
                    shared["_ds_resp_q"] = ds_resp_q
                try:
                    di = int(d["dataset_idx"])
                    cam = d.get("camera", "")
                    if datasets_info and di < len(datasets_info):
                        ds = datasets_info[di]
                        load_fn = ds.get("load_fn")
                        if load_fn:
                            new_frames = load_fn(cam)
                            if new_frames is not None:
                                frames = new_frames
                                n = len(frames)
                                new_thumbs = [_image_to_thumb_url(f) for f in frames]
                                thumbs_json = json.dumps(new_thumbs)
                                per_frame_sam.clear()
                                active_frame_idx = 0
                                set_image_fn(Image.fromarray(frames[0]))
                                with lock:
                                    shared["overlay"] = _image_to_data_url(frames[0])
                                    shared["status"] = "左键=前景 右键=背景"
                                    shared["pts"] = 0
                                ds_resp_q.put({"ok": True, "n_frames": n, "thumbs": new_thumbs})
                            else:
                                ds_resp_q.put({"ok": False, "error": "加载失败"})
                        else:
                            ds_resp_q.put({"ok": False, "error": "无 load_fn"})
                    else:
                        ds_resp_q.put({"ok": False, "error": "无效索引"})
                except Exception as exc:
                    ds_resp_q.put({"ok": False, "error": str(exc)})

            elif path == "/api/done":
                d = json.loads(raw)
                mask_auto = d.get("maskAuto", False)
                mask_refs = [int(x) for x in d.get("maskRefs", d.get("maskRef", []))]
                if isinstance(mask_refs, int):
                    mask_refs = [mask_refs]

                if mask_auto and default_masks is not None:
                    final_masks = [m.copy() for m in default_masks]
                else:
                    final_masks = []
                    for mfi in mask_refs:
                        fs = _get_frame_sam(mfi)
                        final_masks.append(fs["mask"])

                result["val"] = {
                    "start": int(d["start"]),
                    "end": int(d["end"]),
                    "mask_refs": mask_refs,
                    "mask_ref": mask_refs[0],  # 兼容旧接口
                    "tracking_point": (float(d["trackingX"]), float(d["trackingY"])),
                    "masks": final_masks,
                    "mask": final_masks[0] if final_masks else None,  # 兼容旧接口
                    "robot_type": d.get("robotType", default_robot_type),
                }
                break  # 退出标注循环

        # Phase 2: Pipeline（如有）
        if pipeline_fn is not None and result["val"] is not None:
            import gc
            gc.collect()

            def _update(state):
                if "image" in state and isinstance(state["image"], np.ndarray):
                    state["image"] = _image_to_data_url(state["image"])
                if "overlays" in state and isinstance(state["overlays"], list):
                    state["overlays"] = [
                        _image_to_data_url(ov) if isinstance(ov, np.ndarray) else ov
                        for ov in state["overlays"]
                    ]
                with lock:
                    shared["pipeline"] = state

            _update({"stage": "tracking", "message": "Pipeline 启动中…"})
            try:
                pipeline_fn(result["val"], _update)
            except Exception as exc:
                _update({"stage": "done", "message": f"Pipeline 出错: {exc}"})
                import traceback
                traceback.print_exc()

            with lock:
                if shared.get("pipeline", {}).get("stage") != "done":
                    shared["pipeline"]["stage"] = "done"
                    shared["pipeline"]["message"] = "Pipeline 完成！"

            # Phase 3: 等待 restart 或 finish
            while True:
                try:
                    path, raw = cmd_q.get(timeout=0.15)
                except queue.Empty:
                    continue
                if path == "/api/finish":
                    session_done = True
                    break
                elif path == "/api/restart":
                    break
        else:
            session_done = True

        # 如果是 restart，重置状态
        if not session_done:
            per_frame_sam.clear()
            active_frame_idx = first_mask_ref
            set_image_fn(Image.fromarray(frames[first_mask_ref]))
            with lock:
                shared["overlay"] = mask_url
                shared["status"] = "左键=前景 右键=背景"
                shared["pts"] = 0
                shared.pop("pipeline", None)
            print("[web] 重新标注 — 页面已重置")

    srv.shutdown()
    th.join(timeout=2)
    if result["val"] is None:
        raise RuntimeError("未收到交互结果")
    return result["val"]
