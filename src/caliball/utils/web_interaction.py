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
