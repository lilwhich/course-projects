"""临时测试：起一个本地 MJPEG 服务器，模拟手机摄像头。

手机 App（IP Webcam / DroidCam）对外提供的就是这种
multipart/x-mixed-replace 的 MJPEG 流，OpenCV 用同样的方式读取。
用它来端到端验证 video_source 那套逻辑，不需要真手机。

跑法：
    python _test_mjpeg.py            启动服务器（另开窗口跑 check_camera 连它）
    python _test_mjpeg.py --selftest  自己起服务 + 自己连，一条龙验证
"""
from __future__ import annotations

import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np

PORT = 8080


def make_frame(i: int) -> np.ndarray:
    """画一个带时间戳的测试画面。"""
    w, h = 960, 720
    img = np.full((h, w, 3), 40, np.uint8)
    # 网格背景
    for x in range(0, w, 60):
        cv2.line(img, (x, 0), (x, h), (60, 60, 60), 1)
    for y in range(0, h, 60):
        cv2.line(img, (0, y), (w, y), (60, 60, 60), 1)

    cx = 480 + int(120 * np.sin(i / 12))
    cv2.circle(img, (cx, 150), 34, (90, 200, 250), -1)          # 头
    cv2.line(img, (cx, 184), (cx, 400), (90, 200, 250), 10)     # 躯干
    cv2.line(img, (cx, 240), (cx - 90, 330), (200, 160, 90), 8)  # 左右手
    cv2.line(img, (cx, 240), (cx + 90, 330), (200, 160, 90), 8)
    cv2.line(img, (cx, 400), (cx - 60, 600), (90, 200, 250), 9)  # 腿
    cv2.line(img, (cx, 400), (cx + 60, 600), (90, 200, 250), 9)

    cv2.putText(img, f"FAKE PHONE CAM  frame={i}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (240, 240, 240), 2)
    cv2.putText(img, f"{time.strftime('%H:%M:%S')}", (20, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 2)
    return img


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):        # 静音
        pass

    def do_GET(self):
        if not self.path.startswith("/video"):
            self.send_response(404)
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Type",
                         "multipart/x-mixed-replace; boundary=frame")
        self.end_headers()
        i = 0
        try:
            while True:
                img = make_frame(i)
                ok, jpg = cv2.imencode(".jpg", img,
                                       [cv2.IMWRITE_JPEG_QUALITY, 80])
                if not ok:
                    continue
                data = jpg.tobytes()
                self.wfile.write(b"--frame\r\n")
                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                self.wfile.write(f"Content-Length: {len(data)}\r\n\r\n".encode())
                self.wfile.write(data)
                self.wfile.write(b"\r\n")
                self.wfile.flush()
                i += 1
                time.sleep(0.04)
        except (BrokenPipeError, ConnectionResetError, OSError):
            return


def serve_forever():
    srv = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    srv.daemon_threads = True
    srv.serve_forever()


def selftest() -> int:
    threading.Thread(target=serve_forever, daemon=True).start()
    time.sleep(1.2)
    url = f"http://127.0.0.1:{PORT}/video"

    print("== 1. 地址规范化 ==")
    import video_source as V
    print("   ", url, "->", V.normalize(url))
    print("    '192.168.1.7:8080' ->", V.normalize("192.168.1.7:8080"))

    print("\n== 2. probe 能否出帧 ==")
    ok, w, h, fr = V.probe(url, reads=5)
    print(f"    ok={ok}  分辨率={w}x{h}  frame不为空={fr is not None}")
    if not ok:
        print("    [失败] MJPEG 读取不通")
        return 1

    print("\n== 3. SourceReader 连续读 20 帧 ==")
    r = V.SourceReader(url, verbose=True)
    got, sizes = 0, set()
    for _ in range(20):
        ok, f = r.read()
        if ok:
            got += 1
            sizes.add(f.shape[:2])
    r.release()
    print(f"    读到 {got}/20 帧，尺寸集合 {sizes}")

    print("\n== 4. check_camera 走网络流跑 5 帧 ==")
    p = subprocess.run(
        [sys.executable, "-u", "check_camera.py", "--source", url,
         "--max-frames", "5"],
        capture_output=True, text=True, timeout=120)
    tail = [ln for ln in p.stdout.strip().splitlines() if ln.strip()]
    for ln in tail[:4] + tail[-6:]:
        print("   ", ln)
    print("    exit code:", p.returncode)

    print("\n== 结论 ==")
    ok_all = got >= 18 and p.returncode == 0
    print("    MJPEG 链路" + ("可用 ✔" if ok_all else "有问题 ✘"))
    return 0 if ok_all else 1


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        raise SystemExit(selftest())
    print(f"模拟手机摄像头：{PORT} 端口，路径 /video")
    print("另开窗口：python check_camera.py --source "
          f"http://127.0.0.1:{PORT}/video")
    serve_forever()
