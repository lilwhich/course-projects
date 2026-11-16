"""双画面链路验证：起主机服务 + 本机摄像头，检查注册、页面、两路拉流。"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def s():
    x = requests.Session()
    x.trust_env = False
    return x


def main() -> None:
    p1 = subprocess.Popen([sys.executable, "-m", "master.app"], cwd=ROOT,
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(3)
    p2 = subprocess.Popen([sys.executable, "-m", "edge.camera_server", "--port", "8767", "--role", "in"],
                          cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(7)
    try:
        j = s().get("http://127.0.0.1:8765/cameras.json", timeout=5).json()
        print("注册到面板：", len(j["cameras"]), "路")
        for c in j["cameras"]:
            print(f"   {c['name']:10s} role={c.get('role'):7s} {c['ip']}:{c['port']} {c['width']}x{c['height']}")

        h = s().get("http://127.0.0.1:8765/both", timeout=5)
        print(f"/both 页面  HTTP {h.status_code}  双画面布局={'duo' in h.text}  角色标签={'tagOf' in h.text}")

        d = s().get("http://127.0.0.1:8765/camtest",
                    params={"ip": "127.0.0.1", "port": 8767, "seconds": 5, "max_frames": 30},
                    timeout=20).json()
        tag = "通过" if d.get("ok") else "失败"
        print(f"本机入口  {tag}  {d.get('frames')} 帧 · {d.get('fps')} fps · {d.get('resolution')} · 首帧 {d.get('first_frame_ms')}ms")

        try:
            # 改成从机的实际 IP 再跑，或先用 edge/probe.py 自动发现
            d2 = s().get("http://127.0.0.1:8765/camtest",
                         params={"ip": "192.168.1.16",   # check:ignore 通用示例地址
                                 "port": 8767, "seconds": 5, "max_frames": 30},
                         timeout=20).json()
            tag2 = "通过" if d2.get("ok") else "失败"
            print(f"从机出口  {tag2}  {d2.get('frames')} 帧 · {d2.get('fps')} fps · {d2.get('resolution')}")
        except requests.RequestException:
            print("从机出口  当前未运行（先启动 setup_edge.bat）")
    finally:
        p2.terminate()
        p1.terminate()
        time.sleep(1)


if __name__ == "__main__":
    main()
