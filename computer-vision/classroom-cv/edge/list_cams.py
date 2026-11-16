"""列出本机可用摄像头，解决"到底该用 index 0 还是 1"的问题。

    py -3.11 -m edge.list_cams
"""

from __future__ import annotations

import os

os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")  # 必须在 import cv2 之前

import cv2

try:  # 屏蔽 DSHOW 探测不存在索引时的刷屏警告
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
except AttributeError:
    pass


def main() -> None:
    print("逐个尝试摄像头索引，每路约 1~2 秒...\n")
    found = []
    for idx in range(5):
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap.release()
            continue
        ok, frame = cap.read()
        if ok:
            h, w = frame.shape[:2]
            fps = cap.get(cv2.CAP_PROP_FPS)
            print(f"  [{idx}] 可用  {w}x{h}  {fps:.0f}fps")
            found.append(idx)
        else:
            print(f"  [{idx}] 设备存在但读不到画面（可能被其他软件占用）")
        cap.release()

    print()
    if not found:
        print("没找到可用摄像头。检查：是否被腾讯会议/钉钉/浏览器占用；笔记本是否有物理遮挡开关")
        return
    print(f"用其中一路启动：py -3.11 -m edge.camera_server --cam {found[0]}")
    if len(found) > 1:
        print(f"接了两路的话，另一路给入口用：--cam {found[1]}")


if __name__ == "__main__":
    main()
