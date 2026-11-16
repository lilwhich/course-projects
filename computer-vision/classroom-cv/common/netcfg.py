"""网络自发现：让从机不用手填 IP 就能找到主电脑。

两端共用这一份，改协议只改这里。
"""

from __future__ import annotations

import json
import socket
import time
from pathlib import Path

HTTP_PORT = 8765   # 主电脑汇总服务
UDP_PORT = 8766    # 发现广播
CAM_PORT = 8767    # 从机摄像头服务
MAGIC = b"CCV_DISCOVER/1"

# 从机发现结果的缓存，避免每次启动都广播
CACHE = Path(__file__).resolve().parent.parent / "data" / "discovered.json"


def local_ipv4s() -> list[str]:
    """本机可用于局域网通信的 IPv4，默认出口网卡排在最前。"""
    ips: list[str] = []
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("223.5.5.5", 53))  # 不真的发包，只让系统选路由
        ips.append(s.getsockname()[0])
        s.close()
    except OSError:
        pass
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ip = info[4][0]
            if ip not in ips:
                ips.append(ip)
    except OSError:
        pass
    return [ip for ip in ips if not ip.startswith("127.")]


def broadcast_targets() -> list[str]:
    """受限广播 + 每个网段的 /24 广播，覆盖大多数家用路由器。"""
    targets = {"255.255.255.255"}
    for ip in local_ipv4s():
        head = ip.rsplit(".", 1)[0]
        targets.add(f"{head}.255")
    return sorted(targets)


def is_virtual(ip: str) -> bool:
    """VMware / VirtualBox 的虚拟网卡，从机连过去必然失败。"""
    return ip.startswith(("192.168.43.", "192.168.52.", "192.168.56.", "169.254."))


def serve_discovery(stop_event=None, name: str | None = None) -> None:
    """主电脑侧：UDP 应答，告诉从机"我在这，我的 IP 是这个"。"""
    host = name or socket.gethostname()
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    sock.bind(("0.0.0.0", UDP_PORT))
    sock.settimeout(1.0)

    primary = next((ip for ip in local_ipv4s() if not is_virtual(ip)), None)
    while not (stop_event and stop_event.is_set()):
        try:
            data, addr = sock.recvfrom(1024)
        except socket.timeout:
            continue
        except OSError:
            break
        if not data.startswith(MAGIC):
            continue
        # 优先回给对方同网段的地址，避免把虚拟网卡 IP 发出去
        reply_ip = primary
        for ip in local_ipv4s():
            if not is_virtual(ip) and ip.rsplit(".", 2)[0] == addr[0].rsplit(".", 2)[0]:
                reply_ip = ip
                break
        payload = f"{MAGIC.decode()}|{reply_ip}|{HTTP_PORT}|{host}".encode()
        try:
            sock.sendto(payload, addr)
        except OSError:
            pass
    sock.close()


def discover(timeout: float = 1.2, tries: int = 3) -> tuple[str, int, str] | None:
    """从机侧：广播找主电脑，返回 (ip, port, hostname)。"""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    sock.settimeout(timeout / tries)

    for _ in range(tries):
        for target in broadcast_targets():
            try:
                sock.sendto(MAGIC, (target, UDP_PORT))
            except OSError:
                continue
        deadline = time.time() + timeout / tries
        while time.time() < deadline:
            try:
                data, _addr = sock.recvfrom(1024)
            except socket.timeout:
                break
            except OSError:
                sock.close()
                return None
            parts = data.decode(errors="ignore").split("|")
            if len(parts) == 4 and parts[0] == MAGIC.decode():
                sock.close()
                result = (parts[1], int(parts[2]), parts[3])
                save_cache(*result)
                return result
    sock.close()
    return None


def save_cache(ip: str, port: int, name: str) -> None:
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(
        json.dumps({"ip": ip, "port": port, "name": name, "ts": time.time()}, indent=2),
        encoding="utf-8",
    )


def load_cache(max_age_hour: float = 24) -> tuple[str, int, str] | None:
    if not CACHE.exists():
        return None
    try:
        d = json.loads(CACHE.read_text(encoding="utf-8"))
        if time.time() - d["ts"] > max_age_hour * 3600:
            return None
        return d["ip"], int(d["port"]), d["name"]
    except (json.JSONDecodeError, KeyError, ValueError):
        return None
