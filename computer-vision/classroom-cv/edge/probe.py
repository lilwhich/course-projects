"""从机侧探针：自动找到主电脑并验证连通性。

用法：
    py -3.11 -m edge.probe            # 全自动：缓存 -> 广播 -> 主机名 -> 扫网段
    py -3.11 -m edge.probe 192.168.1.10   # 手填 IP 直连（check:ignore 通用示例地址）
"""

from __future__ import annotations

import socket
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import requests

from common import netcfg


def _session() -> requests.Session:
    # 系统里挂着 127.0.0.1:7890 的代理时，requests 会把局域网请求也塞进代理
    s = requests.Session()
    s.trust_env = False
    return s


def check(ip: str, port: int, timeout: float = 1.5):
    t0 = time.time()
    try:
        r = _session().get(f"http://{ip}:{port}/health", timeout=timeout)
        if r.ok:
            data = r.json()
            if data.get("ok"):
                return data, (time.time() - t0) * 1000
    except (requests.RequestException, ValueError):
        return None
    return None


def scan_subnet(port: int):
    """兜底：广播被交换机拦掉时，扫本机所在 /24。"""
    ips = [ip for ip in netcfg.local_ipv4s() if not netcfg.is_virtual(ip)]
    if not ips:
        return None
    head = ips[0].rsplit(".", 1)[0]
    print(f"  正在扫 {head}.0/24，最多 8 秒...", flush=True)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=64) as ex:
        futures = {ex.submit(check, f"{head}.{i}", port, 0.7): f"{head}.{i}" for i in range(2, 255)}
        for fut in futures:
            try:
                res = fut.result(timeout=max(0.1, 8 - (time.time() - t0)))
            except Exception:
                continue
            if res:
                print(f"  扫描命中，耗时 {time.time() - t0:.1f}s", flush=True)
                return futures[fut], res
    return None


def main() -> int:
    port = netcfg.HTTP_PORT
    manual = sys.argv[1] if len(sys.argv) > 1 else None
    print("=" * 58)
    print("  从机探针 · 寻找主电脑")
    print(f"  本机     {socket.gethostname()}  {netcfg.local_ipv4s()}")
    print("=" * 58, flush=True)

    candidates: list[tuple[str, str]] = []
    if manual:
        candidates.append((manual, "手填"))
    else:
        cached = netcfg.load_cache()
        # 缓存是从主电脑那份包一起拷过来时，里面存的是它自己的地址，会连回自己
        if cached and cached[0] not in netcfg.local_ipv4s():
            candidates.append((cached[0], "缓存"))
        found = netcfg.discover()
        if found:
            candidates.append((found[0], "UDP 广播"))
        if not [c for c in candidates if c[1] == "UDP 广播"]:
            candidates.append((socket.gethostname(), "主机名"))

    for ip, how in candidates:
        print(f"  尝试 [{how}] {ip} ...", flush=True)
        res = check(ip, port, 3.0)
        if res:
            data, ms = res
            netcfg.save_cache(ip, port, data.get("name", "master"))
            print(f"  连通 延迟 {ms:.0f}ms  主电脑={data.get('name')} 版本={data.get('version')}")
            print(f"  地址 http://{ip}:{port}")
            return 0
        print("    失败", flush=True)

    if not manual:
        hit = scan_subnet(port)
        if hit:
            ip, (data, ms) = hit
            netcfg.save_cache(ip, port, data.get("name", "master"))
            print(f"  连通 延迟 {ms:.0f}ms  主电脑={data.get('name')} 版本={data.get('version')}")
            print(f"  地址 http://{ip}:{port}")
            return 0

    print("\n  没找到主电脑。按顺序排查：")
    print("  1. 两台机器是否连的同一个路由器/热点（校园网常开客户端隔离，登不上）")
    print("  2. 主电脑窗口是否还开着，防火墙弹窗是否点了「允许」")
    print("  3. 用主电脑窗口里打印的那个 IP，手动跑：py -3.11 -m edge.probe <IP>")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
