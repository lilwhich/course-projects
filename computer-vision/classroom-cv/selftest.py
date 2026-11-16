"""单机自检：在这台机器上起主服务，再跑一遍探针逻辑，验证发现与 HTTP 都通。

通过后再去第二台机器上跑 edge/probe，把「代码问题」和「网络问题」分开排查。
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
from common import netcfg  # noqa: E402
from edge.probe import check  # noqa: E402


def main() -> int:
    if netcfg.CACHE.exists():
        netcfg.CACHE.unlink()

    print("[1/3] 启动主服务子进程")
    proc = subprocess.Popen(
        [sys.executable, "-m", "master.app"],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    time.sleep(4)
    if proc.poll() is not None:
        print("     主服务没起来，输出如下：")
        print(proc.stdout.read() if proc.stdout else "")
        return 1

    ok = True
    try:
        print("[2/3] HTTP /health")
        res = check("127.0.0.1", netcfg.HTTP_PORT, 3.0)
        if res:
            data, ms = res
            print(f"     通过  {ms:.0f}ms  {data}")
        else:
            print("     失败：服务没响应")
            ok = False

        print("[3/3] UDP 广播发现")
        found = netcfg.discover(timeout=2.0, tries=3)
        if found:
            print(f"     通过  {found}")
        else:
            print("     失败：广播没收到应答（防火墙可能拦了 UDP 8766）")
            ok = False
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()

    print("=" * 58)
    print("  自检通过，可以去第二台机器上验证了" if ok else "  自检未全通过，按上面提示排查")
    print("=" * 58)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
