"""往主电脑灌一批事件，验证账本、配对、面板 —— 不必真站在摄像头前。

用法：
    py -3.11 -m tools.simulate                 # 全场景，时间是回填的（从 40 分钟前开始）
    py -3.11 -m tools.simulate --scenario pair # 只跑正常进出配对
    py -3.11 -m tools.simulate --reset         # 先把库里今天的记录删掉再灌

四个场景对应四个最容易出错的环节：
    pair     正常进出配对、停留时长
    unknown  陌生人没身份，按先入先出配对
    orphan   出口先见到人（入口漏检 / 走侧门）
    dup      同一事件重复上报，必须只记一次
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

BASE = "http://127.0.0.1:8765"


def _sess() -> requests.Session:
    s = requests.Session()
    s.trust_env = False
    return s


def post(ses, host: str, payload: dict) -> dict:
    try:
        r = ses.post(f"{host}/event", json=payload, timeout=6)
        return r.json()
    except (requests.RequestException, json.JSONDecodeError) as e:
        return {"ok": False, "error": type(e).__name__ + ": " + str(e)[:60]}


def scenario_pair(now: float) -> list[tuple[float, str, str, str]]:
    """三个人陆续进，两个先走，一个还在里面。"""
    return [
        (now - 2400, "in", "IN", "张三"),
        (now - 2340, "in", "IN", "李四"),
        (now - 2280, "in", "IN", "王五"),
        (now - 1500, "out", "OUT", "张三"),
        (now - 1200, "out", "OUT", "李四"),
    ]


def scenario_unknown(now: float) -> list[tuple[float, str, str, str]]:
    """陌生人没有身份，只能按先入先出配。"""
    return [
        (now - 900, "in", "IN", "unknown"),
        (now - 780, "in", "IN", "unknown"),
        (now - 300, "out", "OUT", "unknown"),
    ]


def scenario_orphan(now: float) -> list[tuple[float, str, str, str]]:
    """出口先见到一个没在入口登记过的人。"""
    return [(now - 200, "out", "OUT", "赵六")]


def scenario_dup(now: float) -> list[tuple[float, str, str, str]]:
    """同一条事件连发三次，只应记一次。"""
    e = (now - 60, "in", "IN", "重复测试")
    return [e, e, e]


SCENARIOS = {
    "pair": scenario_pair,
    "unknown": scenario_unknown,
    "orphan": scenario_orphan,
    "dup": scenario_dup,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default=BASE)
    ap.add_argument("--scenario", default="all", choices=["all", *SCENARIOS])
    ap.add_argument("--dry-run", action="store_true", help="只打印，不发")
    ap.add_argument("--reset", action="store_true", help="先把今天已有的记录清空")
    ap.add_argument("--day-offset", type=int, default=0,
                    help="把整批事件往前挪 N 天，用来造历史数据、验证日期切换")
    args = ap.parse_args()

    if args.reset:
        from common import ledger
        r = ledger.reset_day()
        print(f"已清空 {r['day']}：{r['events']} 条事件、{r['sessions']} 段会话\n")

    now = time.time() - args.day_offset * 86400
    names = list(SCENARIOS) if args.scenario == "all" else [args.scenario]
    ses = _sess()

    print(f"目标 {args.host}   场景 {', '.join(names)}")
    print("-" * 74)
    total, okc, dup_uuid = 0, 0, None

    for name in names:
        print(f"[{name}]")
        for off, cam, kind, person in SCENARIOS[name](now):
            key = (person, round(off, 1))
            uid = dup_uuid or uuid.uuid4().hex
            if name == "dup":
                dup_uuid = uid          # 同一场景内复用同一个 uuid
            payload = {"uuid": uid, "cam": cam, "kind": kind, "person": person,
                       "sim": 0.62 if person != "unknown" else 0.18,
                       "ts": off, "dim": 0.06,
                       "trust_ts": True}      # 声明这是回填的历史时间，别做时钟校正
            total += 1
            if args.dry_run:
                print(f"   {time.strftime('%H:%M:%S', time.localtime(off))} "
                      f"{kind:3s} {person}")
                continue
            res = post(ses, args.host, payload)
            flag = "ok " if res.get("ok") else "ERR"
            if res.get("ok"):
                okc += 1
            print(f"   {time.strftime('%H:%M:%S', time.localtime(off))} "
                  f"{kind:3s} {person:<8s} {flag} {res.get('action') or res.get('error')}"
                  f"  在室 {res.get('inside', '-')}"
                  + (f"  时长 {res.get('duration_s')}s" if res.get('duration_s') else ""))
        print()

    if args.dry_run:
        print(f"共 {total} 条（未发送）")
        return

    print("-" * 74)
    print(f"发出 {total} 条，成功 {okc} 条")
    try:
        st = ses.get(f"{args.host}/stats.json", timeout=5).json()
        print(f"统计  在室 {st['inside']} · 进入 {st['in']} · 离开 {st['out']} "
              f"· 配对 {st['closed_sessions']} 段 · 平均停留 {st['avg_duration_s']}s "
              f"· 异常出口 {st['orphan_out']}")
        print(f"面板  {args.host}/board")
    except (requests.RequestException, json.JSONDecodeError):
        print("统计接口取不到，主电脑服务在跑吗？")


if __name__ == "__main__":
    main()
