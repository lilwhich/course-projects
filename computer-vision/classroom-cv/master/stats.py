"""主电脑统计层：事件入库、指标聚合、面板页、CSV 导出。

分工：
  POST /event            从机上报（唯一从机用得上的接口）
  GET  /stats.json       面板轮询的指标
  GET  /api/recent       在室名单 + 最近会话 + 事件流水
  GET  /board            统计面板
  GET  /export/*.csv     导出，报告里直接用
"""

from __future__ import annotations

import base64
import socket
import time
from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse

from common import ledger

APP_DIR = Path(__file__).resolve().parent.parent
SNAP_DIR = APP_DIR / "data" / "snapshots"
BOARD_HTML = Path(__file__).resolve().parent / "board.html"

router = APIRouter()


def _save_snap(uid: str, b64: str | None) -> str | None:
    """抓拍只在事件发生时传一次，落盘后用文件名引用，不给库塞二进制。"""
    if not b64:
        return None
    try:
        raw = base64.b64decode(b64)
    except (ValueError, TypeError):
        return None
    if len(raw) < 200:
        return None
    day = time.strftime("%Y-%m-%d")
    (SNAP_DIR / day).mkdir(parents=True, exist_ok=True)
    name = f"{day}/{uid[:16]}.jpg"
    (SNAP_DIR / name).write_bytes(raw)
    return name


@router.post("/event")
def post_event(ev: dict):
    uid = str(ev.get("uuid") or "")
    ev["snap"] = _save_snap(uid, ev.pop("snap_b64", None))
    return ledger.ingest(ev)


@router.get("/stats.json")
def stats_json(day: str | None = None):
    return ledger.stats(day)


@router.get("/api/people")
def api_people(day: str | None = None):
    return {"day": day or time.strftime("%Y-%m-%d"), "people": ledger.people_summary(day)}


@router.get("/api/days")
def api_days():
    days = ledger.days()
    today = time.strftime("%Y-%m-%d")
    if today not in days:
        days.insert(0, today)
    return {"today": today, "days": days}


@router.get("/api/occupancy")
def api_occupancy(day: str | None = None, bin_min: int = 5):
    return ledger.occupancy(day, max(1, min(bin_min, 60)))


@router.get("/api/recent")
def api_recent(limit: int = 25, day: str | None = None):
    return {
        "day": day or time.strftime("%Y-%m-%d"),
        "inside": ledger.inside_list(day),
        "sessions": ledger.recent_sessions(15, day),
        "events": ledger.recent_events(max(1, min(limit, 200)), day),
        "ts": time.time(),
    }


@router.post("/stale/close")
def close_stale():
    n = ledger.close_stale()
    return {"ok": True, "closed": n}


@router.post("/api/reset_day")
def api_reset_day(day: str | None = None, confirm: str = ""):
    """把某一天（默认今天）的账本清空。演示前重来一遍用。

    必须显式带 confirm=yes：这是唯一会删数据的接口，不能靠一个 GET 误触发。
    """
    if confirm != "yes":
        return {"ok": False, "error": "要清空请带 confirm=yes"}
    out = ledger.reset_day(day)
    # 只清了当天区间；更早开出去、一直没关的会话还挂在"在室"里，一并报给前端
    out["open_left"] = ledger.open_count()
    return out


@router.get("/export/events.csv", response_class=PlainTextResponse)
def export_events(day: str | None = None):
    body = "\ufeff" + ledger.events_csv(day)   # BOM：Excel 直接双击打开不乱码
    label = (day or time.strftime("%Y-%m-%d")).replace("-", "")
    return PlainTextResponse(
        body, media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="events-{label}.csv"',
                 "Content-Type": "text/csv; charset=utf-8"})


@router.get("/export/sessions.csv", response_class=PlainTextResponse)
def export_sessions(day: str | None = None):
    body = "\ufeff" + ledger.sessions_csv(day)   # BOM：Excel 直接双击打开不乱码
    label = (day or time.strftime("%Y-%m-%d")).replace("-", "")
    return PlainTextResponse(
        body, media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="sessions-{label}.csv"',
                 "Content-Type": "text/csv; charset=utf-8"})


@router.get("/export/summary.csv", response_class=PlainTextResponse)
def export_summary(day: str | None = None):
    body = "\ufeff" + ledger.summary_csv(day)
    label = (day or time.strftime("%Y-%m-%d")).replace("-", "")
    return PlainTextResponse(
        body, media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="summary-{label}.csv"',
                 "Content-Type": "text/csv; charset=utf-8"})


@router.get("/snap/{name:path}")
def snap(name: str):
    p = (SNAP_DIR / name).resolve()
    if not str(p).startswith(str(SNAP_DIR.resolve())) or not p.exists():
        return PlainTextResponse("not found", status_code=404)
    return FileResponse(p, media_type="image/jpeg")


@router.get("/board", response_class=HTMLResponse)
def board(day: str | None = None):
    html = BOARD_HTML.read_text(encoding="utf-8")
    html = html.replace("@HOST@", socket.gethostname())
    html = html.replace("@DAY@", day or time.strftime("%Y-%m-%d"))
    return HTMLResponse(html)
