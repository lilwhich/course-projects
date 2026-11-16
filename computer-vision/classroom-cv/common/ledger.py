"""主电脑账本 —— 事件流水、进出配对、停留时长统计。

三条设计原则：

1. ``events`` 以 uuid 为主键。从机断网补传、重复上报都不会重复记账，
   这是"两台机器各自识别"能安全合并的前提。
2. 不单独存"在室表"。在室名单直接由 ``status='OPEN'`` 的会话算出来，
   少一张表就少一份会打架的数据。
3. 陌生人按先入先出配对。没有身份时无法用身份把人串起来，
   只剩"最早进去的那个先出来"这一条合理假设。
"""

from __future__ import annotations

import csv
import io
import sqlite3
import threading
import time
import uuid as uuidlib
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "ledger.db"

COOLDOWN_S = 20.0     # 同一人同一机位多少秒内不重复记账
MAX_DRIFT_S = 300.0   # 上报时间与本机时间差超过这个值，以接收时间为准
STALE_HOUR = 8.0      # 超过这么多小时未关闭的会话，标为 stale

_LOCK = threading.Lock()

SCHEMA = """
CREATE TABLE IF NOT EXISTS events(
  uuid       TEXT PRIMARY KEY,
  cam        TEXT NOT NULL,
  kind       TEXT NOT NULL,
  person     TEXT NOT NULL,
  sim        REAL,
  ts         REAL NOT NULL,
  recv_ts    REAL NOT NULL,
  dim        REAL,
  snap       TEXT,
  session_id INTEGER
);
CREATE INDEX IF NOT EXISTS idx_events_ts ON events(ts);
CREATE INDEX IF NOT EXISTS idx_events_person ON events(person);

CREATE TABLE IF NOT EXISTS sessions(
  id        INTEGER PRIMARY KEY AUTOINCREMENT,
  person    TEXT NOT NULL,
  cam_in    TEXT,
  cam_out   TEXT,
  enter_ts  REAL NOT NULL,
  exit_ts   REAL,
  duration_s REAL,
  status    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_sessions_status ON sessions(status);
CREATE INDEX IF NOT EXISTS idx_sessions_enter ON sessions(enter_ts);
"""


# ---------------------------------------------------------------- 基础设施

def _db() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=10.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    return conn


def init() -> None:
    conn = _db()
    try:
        conn.executescript(SCHEMA)
        conn.commit()
    finally:
        conn.close()


def _rows(cur) -> list[dict]:
    return [dict(r) for r in cur.fetchall()]


def _day_range(day: str | None = None) -> tuple[float, float, str]:
    """把 YYYY-MM-DD（本地时区）换成 [start, end) 的 epoch 秒。"""
    if day:
        try:
            y, m, d = (int(x) for x in day.split("-"))
        except ValueError:
            raise ValueError("day 格式应为 YYYY-MM-DD")
    else:
        lt = time.localtime()
        y, m, d = lt.tm_year, lt.tm_mon, lt.tm_mday
    start = time.mktime((y, m, d, 0, 0, 0, 0, 0, -1))
    return start, start + 86400.0, f"{y:04d}-{m:02d}-{d:02d}"


# ---------------------------------------------------------------- 记账

def ingest(ev: dict) -> dict:
    """收一条事件，落库并完成配对。幂等：同一个 uuid 只认第一次。"""
    uid = str(ev.get("uuid") or uuidlib.uuid4().hex)
    cam = str(ev.get("cam") or "camera")
    kind = str(ev.get("kind") or "").upper()
    person = str(ev.get("person") or "unknown")
    if kind not in ("IN", "OUT"):
        return {"ok": False, "error": "kind 只能是 IN 或 OUT", "uuid": uid}

    recv_ts = time.time()
    ts = float(ev.get("ts") or recv_ts)
    # 两机时钟可能没对齐，偏差太大就以接收时间为准。
    # 但补传测试数据时调用方明确声明 trust_ts，那种情况要保留原始时间戳。
    if not ev.get("trust_ts") and abs(recv_ts - ts) > MAX_DRIFT_S:
        ts = recv_ts

    with _LOCK:
        conn = _db()
        try:
            if conn.execute("SELECT 1 FROM events WHERE uuid=?", (uid,)).fetchone():
                return {"ok": True, "action": "duplicate", "uuid": uid}

            # 冷却只对"比已有记录更晚"的事件成立。补传历史事件时 ts 落在过去，
            # 若拿全库最新一条来比会算出负间隔，被误判成冷却（历史数据永远进不来）。
            last = conn.execute(
                "SELECT ts FROM events WHERE person=? AND cam=? AND ts<=? ORDER BY ts DESC LIMIT 1",
                (person, cam, ts),
            ).fetchone()
            if last and ts - last["ts"] < COOLDOWN_S:
                return {"ok": True, "action": "cooldown", "uuid": uid,
                        "gap_s": round(ts - last["ts"], 1)}

            action, session_id, duration = _apply(conn, kind, person, cam, ts)

            conn.execute(
                "INSERT INTO events(uuid,cam,kind,person,sim,ts,recv_ts,dim,snap,session_id)"
                " VALUES(?,?,?,?,?,?,?,?,?,?)",
                (uid, cam, kind, person, ev.get("sim"), ts, recv_ts,
                 ev.get("dim"), ev.get("snap"), session_id),
            )
            conn.commit()
            return {"ok": True, "action": action, "uuid": uid, "session_id": session_id,
                    "person": person, "duration_s": duration,
                    "inside": conn.execute(
                        "SELECT COUNT(*) c FROM sessions WHERE status='OPEN'").fetchone()["c"]}
        finally:
            conn.close()


def _apply(conn: sqlite3.Connection, kind: str, person: str, cam: str,
           ts: float) -> tuple[str, int | None, float | None]:
    """真正的状态机：IN 会话，OUT 关闭。返回 (动作, 会话号, 时长)。

    所有"找在室会话"的查询都带 ``enter_ts<=ts``：补传的历史事件不能和
    在它之后才建立的会话发生关系，否则昨天的 IN 会被今天的未关闭会话吞掉。
    """
    opened = conn.execute(
        "SELECT * FROM sessions WHERE person=? AND status='OPEN' AND enter_ts<=?"
        " ORDER BY enter_ts DESC LIMIT 1", (person, ts),
    ).fetchone()

    if kind == "IN":
        # 有身份的人重复入镜不新开会话（在门口晃会被认多次）；
        # 陌生人没有身份，两个陌生人进来必须算两个人，所以每次都开新会话，
        # 由出口按先入先出关掉最早那个。去重靠从机的轨迹确认，不靠这里。
        if opened and person != "unknown":
            return "in_ignored_already_inside", opened["id"], None
        cur = conn.execute(
            "INSERT INTO sessions(person,cam_in,enter_ts,status) VALUES(?,?,?,'OPEN')",
            (person, cam, ts),
        )
        return "in_open", cur.lastrowid, None

    # ---- OUT
    if opened:
        dur = max(0.0, ts - opened["enter_ts"])
        conn.execute(
            "UPDATE sessions SET exit_ts=?, duration_s=?, status='CLOSED', cam_out=? WHERE id=?",
            (ts, dur, cam, opened["id"]),
        )
        return "out_close", opened["id"], round(dur, 1)

    # 身份对不上时，陌生人按先入先出关掉最早那个
    if person == "unknown":
        cand = conn.execute(
            "SELECT * FROM sessions WHERE status='OPEN' AND person='unknown'"
            " AND enter_ts<=? ORDER BY enter_ts ASC LIMIT 1", (ts,),
        ).fetchone()
        if cand:
            dur = max(0.0, ts - cand["enter_ts"])
            conn.execute(
                "UPDATE sessions SET exit_ts=?, duration_s=?, status='CLOSED',"
                " cam_out=?, person=? WHERE id=?",
                (ts, dur, cam, person, cand["id"]),
            )
            return "out_close_fifo", cand["id"], round(dur, 1)

    # 出口先见到人：入口漏检、走侧门、或者跨天了
    cur = conn.execute(
        "INSERT INTO sessions(person,cam_in,cam_out,enter_ts,exit_ts,duration_s,status)"
        " VALUES(?,?,?,?,?,0,'ORPHAN')",
        (person, None, cam, ts, ts),
    )
    return "orphan_out", cur.lastrowid, 0.0


# ---------------------------------------------------------------- 查询

def inside_list(day: str | None = None) -> list[dict]:
    """未关闭的会话。查历史日期时，就是"那天结束前还留在教室里的人"。

    查今天时不按 enter_ts 过滤：昨晚没走、今天还在教室里的人也要算在室，
    这样"当前在室"卡片和名单才不会一个 6 一个 3。
    """
    sql = ("SELECT id, person, enter_ts, cam_in FROM sessions"
           " WHERE status='OPEN'")
    args: list = []
    if day and day != time.strftime("%Y-%m-%d"):
        start, end, _ = _day_range(day)
        sql += " AND enter_ts>=? AND enter_ts<?"
        args = [start, end]
    sql += " ORDER BY enter_ts ASC"
    conn = _db()
    try:
        rows = _rows(conn.execute(sql, args))
    finally:
        conn.close()
    now = time.time()
    for r in rows:
        r["dwell_s"] = round(now - r["enter_ts"], 0)
    return rows


def recent_events(limit: int = 30, day: str | None = None) -> list[dict]:
    sql = ("SELECT uuid,cam,kind,person,sim,ts,dim,snap,session_id FROM events")
    args: list = []
    if day:
        start, end, _ = _day_range(day)
        sql += " WHERE ts>=? AND ts<?"
        args = [start, end]
    sql += " ORDER BY ts DESC LIMIT ?"
    args.append(limit)
    conn = _db()
    try:
        return _rows(conn.execute(sql, args))
    finally:
        conn.close()


def recent_sessions(limit: int = 20, day: str | None = None) -> list[dict]:
    sql = "SELECT * FROM sessions"
    args: list = []
    if day:
        start, end, _ = _day_range(day)
        sql += " WHERE enter_ts>=? AND enter_ts<?"
        args = [start, end]
    sql += " ORDER BY COALESCE(exit_ts, enter_ts) DESC LIMIT ?"
    args.append(limit)
    conn = _db()
    try:
        return _rows(conn.execute(sql, args))
    finally:
        conn.close()


def stats(day: str | None = None) -> dict:
    start, end, label = _day_range(day)
    conn = _db()
    try:
        ev = conn.execute(
            "SELECT COUNT(*) n,"
            " SUM(kind='IN') n_in, SUM(kind='OUT') n_out,"
            " SUM(person='unknown') n_unknown,"
            " MIN(ts) first_ts, MAX(ts) last_ts,"
            " COUNT(DISTINCT person) n_people"
            " FROM events WHERE ts>=? AND ts<?", (start, end)).fetchone()

        closed = conn.execute(
            "SELECT COUNT(*) n, AVG(duration_s) avg_s FROM sessions"
            " WHERE status='CLOSED' AND enter_ts>=? AND enter_ts<?", (start, end)).fetchone()

        durs = [r["duration_s"] for r in conn.execute(
            "SELECT duration_s FROM sessions WHERE status='CLOSED'"
            " AND enter_ts>=? AND enter_ts<? ORDER BY duration_s", (start, end))]
        median = durs[len(durs) // 2] if durs else None

        orphan = conn.execute(
            "SELECT COUNT(*) n FROM sessions WHERE status='ORPHAN'"
            " AND enter_ts>=? AND enter_ts<?", (start, end)).fetchone()["n"]

        opened_today = conn.execute(
            "SELECT COUNT(*) n FROM sessions WHERE status='OPEN'"
            " AND enter_ts>=? AND enter_ts<?", (start, end)).fetchone()["n"]

        hourly = [{"hour": h, "in": 0, "out": 0} for h in range(24)]
        for r in conn.execute(
                "SELECT ts, kind FROM events WHERE ts>=? AND ts<?", (start, end)):
            h = time.localtime(r["ts"]).tm_hour
            hourly[h]["in" if r["kind"] == "IN" else "out"] += 1

        inside_now = conn.execute(
            "SELECT COUNT(*) n FROM sessions WHERE status='OPEN'").fetchone()["n"]
        if label != time.strftime("%Y-%m-%d"):
            # 看历史日期时，"在室"指的是那天结束前还没离开的人，不是现在教室里的人
            inside_now = conn.execute(
                "SELECT COUNT(*) n FROM sessions WHERE status='OPEN'"
                " AND enter_ts>=? AND enter_ts<?", (start, end)).fetchone()["n"]
        stale = conn.execute(
            "SELECT COUNT(*) n FROM sessions WHERE status='OPEN' AND enter_ts<?",
            (time.time() - STALE_HOUR * 3600,)).fetchone()["n"]

        spans = _spans(conn, start, end)
    finally:
        conn.close()

    n_total = ev["n"] or 0
    n_unknown = ev["n_unknown"] or 0
    identified = n_total - n_unknown
    n_closed = closed["n"] or 0
    # 配对成功率：所有"进过人"的会话里，多少真的等到了对应的离开
    attempts = n_closed + orphan + opened_today
    return {
        "day": label,
        "generated_at": time.time(),
        "inside": inside_now,
        "stale_sessions": stale,
        "events": n_total,
        "in": ev["n_in"] or 0,
        "out": ev["n_out"] or 0,
        "distinct_people": ev["n_people"] or 0,
        "unknown_events": n_unknown,
        "identified_events": identified,
        "recognize_rate": round(identified / n_total, 3) if n_total else None,
        "closed_sessions": n_closed,
        "open_sessions": opened_today,
        "avg_duration_s": round(closed["avg_s"], 1) if closed["avg_s"] is not None else None,
        "median_duration_s": round(median, 1) if median is not None else None,
        "orphan_out": orphan,
        "pair_rate": round(n_closed / attempts, 3) if attempts else None,
        "peak_inside": _peak(spans),
        "first_event_ts": ev["first_ts"],
        "last_event_ts": ev["last_ts"],
        "hourly": hourly,
        "cooldown_s": COOLDOWN_S,
    }


def _spans(conn: sqlite3.Connection, start: float, end: float) -> list[tuple[float, float]]:
    """把会话裁到查询区间内，得到一串 [进, 出) 区间。未关闭的按"到现在还在这"算。"""
    now = time.time()
    rows = conn.execute(
        "SELECT enter_ts, COALESCE(exit_ts, ?) AS out_ts FROM sessions"
        " WHERE status!='ORPHAN' AND enter_ts<? AND COALESCE(exit_ts,?)>?",
        (now, end, now, start)).fetchall()
    out = []
    for r in rows:
        a, b = max(start, r["enter_ts"]), min(end, r["out_ts"])
        if b > a:
            out.append((a, b))
    return out


def _peak(spans: list[tuple[float, float]]) -> int:
    """同时在室人数的峰值。同一秒先处理离开再处理进入，别把换人算成两个人。"""
    marks: list[tuple[float, int]] = []
    for a, b in spans:
        marks.append((a, 1))
        marks.append((b, -1))
    marks.sort(key=lambda m: (m[0], m[1]))
    cur = peak = 0
    for _, d in marks:
        cur += d
        peak = max(peak, cur)
    return peak


def occupancy(day: str | None = None, bin_min: int = 5) -> dict:
    """在室人数随时间变化，按 bin_min 分钟一档，画曲线用。

    每档取区间中点这一刻的人数，而不是"有多少人在这 5 分钟里待过"。
    后者会把"先走后进"的两个人算成一个峰值，和顶部峰值指标对不上。
    """
    start, end, label = _day_range(day)
    step = max(1, bin_min) * 60.0
    n_bins = int(round(86400 / step))
    conn = _db()
    try:
        spans = _spans(conn, start, end)
    finally:
        conn.close()
    # 今天只画到当前时刻，后面的空白别拉成一条平线
    upto = n_bins
    if label == time.strftime("%Y-%m-%d"):
        upto = max(1, min(n_bins, int((time.time() - start) // step) + 1))
    series = []
    for i in range(upto):
        probe = start + i * step + step / 2.0
        n = sum(1 for a, b in spans if a <= probe < b)
        series.append({"t": start + i * step, "n": n})
    return {"day": label, "bin_min": bin_min, "series": series}


def people_summary(day: str | None = None) -> list[dict]:
    """按人汇总：来了几次、待了多久。报告里的"个人明细表"直接用这个。"""
    start, end, _ = _day_range(day)
    conn = _db()
    try:
        rows = _rows(conn.execute(
            "SELECT person,"
            " COUNT(*) n,"
            " SUM(status='CLOSED') n_closed,"
            " SUM(status='ORPHAN') n_orphan,"
            " SUM(status='OPEN') n_open,"
            " SUM(CASE WHEN status='CLOSED' THEN duration_s ELSE 0 END) total_s,"
            " AVG(CASE WHEN status='CLOSED' THEN duration_s END) avg_s,"
            " MIN(enter_ts) first_ts,"
            " MAX(COALESCE(exit_ts, enter_ts)) last_ts"
            " FROM sessions WHERE enter_ts>=? AND enter_ts<?"
            " GROUP BY person ORDER BY total_s DESC, n DESC", (start, end)))
    finally:
        conn.close()
    for r in rows:
        r["total_s"] = round(r["total_s"] or 0.0, 1)
        r["avg_s"] = round(r["avg_s"], 1) if r["avg_s"] is not None else None
    return rows


def days() -> list[str]:
    """有数据的日期，倒序。面板的日期切换用它填下拉。"""
    conn = _db()
    try:
        stamps = [r["ts"] for r in conn.execute(
            "SELECT DISTINCT ts FROM events ORDER BY ts DESC LIMIT 5000")]
    finally:
        conn.close()
    seen: list[str] = []
    for ts in stamps:
        d = time.strftime("%Y-%m-%d", time.localtime(ts))
        if d not in seen:
            seen.append(d)
    return seen


# ---------------------------------------------------------------- 导出

def events_csv(day: str | None = None) -> str:
    start, end, _ = _day_range(day)
    conn = _db()
    try:
        rows = _rows(conn.execute(
            "SELECT * FROM events WHERE ts>=? AND ts<? ORDER BY ts", (start, end)))
    finally:
        conn.close()
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["时间", "机位", "方向", "身份", "相似度", "人脸占比", "会话号", "uuid"])
    for r in rows:
        w.writerow([
            time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(r["ts"])),
            r["cam"], r["kind"], r["person"],
            "" if r["sim"] is None else round(r["sim"], 3),
            "" if r["dim"] is None else round(r["dim"], 4),
            r["session_id"], r["uuid"],
        ])
    return buf.getvalue()


def sessions_csv(day: str | None = None) -> str:
    start, end, _ = _day_range(day)
    conn = _db()
    try:
        rows = _rows(conn.execute(
            "SELECT * FROM sessions WHERE enter_ts>=? AND enter_ts<? ORDER BY enter_ts",
            (start, end)))
    finally:
        conn.close()
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["身份", "进入时间", "离开时间", "停留秒数", "停留分钟", "状态", "入口机位", "出口机位"])
    for r in rows:
        w.writerow([
            r["person"],
            time.strftime("%H:%M:%S", time.localtime(r["enter_ts"])),
            time.strftime("%H:%M:%S", time.localtime(r["exit_ts"])) if r["exit_ts"] else "",
            "" if r["duration_s"] is None else round(r["duration_s"], 1),
            "" if r["duration_s"] is None else round(r["duration_s"] / 60.0, 1),
            r["status"], r["cam_in"] or "", r["cam_out"] or "",
        ])
    return buf.getvalue()


def summary_csv(day: str | None = None) -> str:
    """一页纸汇总：指标块 + 个人明细。报告正文里可以直接贴。"""
    s = stats(day)
    people = people_summary(day)
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["教室进出统计汇总", s["day"]])
    w.writerow(["生成时间", time.strftime("%Y-%m-%d %H:%M:%S")])
    w.writerow([])
    w.writerow(["指标", "数值", "说明"])
    w.writerow(["当前在室", s["inside"], "以未关闭的会话计"])
    w.writerow(["全天进入", s["in"], "入口机位事件数"])
    w.writerow(["全天离开", s["out"], "出口机位事件数"])
    w.writerow(["峰值在室", s["peak_inside"], "同时在场人数最大值"])
    w.writerow(["配对成功", s["closed_sessions"], "拿到进出两端的会话"])
    w.writerow(["异常出口", s["orphan_out"], "只有离开、没有对应进入"])
    w.writerow(["配对成功率", _pct(s["pair_rate"]), "配对 / (配对+异常+未关闭)"])
    w.writerow(["识别率", _pct(s["recognize_rate"]), "非 unknown 事件占比"])
    w.writerow(["平均停留", _mmss(s["avg_duration_s"]), "已配对会话"])
    w.writerow(["中位停留", _mmss(s["median_duration_s"]), "已配对会话"])
    w.writerow(["出现人数", s["distinct_people"], "含 unknown"])
    w.writerow([])
    w.writerow(["身份", "停留次数", "已配对", "未关闭", "异常出口",
                "总停留(分)", "平均停留(分)", "首次进入", "最后离开"])
    for p in people:
        w.writerow([
            p["person"], p["n"], p["n_closed"], p["n_open"], p["n_orphan"],
            round(p["total_s"] / 60.0, 1),
            "" if p["avg_s"] is None else round(p["avg_s"] / 60.0, 1),
            time.strftime("%H:%M:%S", time.localtime(p["first_ts"])),
            time.strftime("%H:%M:%S", time.localtime(p["last_ts"])),
        ])
    return buf.getvalue()


def _pct(v: float | None) -> str:
    return "" if v is None else f"{v * 100:.1f}%"


def _mmss(sec: float | None) -> str:
    if sec is None:
        return ""
    sec = int(round(sec))
    h, m, s = sec // 3600, (sec % 3600) // 60, sec % 60
    return f"{h}时{m:02d}分{s:02d}秒" if h else f"{m}分{s:02d}秒"


def reset_day(day: str | None = None) -> dict:
    """清掉某一天（默认今天）的记录。只在演示前重来一遍时用。"""
    start, end, label = _day_range(day)
    with _LOCK:
        conn = _db()
        try:
            n_ev = conn.execute("DELETE FROM events WHERE ts>=? AND ts<?",
                                (start, end)).rowcount
            n_se = conn.execute("DELETE FROM sessions WHERE enter_ts>=? AND enter_ts<?",
                                (start, end)).rowcount
            conn.commit()
            return {"ok": True, "day": label, "events": n_ev, "sessions": n_se}
        finally:
            conn.close()


def open_count() -> int:
    """当前未关闭的会话数，也就是"在室人数"。"""
    conn = _db()
    try:
        return conn.execute("SELECT COUNT(*) n FROM sessions WHERE status='OPEN'").fetchone()["n"]
    finally:
        conn.close()


def close_stale() -> int:
    """收尾：把开了太久没关的会话截断，别让在室人数一直虚高。"""
    cutoff = time.time() - STALE_HOUR * 3600
    with _LOCK:
        conn = _db()
        try:
            cur = conn.execute(
                "UPDATE sessions SET status='STALE', exit_ts=?, duration_s=?"
                " WHERE status='OPEN' AND enter_ts<?",
                (cutoff, STALE_HOUR * 3600, cutoff))
            conn.commit()
            return cur.rowcount
        finally:
            conn.close()


init()
