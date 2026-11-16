"""主电脑汇总服务。

当前能力：
  - 被发现（UDP 应答），从机不用填 IP
  - 接收从机摄像头注册
  - 双画面实时预览（入口 / 出口）
  - 一键测链路（帧率、首帧延迟、分辨率）
"""

from __future__ import annotations

import socket
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import requests
import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from common import netcfg
from master.stats import router as stats_router

VERSION = "0.6.0"
EDGE_EXPECT = "0.5.0"      # 主电脑认为的最新从机版本；不一致就在界面上点出来
APP_DIR = Path(__file__).resolve().parent.parent
SNAP_DIR = APP_DIR / "data" / "camtest"

app = FastAPI(title="Classroom CV Master", version=VERSION)
app.include_router(stats_router)
CAMERAS: dict[str, dict] = {}

STYLE = """
body{margin:0;background:#14161b;color:#e6e8eb;font:14px/1.7 system-ui,-apple-system,"Segoe UI",sans-serif;padding:22px 26px}
h1{font-size:17px;font-weight:500;margin:0 0 3px}
.sub{color:#8b93a1;font-size:12px;margin:0 0 16px}
.duo{display:flex;gap:14px;align-items:flex-start}
.pane{flex:1;min-width:0;background:#1b1e25;border:1px solid #2a3040;border-radius:12px;overflow:hidden}
.pane .bar{display:flex;justify-content:space-between;padding:9px 13px;font-size:12.5px;color:#c8ccd4;background:#20242c}
.pane .bar b{font-weight:500;color:#e6e8eb}
.pane img{width:100%;display:block;background:#0f1115;min-height:200px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(340px,1fr));gap:14px}
.card{background:#1b1e25;border:1px solid #2a3040;border-radius:12px;overflow:hidden}
.card img{width:100%;display:block;background:#0f1115;min-height:160px}
.meta{color:#8b93a1;font-size:12px;padding:9px 13px;word-break:break-all}
button{background:#1f2937;color:#dbe3f0;border:1px solid #374151;border-radius:8px;padding:5px 11px;font-size:12px;cursor:pointer;margin:0 0 9px 13px}
button:hover{border-color:#5b9dff;color:#fff}
.out{font-family:ui-monospace,Consolas,monospace;font-size:11.5px;color:#9fe1cb;padding:0 13px 10px;white-space:pre-wrap}
.ok{color:#97c459}.bad{color:#f09595}
.row{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-top:18px}
input{background:#0f1115;border:1px solid #2a3040;color:#e6e8eb;border-radius:8px;padding:6px 10px;font-size:12.5px;width:160px}
.empty{color:#8b93a1;font-size:13px;padding:30px;text-align:center;border:1px dashed #2a3040;border-radius:12px}
a{color:#85b7eb;text-decoration:none;font-size:12.5px}
.live{display:flex;gap:18px;align-items:center;flex-wrap:wrap;margin-top:14px;
  background:#1b1e25;border:1px solid #2a3040;border-radius:12px;padding:12px 16px}
.lk{min-width:74px}
.lk span{display:block;color:#8b93a1;font-size:11.5px;line-height:1.4}
.lk b{font-size:23px;font-weight:500;font-variant-numeric:tabular-nums;line-height:1.25}
.lk i{font-size:11.5px;color:#8b93a1;font-style:normal;margin-left:2px}
.lk.hl b{color:#97c459}
.lk.warn b{color:#e0b060}
.lk.bad b{color:#f09595}
.lk.sm b{font-size:16px}
.who{flex:1;min-width:220px;font-size:12.5px;color:#c8ccd4;line-height:1.9}
.who em{font-style:normal;color:#8b93a1;font-size:11.5px;margin-left:3px}
.who .tag{display:inline-block;font-size:11px;padding:0 6px;border-radius:99px;
  border:1px solid #5a4a33;background:#2a2319;color:#e0b060}
.live .go{font-size:12.5px;white-space:nowrap}
"""

JS = """
const PORT=@PORT@;
let SHOW_BOX = localStorage.getItem('ccv_box') !== '0';
function feed(ip,port){
  const path = SHOW_BOX ? 'overlay_feed' : 'video_feed';
  return `http://${ip}:${port}/${path}?t=${Date.now()}`;
}
function tagOf(c){return c.role==='in'?'入口':c.role==='out'?'出口':'通道';}
function label(c){const t=tagOf(c);return t===c.name?t:`${t} · ${c.name}`;}
function verOf(c){
  if(!c.version) return '<span class="bad">旧版代码</span>';
  return c.version==='@EDGE@' ? `v${c.version}` : `<span class="bad">v${c.version} 旧于 v@EDGE@</span>`;
}
function rid(ip){return 'r-'+ip.replace(/\\./g,'_');}
function toggleBox(){
  SHOW_BOX=!SHOW_BOX;
  localStorage.setItem('ccv_box', SHOW_BOX?'1':'0');
  const b=document.getElementById('boxBtn');
  if(b) b.textContent = SHOW_BOX?'识别框 开':'识别框 关';
  sig=''; load();
}
async function probe(ip,port,el){
  if(el) el.textContent='拉流中，最多 6 秒...';
  try{
    const d=await (await fetch(`/camtest?ip=${ip}&port=${port}`)).json();
    const t=d.ok?`通过 · ${d.frames} 帧 · ${d.fps} fps · 首帧 ${d.first_frame_ms}ms · ${d.resolution}`:`失败 · ${d.error}`;
    if(el){el.textContent=t;el.className='out '+(d.ok?'ok':'bad');}
  }catch(e){ if(el){el.textContent='请求出错 '+e;el.className='out bad';} }
}
async function pollRecog(){
  let d;
  try{ d=await (await fetch('/cameras.json')).json(); }catch(e){ return; }
  for(const c of d.cameras){
    const el=document.getElementById(rid(c.ip));
    if(!el) continue;
    try{
      const resp=await fetch(`http://${c.ip}:${c.port}/recog.json?t=${Date.now()}`);
      if(!resp.ok) throw new Error('HTTP '+resp.status);
      const r=await resp.json();
      const e=r.engine||{};
      if(!e.detector || e.detector==='none'){
        el.innerHTML='<span class="bad">识别未运行</span>（该机没开统计或模型缺失）';
      }else{
        el.innerHTML=`<span class="ok">识别中</span> 检测 ${e.detector} · 身份库 ${e.gallery} 人`
          + ` · 画面 ${r.faces||0} 张脸 · 已上报 ${r.events||0} 条 · 识别 ${r.fps||0} fps`
          + (r.queue? ` · <span class="bad">待补传 ${r.queue} 条</span>`:'');
      }
    }catch(err){
      el.innerHTML='<span class="bad">识别状态取不到</span>'
        + `（${err.message||'连不上'}）· 从机如果是旧版代码，重跑一次 setup_edge.bat`;
    }
  }
}
let sig='';
async function load(){
  const d=await (await fetch('/cameras.json')).json();
  const s=d.cameras.map(c=>c.name+c.ip+c.cam_ok+c.role+(c.version||'')).join('|')+'|'+(SHOW_BOX?1:0);
  if(s===sig) return;
  sig=s;
  const box=document.getElementById('stage');
  if(!d.cameras.length){box.innerHTML='<div class="empty">还没有从机注册。在另一台机器运行 setup_edge.bat</div>';return;}
  const list=d.cameras.slice(0,2);
  if(MODE==='both'){
    box.className='duo';
    box.innerHTML=list.map((c,i)=>`
      <div class="pane">
        <div class="bar"><span><b>${label(c)}</b></span><span>${c.width}x${c.height}</span></div>
        <img src="${feed(c.ip,c.port)}">
        <div class="meta">${c.ip}:${c.port} · ${c.cam_ok?'摄像头正常':'占位画面'} · ${verOf(c)}</div>
        <div class="meta" id="${rid(c.ip)}">识别状态读取中...</div>
        <button onclick="probe('${c.ip}',${c.port},document.getElementById('o'+${i}))">测连通性</button>
        <div class="out" id="o${i}"></div>
      </div>`).join('');
  }else{
    box.className='grid';
    box.innerHTML=list.map((c,i)=>`
      <div class="card">
        <img src="${feed(c.ip,c.port)}">
        <div class="meta">${label(c)} · ${c.ip}:${c.port} · ${c.cam_ok?'正常':'占位'} · ${c.width}x${c.height} · ${verOf(c)}</div>
        <div class="meta" id="${rid(c.ip)}">识别状态读取中...</div>
        <button onclick="probe('${c.ip}',${c.port},document.getElementById('o'+${i}))">测连通性</button>
        <div class="out" id="o${i}"></div>
      </div>`).join('');
  }
  pollRecog();
}
function probeManual(){
  const ip=document.getElementById('ip').value.trim();
  const port=document.getElementById('port').value.trim()||PORT;
  if(ip) probe(ip,port,document.getElementById('manualOut'));
}

// ---------------- 实时人数条 ----------------
// 数字全部来自主电脑那本账（/stats.json、/api/recent），前端只负责显示，
// 不在浏览器里自己数，免得页面上的数和导出的 CSV 对不上。
function p2(n){return String(n).padStart(2,'0');}
function esc(s){return String(s).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));}
function fmtDur(s){
  s=Math.round(s||0);
  if(s<60) return s+' 秒';
  const m=Math.floor(s/60), r=s%60;
  if(m<60) return m+' 分'+(r?' '+p2(r)+' 秒':'');
  return Math.floor(m/60)+' 时 '+p2(m%60)+' 分';
}
async function pollStats(){
  const box=document.getElementById('live');
  if(!box) return;
  try{
    const st=await (await fetch('/stats.json?t='+Date.now())).json();
    const r=await (await fetch('/api/recent?limit=1&t='+Date.now())).json();
    const inside=r.inside||[];
    const who = inside.length
      ? inside.map(p=> (p.person==='unknown'
            ? '<span class="tag">陌生</span>'
            : esc(p.person)) + `<em>${fmtDur(p.dwell_s)}</em>`).join(' &nbsp; ')
      : '<span style="color:#8b93a1">教室现在是空的</span>';
    const rate = st.recognize_rate==null ? '–' : (st.recognize_rate*100).toFixed(0)+'%';
    const avg  = st.avg_duration_s==null ? '–' : fmtDur(st.avg_duration_s);
    box.innerHTML = `
      <div class="lk hl"><span>当前在室</span><b>${st.inside}</b><i>人</i></div>
      <div class="lk"><span>今日进入</span><b>${st.in}</b><i>次</i></div>
      <div class="lk"><span>今日离开</span><b>${st.out}</b><i>次</i></div>
      <div class="lk"><span>峰值在室</span><b>${st.peak_inside}</b><i>人</i></div>
      <div class="lk sm"><span>平均停留</span><b>${avg}</b></div>
      <div class="lk"><span>识别率</span><b>${rate}</b></div>
      <div class="lk ${st.orphan_out?'bad':''}"><span>异常出口</span><b>${st.orphan_out}</b><i>次</i></div>
      <div class="who"><span style="color:#8b93a1">在室名单</span><br>${who}</div>
      <a class="go" href="/board">详细统计 →</a>`;
  }catch(e){
    box.innerHTML='<span style="color:#f09595">统计接口没响应，先看主电脑窗口还在不在</span>';
  }
}

load(); setInterval(load,4000); setInterval(pollRecog,3000);
pollStats(); setInterval(pollStats,2000);
const _bb=document.getElementById('boxBtn');
if(_bb) _bb.textContent = SHOW_BOX?'识别框 开':'识别框 关';
"""


def _session() -> requests.Session:
    s = requests.Session()
    s.trust_env = False
    return s


@app.get("/health")
def health():
    return {
        "ok": True,
        "version": VERSION,
        "name": socket.gethostname(),
        "ts": time.time(),
        "role": "master",
        "cameras": len(CAMERAS),
    }


@app.post("/register")
def register(info: dict):
    key = str(info.get("name") or info.get("ip"))
    info["last_seen"] = time.time()
    CAMERAS[key] = info
    print(f"  [注册] {key} {info.get('ip')}:{info.get('port')} "
          f"cam_ok={info.get('cam_ok')} {info.get('width')}x{info.get('height')}", flush=True)
    return {"ok": True, "count": len(CAMERAS)}


@app.get("/cameras.json")
def cameras_json():
    now = time.time()
    out = []
    for key, info in CAMERAS.items():
        item = dict(info)
        item["key"] = key
        item["online"] = now - info.get("last_seen", 0) < 120
        out.append(item)
    out.sort(key=lambda x: x.get("name", ""))
    return {"master": socket.gethostname(), "cameras": out}


@app.get("/camtest")
def camtest(ip: str, port: int = netcfg.CAM_PORT, seconds: float = 6.0, max_frames: int = 45):
    """从主电脑这边真的去拉一次流，测的是端到端链路。"""
    url = f"http://{ip}:{port}/video_feed"
    t0 = time.time()
    frames = 0
    first_ms = None
    stamps: list[float] = []
    last_jpg = None
    buf = bytearray()
    error = None

    try:
        with _session().get(url, stream=True, timeout=(3, 8)) as r:
            if r.status_code != 200:
                error = f"HTTP {r.status_code}"
            else:
                for chunk in r.iter_content(16384):
                    buf.extend(chunk)
                    while True:
                        start = buf.find(b"\xff\xd8")
                        if start < 0:
                            break
                        end = buf.find(b"\xff\xd9", start + 2)
                        if end < 0:
                            break
                        last_jpg = bytes(buf[start:end + 2])
                        del buf[:end + 2]
                        frames += 1
                        stamps.append(time.time())
                        if first_ms is None:
                            first_ms = round((time.time() - t0) * 1000, 1)
                    if len(buf) > 4_000_000:
                        del buf[:-2_000_000]
                    if frames >= max_frames or time.time() - t0 > seconds:
                        break
    except requests.RequestException as e:
        error = type(e).__name__ + ": " + str(e)[:80]

    if error or frames == 0:
        return {
            "ok": False, "ip": ip, "port": port, "frames": frames,
            "error": error or "没读到任何 JPEG 帧",
            "hint": "确认从机窗口还在、防火墙放行、从机 /health 能打开",
        }

    span = stamps[-1] - stamps[0]
    fps = round((frames - 1) / span, 1) if span > 0 else 0.0
    width = height = 0
    saved = None
    if last_jpg:
        img = cv2.imdecode(np.frombuffer(last_jpg, np.uint8), cv2.IMREAD_COLOR)
        if img is not None:
            height, width = img.shape[:2]
            SNAP_DIR.mkdir(parents=True, exist_ok=True)
            p = SNAP_DIR / f"{ip.replace('.', '_')}_{int(time.time())}.jpg"
            cv2.imwrite(str(p), img)
            saved = str(p)

    return {
        "ok": True, "ip": ip, "port": port, "frames": frames, "fps": fps,
        "first_frame_ms": first_ms, "resolution": f"{width}x{height}",
        "total_ms": round((time.time() - t0) * 1000, 0), "saved": saved,
    }


def _page(mode: str) -> HTMLResponse:
    master_ips = [i for i in netcfg.local_ipv4s() if not netcfg.is_virtual(i)]
    title = "双画面 · 入口 / 出口" if mode == "both" else "摄像头连通性"
    stage_cls = "duo" if mode == "both" else "grid"
    return HTMLResponse(f"""<!doctype html><meta charset="utf-8"><title>{title}</title>
<style>{STYLE}</style>
<h1>{title}</h1>
<p class="sub">主电脑 {socket.gethostname()} · {' · '.join(master_ips) or '无局域网地址'}
 · <a href="/board">进出统计</a> · <a href="/both">双画面</a> · <a href="/cameras">网格</a>
 · 框色：绿=已知身份，蓝=陌生人</p>
<div id="stage" class="{stage_cls}"><div class="empty">等待摄像头注册...</div></div>
<div id="live" class="live"><span style="color:#8b93a1">统计加载中…</span></div>

<div class="row">
  <input id="ip" placeholder="手动填 IP">
  <input id="port" value="{netcfg.CAM_PORT}" style="width:76px">
  <button onclick="probeManual()">测连通性</button>
  <button id="boxBtn" onclick="toggleBox()">识别框 开</button>
  <span class="out" id="manualOut"></span>
</div>
<script>
const MODE='{mode}';
{JS.replace("@PORT@", str(netcfg.CAM_PORT)).replace("@EDGE@", EDGE_EXPECT)}
</script>""")


@app.get("/", response_class=HTMLResponse)
def index():
    return _page("grid")


@app.get("/cameras", response_class=HTMLResponse)
def page_grid():
    return _page("grid")


@app.get("/both", response_class=HTMLResponse)
def page_both():
    return _page("both")


def _print_banner() -> None:
    ips = [ip for ip in netcfg.local_ipv4s() if not netcfg.is_virtual(ip)]
    print("=" * 58)
    print("  主电脑 · 汇总服务")
    print(f"  主机名   {socket.gethostname()}")
    for ip in ips:
        print(f"  双画面   http://{ip}:{netcfg.HTTP_PORT}/both")
        print(f"  统计     http://{ip}:{netcfg.HTTP_PORT}/board")
        print(f"  网格     http://{ip}:{netcfg.HTTP_PORT}/cameras")
    if not ips:
        print("  警告     没找到可用局域网地址，先连路由器或热点")
    print(f"  发现     UDP {netcfg.UDP_PORT}（从机自动找过来）")
    print("=" * 58, flush=True)


def main() -> None:
    _print_banner()
    threading.Thread(target=netcfg.serve_discovery, daemon=True).start()
    uvicorn.run(app, host="0.0.0.0", port=netcfg.HTTP_PORT, log_level="warning")


if __name__ == "__main__":
    main()
