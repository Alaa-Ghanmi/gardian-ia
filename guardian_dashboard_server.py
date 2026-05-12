#!/usr/bin/env python3
"""
guardian_dashboard_server.py
=============================
Serves the Guardian AI dashboard on port 8888.
Open in browser: http://YOUR-UBUNTU-IP:8888

- Reads predictions directly from guardian_predictions.json
- Auto-refreshes every 30 seconds
- No manual steps needed
"""

import http.server
import json
import socketserver
from pathlib import Path
from datetime import datetime

PREDICTIONS_FILE = Path("/var/ossec/logs/alerts/guardian_predictions.json")
PORT = 8888

HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>GUARDIAN AI — Attack Forecasting</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Orbitron:wght@400;700;900&family=Rajdhani:wght@400;500;600;700&display=swap');

:root {
  --bg:     #020810;
  --panel:  #050f1a;
  --border: #0a2440;
  --accent: #00d4ff;
  --green:  #00ff88;
  --yellow: #ffd200;
  --orange: #ff6b00;
  --red:    #ff1744;
  --blue:   #4488ff;
  --muted:  #4a7090;
  --text:   #c8e0f4;
  --dim:    #1a3a5c;
}

* { margin:0; padding:0; box-sizing:border-box; }

body {
  background: var(--bg);
  color: var(--text);
  font-family: 'Rajdhani', sans-serif;
  min-height: 100vh;
}

body::before {
  content:'';
  position:fixed; inset:0;
  background-image:
    linear-gradient(rgba(0,212,255,0.03) 1px, transparent 1px),
    linear-gradient(90deg, rgba(0,212,255,0.03) 1px, transparent 1px);
  background-size: 40px 40px;
  pointer-events:none; z-index:0;
}

.wrap { position:relative; z-index:1; }

/* HEADER */
header {
  display:flex; align-items:center; justify-content:space-between;
  padding: 18px 32px;
  border-bottom: 1px solid var(--border);
  background: linear-gradient(180deg, rgba(0,212,255,0.05) 0%, transparent 100%);
}

.logo { display:flex; align-items:center; gap:14px; }

.logo-icon {
  width:46px; height:46px;
  border: 2px solid var(--accent); border-radius:8px;
  display:flex; align-items:center; justify-content:center;
  font-size:22px;
  box-shadow: 0 0 20px rgba(0,212,255,0.3);
  animation: glow 2s ease-in-out infinite;
}

@keyframes glow {
  0%,100% { box-shadow: 0 0 20px rgba(0,212,255,0.3); }
  50%      { box-shadow: 0 0 40px rgba(0,212,255,0.7); }
}

.logo h1 {
  font-family:'Orbitron',monospace; font-size:20px; font-weight:900;
  color:var(--accent); letter-spacing:4px;
  text-shadow: 0 0 20px rgba(0,212,255,0.5);
}

.logo p {
  font-family:'Share Tech Mono',monospace; font-size:10px;
  color:var(--muted); letter-spacing:2px; margin-top:2px;
}

.header-right { display:flex; align-items:center; gap:20px; }

.live {
  display:flex; align-items:center; gap:8px;
  font-family:'Share Tech Mono',monospace; font-size:11px; color:var(--green);
}

.dot {
  width:8px; height:8px; border-radius:50%;
  background:var(--green); box-shadow:0 0 10px var(--green);
  animation: blink 1.2s ease-in-out infinite;
}

@keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.2} }

.clock {
  font-family:'Share Tech Mono',monospace; font-size:14px;
  color:var(--accent); letter-spacing:2px;
}

.countdown {
  font-family:'Share Tech Mono',monospace; font-size:11px;
  color:var(--muted); letter-spacing:1px;
}

/* STATS */
.stats {
  display:grid; grid-template-columns:repeat(4,1fr);
  gap:1px; background:var(--border);
  border-bottom:1px solid var(--border);
}

.stat {
  background:var(--panel); padding:18px 24px;
  position:relative; overflow:hidden;
}

.stat::before {
  content:''; position:absolute; top:0; left:0; right:0; height:2px;
}

.stat.total::before   { background:var(--accent); }
.stat.danger::before  { background:var(--red);    }
.stat.caution::before { background:var(--yellow); }
.stat.ok::before      { background:var(--green);  }

.stat-label {
  font-family:'Share Tech Mono',monospace; font-size:9px;
  color:var(--muted); letter-spacing:2px; margin-bottom:8px;
}

.stat-val {
  font-family:'Orbitron',monospace; font-size:32px; font-weight:700;
  color:var(--accent); line-height:1;
}

.stat.danger  .stat-val { color:var(--red);    text-shadow:0 0 20px rgba(255,23,68,0.5);  }
.stat.caution .stat-val { color:var(--yellow); text-shadow:0 0 20px rgba(255,210,0,0.5);  }
.stat.ok      .stat-val { color:var(--green);  text-shadow:0 0 20px rgba(0,255,136,0.5);  }

.stat-sub { font-size:11px; color:var(--muted); margin-top:6px; font-weight:500; }

/* GRID */
.grid {
  display:grid; grid-template-columns:1fr 380px;
  min-height: calc(100vh - 170px);
}

.left  { border-right:1px solid var(--border); }

.section-head {
  display:flex; align-items:center; justify-content:space-between;
  padding:14px 22px; border-bottom:1px solid var(--border);
  background:rgba(0,212,255,0.02);
}

.section-title {
  font-family:'Share Tech Mono',monospace; font-size:10px;
  color:var(--accent); letter-spacing:3px;
}

.section-count {
  font-family:'Share Tech Mono',monospace; font-size:10px; color:var(--muted);
}

/* CARDS */
.feed {
  padding:14px; display:flex; flex-direction:column; gap:10px;
  max-height:calc(100vh - 220px); overflow-y:auto;
}

.feed::-webkit-scrollbar { width:3px; }
.feed::-webkit-scrollbar-thumb { background:var(--dim); border-radius:2px; }

.card {
  background:var(--panel); border:1px solid var(--border); border-radius:6px;
  padding:16px 18px; cursor:pointer; transition:all 0.2s;
  position:relative; overflow:hidden;
}

.card::before {
  content:''; position:absolute; left:0; top:0; bottom:0; width:3px;
}

.card.CRITICAL::before { background:var(--red);    }
.card.HIGH::before     { background:var(--orange); }
.card.ELEVATED::before { background:var(--yellow); }
.card.LOW::before      { background:var(--blue);   }
.card.NORMAL::before   { background:var(--green);  }

.card:hover, .card.active {
  border-color:var(--accent);
  background:rgba(0,212,255,0.05);
  transform:translateX(2px);
}

.card-top {
  display:flex; align-items:center; justify-content:space-between;
  margin-bottom:8px;
}

.badge {
  font-family:'Orbitron',monospace; font-size:9px; font-weight:700;
  letter-spacing:2px; padding:3px 10px; border-radius:3px;
  border:1px solid currentColor;
}

.CRITICAL .badge { color:var(--red);    background:rgba(255,23,68,0.1);  }
.HIGH     .badge { color:var(--orange); background:rgba(255,107,0,0.1); }
.ELEVATED .badge { color:var(--yellow); background:rgba(255,210,0,0.1); }
.LOW      .badge { color:var(--blue);   background:rgba(68,136,255,0.1);}
.NORMAL   .badge { color:var(--green);  background:rgba(0,255,136,0.1); }

.card-time {
  font-family:'Share Tech Mono',monospace; font-size:10px; color:var(--muted);
}

.card-type {
  font-family:'Orbitron',monospace; font-size:13px; font-weight:700;
  color:var(--text); margin-bottom:6px; letter-spacing:1px;
}

.card-meta {
  display:flex; gap:14px; font-size:12px; color:var(--muted); font-weight:500;
}

/* DETAIL */
.detail { padding:18px; max-height:calc(100vh - 170px); overflow-y:auto; }
.detail::-webkit-scrollbar { width:3px; }
.detail::-webkit-scrollbar-thumb { background:var(--dim); }

.detail-empty {
  display:flex; flex-direction:column; align-items:center;
  justify-content:center; height:300px;
  color:var(--muted); text-align:center; gap:10px;
}

.detail-empty .icon { font-size:36px; opacity:0.2; }
.detail-empty p { font-family:'Share Tech Mono',monospace; font-size:11px; letter-spacing:1px; }

.d-head {
  display:flex; align-items:center; gap:12px;
  margin-bottom:18px; padding-bottom:14px;
  border-bottom:1px solid var(--border);
}

.d-icon {
  width:52px; height:52px; border-radius:8px;
  display:flex; align-items:center; justify-content:center;
  font-size:28px; border:1px solid var(--border);
}

.d-title { font-family:'Orbitron',monospace; font-size:12px; font-weight:700; letter-spacing:2px; }
.d-ts    { font-family:'Share Tech Mono',monospace; font-size:10px; color:var(--muted); margin-top:3px; }

.d-sec { margin-bottom:16px; }

.d-sec-title {
  font-family:'Share Tech Mono',monospace; font-size:9px;
  color:var(--muted); letter-spacing:3px; text-transform:uppercase;
  margin-bottom:8px; padding-bottom:5px; border-bottom:1px solid var(--border);
}

.d-row {
  display:flex; justify-content:space-between; align-items:center;
  padding:6px 0; font-size:12px; border-bottom:1px solid rgba(10,36,64,0.5);
}
.d-row:last-child { border:none; }
.d-row-label { color:var(--muted); font-weight:500; }
.d-row-val   { font-family:'Share Tech Mono',monospace; font-size:11px; text-align:right; }

.prob-wrap { margin:5px 0 10px; }
.prob-head { display:flex; justify-content:space-between; font-size:11px; margin-bottom:4px; }
.prob-name { color:var(--muted); font-weight:500; }
.prob-val  { color:var(--accent); font-family:'Share Tech Mono',monospace; }
.prob-bar  { height:3px; background:var(--dim); border-radius:2px; overflow:hidden; }
.prob-fill { height:100%; background:linear-gradient(90deg,var(--accent),rgba(0,212,255,0.4)); border-radius:2px; }

.ev-item {
  display:flex; gap:8px; padding:6px 0; font-size:12px;
  border-bottom:1px solid rgba(10,36,64,0.5); line-height:1.4;
}
.ev-item:last-child { border:none; }
.ev-dot { color:var(--accent); font-size:10px; margin-top:2px; flex-shrink:0; }

.an-item {
  display:flex; gap:8px; padding:6px 0; font-size:12px;
  color:var(--yellow); border-bottom:1px solid rgba(10,36,64,0.5); line-height:1.4;
}
.an-item:last-child { border:none; }

.rec-box {
  background:rgba(0,212,255,0.04); border:1px solid rgba(0,212,255,0.15);
  border-radius:6px; padding:12px; font-size:13px; line-height:1.6; font-weight:500;
}

.rec-box.CRITICAL { border-color:rgba(255,23,68,0.3);  background:rgba(255,23,68,0.05); }
.rec-box.HIGH     { border-color:rgba(255,107,0,0.3);  background:rgba(255,107,0,0.05); }
.rec-box.ELEVATED { border-color:rgba(255,210,0,0.3);  background:rgba(255,210,0,0.05); }

.stages { display:flex; align-items:center; gap:5px; flex-wrap:wrap; padding:8px 0; }
.s-pill {
  font-family:'Share Tech Mono',monospace; font-size:9px;
  padding:3px 8px; border-radius:3px;
  border:1px solid var(--border); color:var(--muted); letter-spacing:1px;
}
.s-pill.on {
  border-color:var(--accent); color:var(--accent);
  background:rgba(0,212,255,0.1); box-shadow:0 0 10px rgba(0,212,255,0.2);
}
.s-arr { color:var(--muted); font-size:10px; }

.no-data {
  display:flex; flex-direction:column; align-items:center;
  justify-content:center; height:400px; gap:14px;
  color:var(--muted); text-align:center;
}
.no-data .big { font-size:56px; opacity:0.12; }
.no-data h3 { font-family:'Orbitron',monospace; font-size:14px; color:var(--dim); letter-spacing:2px; }
.no-data p  { font-family:'Share Tech Mono',monospace; font-size:11px; line-height:1.7; max-width:300px; }
</style>
</head>
<body>
<div class="wrap">

<header>
  <div class="logo">
    <div class="logo-icon">🛡️</div>
    <div>
      <h1>GUARDIAN AI</h1>
      <p>ATTACK FORECASTING ENGINE · LIVE</p>
    </div>
  </div>
  <div class="header-right">
    <div class="live"><div class="dot"></div>AI RUNNING</div>
    <div class="clock" id="clock">--:--:--</div>
    <div class="countdown" id="countdown">Next refresh in 30s</div>
  </div>
</header>

<div class="stats">
  <div class="stat total">
    <div class="stat-label">TOTAL PREDICTIONS</div>
    <div class="stat-val" id="sTotal">0</div>
    <div class="stat-sub">All time</div>
  </div>
  <div class="stat danger">
    <div class="stat-label">CRITICAL / HIGH</div>
    <div class="stat-val" id="sDanger">0</div>
    <div class="stat-sub">Immediate action</div>
  </div>
  <div class="stat caution">
    <div class="stat-label">ELEVATED</div>
    <div class="stat-val" id="sCaution">0</div>
    <div class="stat-sub">Monitor closely</div>
  </div>
  <div class="stat ok">
    <div class="stat-label">NORMAL / LOW</div>
    <div class="stat-val" id="sOk">0</div>
    <div class="stat-sub">Routine monitoring</div>
  </div>
</div>

<div class="grid">
  <div class="left">
    <div class="section-head">
      <span class="section-title">⬡ LIVE PREDICTION FEED</span>
      <span class="section-count" id="feedCount">loading...</span>
    </div>
    <div class="feed" id="feed">
      <div style="display:flex;align-items:center;justify-content:center;height:200px;gap:12px;color:var(--muted);font-family:'Share Tech Mono',monospace;font-size:12px;">
        <div style="width:18px;height:18px;border:2px solid var(--dim);border-top-color:var(--accent);border-radius:50%;animation:spin 0.8s linear infinite"></div>
        LOADING PREDICTIONS...
      </div>
      <style>@keyframes spin{to{transform:rotate(360deg)}}</style>
    </div>
  </div>
  <div>
    <div class="section-head">
      <span class="section-title">⬡ DETAIL</span>
    </div>
    <div class="detail" id="detail">
      <div class="detail-empty">
        <div class="icon">🔍</div>
        <p>SELECT A PREDICTION<br>TO VIEW DETAILS</p>
      </div>
    </div>
  </div>
</div>

</div>

<script>
let preds = [];
let sel   = -1;
let countdown = 30;

// Clock
setInterval(() => {
  document.getElementById('clock').textContent =
    new Date().toLocaleTimeString('en-GB', {hour12:false});
}, 1000);

// Countdown + auto-refresh
setInterval(() => {
  countdown--;
  document.getElementById('countdown').textContent = `Next refresh in ${countdown}s`;
  if (countdown <= 0) { countdown = 30; load(); }
}, 1000);

function riskEmoji(r) {
  return {CRITICAL:'🔴',HIGH:'🟠',ELEVATED:'🟡',LOW:'🔵',NORMAL:'🟢'}[r] || '⚪';
}

function riskColor(r) {
  return {CRITICAL:'#ff1744',HIGH:'#ff6b00',ELEVATED:'#ffd200',LOW:'#4488ff',NORMAL:'#00ff88'}[r] || '#c8e0f4';
}

function fmt(ts) {
  if (!ts) return '—';
  try {
    return new Date(ts).toLocaleString('en-GB',{
      day:'2-digit',month:'short',hour:'2-digit',minute:'2-digit',second:'2-digit',hour12:false
    });
  } catch(e) { return ts; }
}

async function load() {
  try {
    const res  = await fetch('/predictions');
    const data = await res.json();
    preds = data.sort((a,b) => new Date(b.timestamp) - new Date(a.timestamp));
    renderStats();
    renderFeed();
    if (sel === -1 && preds.length > 0) showDetail(0);
  } catch(e) {
    document.getElementById('feed').innerHTML =
      '<div style="color:var(--red);font-family:Share Tech Mono,monospace;font-size:11px;padding:20px;text-align:center">Cannot load predictions — check guardian service is running</div>';
  }
}

function renderStats() {
  const danger   = preds.filter(p => ['CRITICAL','HIGH'].includes(p.guardian_ai?.risk_level)).length;
  const caution  = preds.filter(p => p.guardian_ai?.risk_level === 'ELEVATED').length;
  const ok       = preds.filter(p => ['NORMAL','LOW'].includes(p.guardian_ai?.risk_level)).length;
  document.getElementById('sTotal').textContent   = preds.length;
  document.getElementById('sDanger').textContent  = danger;
  document.getElementById('sCaution').textContent = caution;
  document.getElementById('sOk').textContent      = ok;
}

function renderFeed() {
  const feed = document.getElementById('feed');
  document.getElementById('feedCount').textContent = preds.length + ' records';

  if (!preds.length) {
    feed.innerHTML = `<div class="no-data">
      <div class="big">📡</div>
      <h3>NO PREDICTIONS YET</h3>
      <p>The AI runs every 5 minutes.<br>Wait a moment and the page will refresh automatically.</p>
    </div>`;
    return;
  }

  feed.innerHTML = preds.map((p, i) => {
    const ai   = p.guardian_ai || {};
    const risk = ai.risk_level || 'NORMAL';
    const type = (ai.attack_prediction || 'unknown').toUpperCase().replace(/_/g,' ');
    const prob = ai.attack_probability || ai.attack_probability_pct || '—';
    const eta  = ai.estimated_time_to_attack || '—';

    return `<div class="card ${risk} ${i===sel?'active':''}" onclick="showDetail(${i})">
      <div class="card-top">
        <span class="badge">${riskEmoji(risk)} ${risk}</span>
        <span class="card-time">${fmt(p.timestamp)}</span>
      </div>
      <div class="card-type">${type}</div>
      <div class="card-meta">
        <span>📊 ${prob}</span>
        <span>⏱ ${eta}</span>
        <span>🎯 ${ai.confidence||'—'}</span>
      </div>
    </div>`;
  }).join('');
}

function showDetail(idx) {
  sel = idx;
  renderFeed();
  const p  = preds[idx];
  const ai = p.guardian_ai || {};
  const risk = ai.risk_level || 'NORMAL';

  const stages = ['recon','initial_access','exploitation','post_compromise','impact'];
  const cur    = ai.attack_stage || 'none';
  const stageHtml = stages.map((s,i) =>
    `<span class="s-pill ${s===cur?'on':''}">${s.replace('_',' ')}</span>` +
    (i<stages.length-1 ? '<span class="s-arr">›</span>' : '')
  ).join('');

  const classes   = ai.top_classes || ai.top_attack_classes || {};
  const classHtml = Object.entries(classes).map(([k,v]) => {
    const pct = parseFloat(v)||0;
    return `<div class="prob-wrap">
      <div class="prob-head"><span class="prob-name">${k.replace(/_/g,' ').toUpperCase()}</span><span class="prob-val">${v}</span></div>
      <div class="prob-bar"><div class="prob-fill" style="width:${pct}%"></div></div>
    </div>`;
  }).join('');

  const evidence = (ai.evidence_signals||ai.evidence||[]);
  const evHtml   = evidence.length
    ? evidence.map(e=>`<div class="ev-item"><span class="ev-dot">◆</span>${e}</div>`).join('')
    : '<div style="color:var(--muted);font-size:11px;padding:6px 0">No strong signals</div>';

  const anomalies = (ai.behavioral_anomalies||ai.anomalies||[]);
  const anHtml    = anomalies.length
    ? anomalies.map(a=>`<div class="an-item">⚠ ${a}</div>`).join('')
    : '<div style="color:var(--muted);font-size:11px;padding:6px 0">No anomalies detected</div>';

  document.getElementById('detail').innerHTML = `
    <div class="d-head">
      <div class="d-icon">${riskEmoji(risk)}</div>
      <div>
        <div class="d-title" style="color:${riskColor(risk)}">${risk} RISK</div>
        <div class="d-ts">${fmt(p.timestamp)}</div>
      </div>
    </div>

    <div class="d-sec">
      <div class="d-sec-title">PREDICTION</div>
      <div class="d-row">
        <span class="d-row-label">Attack Type</span>
        <span class="d-row-val" style="color:${riskColor(risk)}">${(ai.attack_prediction||'—').toUpperCase().replace(/_/g,' ')}</span>
      </div>
      <div class="d-row">
        <span class="d-row-label">Probability</span>
        <span class="d-row-val">${ai.attack_probability||ai.attack_probability_pct||'—'}</span>
      </div>
      <div class="d-row">
        <span class="d-row-label">Anomaly Risk</span>
        <span class="d-row-val">${ai.unknown_anomaly_risk||ai.unknown_anomaly_risk_pct||'—'}</span>
      </div>
      <div class="d-row">
        <span class="d-row-label">Time to Attack</span>
        <span class="d-row-val" style="color:${riskColor(risk)}">${ai.estimated_time_to_attack||'—'}</span>
      </div>
      <div class="d-row">
        <span class="d-row-label">Confidence</span>
        <span class="d-row-val">${ai.confidence||'—'} (${Math.round((ai.confidence_score||0)*100)}%)</span>
      </div>
    </div>

    <div class="d-sec">
      <div class="d-sec-title">KILL-CHAIN STAGE</div>
      <div class="stages">${stageHtml}</div>
    </div>

    ${classHtml?`<div class="d-sec"><div class="d-sec-title">ATTACK CLASS PROBABILITIES</div>${classHtml}</div>`:''}

    <div class="d-sec">
      <div class="d-sec-title">EVIDENCE SIGNALS</div>
      ${evHtml}
    </div>

    <div class="d-sec">
      <div class="d-sec-title">BEHAVIORAL ANOMALIES</div>
      ${anHtml}
    </div>

    <div class="d-sec">
      <div class="d-sec-title">RECOMMENDATION</div>
      <div class="rec-box ${risk}">${ai.recommendation||'—'}</div>
    </div>
  `;
}

load();
</script>
</body>
</html>"""


class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path in ('/', '/dashboard'):
            self.send_response(200)
            self.send_header('Content-type', 'text/html; charset=utf-8')
            self.end_headers()
            self.wfile.write(HTML.encode('utf-8'))

        elif self.path == '/predictions':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            try:
                lines = PREDICTIONS_FILE.read_text(errors='replace').strip().split('\n')
                preds = []
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        if obj.get('guardian_ai'):
                            preds.append(obj)
                    except Exception:
                        continue
                self.wfile.write(json.dumps(preds).encode())
            except Exception as e:
                self.wfile.write(json.dumps([]).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        ts = datetime.now().strftime('%H:%M:%S')
        print(f"{ts}  [DASHBOARD]  {args[0]} {args[1]}")


if __name__ == '__main__':
    with socketserver.TCPServer(('0.0.0.0', PORT), Handler) as httpd:
        httpd.allow_reuse_address = True
        print(f"\n{'='*55}")
        print(f"  GUARDIAN DASHBOARD — http://0.0.0.0:{PORT}")
        print(f"  Open in browser: http://192.168.253.157:{PORT}")
        print(f"  Auto-refreshes every 30 seconds")
        print(f"{'='*55}\n")
        httpd.serve_forever()
