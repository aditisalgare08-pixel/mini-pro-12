"""
Motion Detection System - Single File Flask Project
Python + OpenCV + NumPy + Flask + SQLite + Optional Pandas/Matplotlib/Tkinter
Fixed version: Matplotlib is NOT imported at startup, so Python 3.14 ft2font errors won't stop the app.
"""

import os
import io
import sys
import cv2
import time
import json
import math
import base64
import sqlite3
import threading
import webbrowser
from datetime import datetime, date
from pathlib import Path

import numpy as np
from flask import Flask, Response, jsonify, request, redirect, url_for, send_file

APP_NAME = "Motion Detection System"
BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "motion_logs.db"
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024

settings = {
    "camera_index": 0,
    "min_area": 850,
    "threshold": 25,
    "blur": 21,
    "show_mask": False,
    "logging_gap": 3.0,
    "demo_mode": False,
}

state = {
    "running": True,
    "last_motion_time": 0,
    "last_log_time": 0,
    "motion_now": False,
    "camera_ok": False,
    "frames": 0,
    "fps": 0,
    "last_frame_time": time.time(),
    "started_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "last_event": None,
}

camera_lock = threading.Lock()


def connect_db():
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    return con


def init_db():
    con = connect_db()
    cur = con.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS motion_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_time TEXT NOT NULL,
            camera TEXT NOT NULL,
            motion_area INTEGER NOT NULL,
            boxes INTEGER NOT NULL,
            note TEXT DEFAULT 'Motion detected'
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS system_notes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            note TEXT NOT NULL
        )
        """
    )
    con.commit()
    con.close()


def log_motion(area: int, boxes: int, camera: str = "webcam"):
    now = time.time()
    if now - state["last_log_time"] < settings["logging_gap"]:
        return
    state["last_log_time"] = now
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    state["last_event"] = ts
    con = connect_db()
    con.execute(
        "INSERT INTO motion_events(event_time, camera, motion_area, boxes) VALUES(?,?,?,?)",
        (ts, camera, int(area), int(boxes)),
    )
    con.commit()
    con.close()


def fetch_stats():
    con = connect_db()
    total = con.execute("SELECT COUNT(*) FROM motion_events").fetchone()[0]
    today_text = date.today().strftime("%Y-%m-%d")
    today_count = con.execute(
        "SELECT COUNT(*) FROM motion_events WHERE event_time LIKE ?", (today_text + "%",)
    ).fetchone()[0]
    last = con.execute(
        "SELECT event_time, motion_area, boxes FROM motion_events ORDER BY id DESC LIMIT 1"
    ).fetchone()
    recent_rows = con.execute(
        "SELECT id, event_time, camera, motion_area, boxes FROM motion_events ORDER BY id DESC LIMIT 12"
    ).fetchall()
    hourly_rows = con.execute(
        """
        SELECT substr(event_time, 12, 2) AS hour, COUNT(*) AS count
        FROM motion_events
        WHERE event_time LIKE ?
        GROUP BY substr(event_time, 12, 2)
        ORDER BY hour
        """,
        (today_text + "%",),
    ).fetchall()
    con.close()

    recent = [dict(r) for r in recent_rows]
    hourly = [{"hour": r["hour"] + ":00", "count": r["count"]} for r in hourly_rows]
    if not hourly:
        hourly = [{"hour": f"{h:02d}:00", "count": 0} for h in range(0, 24, 3)]
    return {
        "total": total,
        "today": today_count,
        "last": dict(last) if last else None,
        "recent": recent,
        "hourly": hourly,
        "motion_now": state["motion_now"],
        "camera_ok": state["camera_ok"],
        "fps": round(state["fps"], 1),
        "frames": state["frames"],
        "started_at": state["started_at"],
        "settings": settings,
    }


def fetch_hourly_with_optional_pandas():
    """Uses Pandas if available; falls back to SQLite/Python if Pandas is missing."""
    try:
        import pandas as pd  # optional
        con = connect_db()
        df = pd.read_sql_query("SELECT event_time FROM motion_events", con)
        con.close()
        if df.empty:
            return []
        df["event_time"] = pd.to_datetime(df["event_time"])
        df = df[df["event_time"].dt.date == date.today()]
        if df.empty:
            return []
        grouped = df.groupby(df["event_time"].dt.strftime("%H:00")).size().reset_index(name="count")
        grouped.columns = ["hour", "count"]
        return grouped.to_dict(orient="records")
    except Exception:
        return fetch_stats()["hourly"]


def detect_motion(frame, back_subtractor):
    blur_size = int(settings["blur"])
    if blur_size % 2 == 0:
        blur_size += 1
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (blur_size, blur_size), 0)
    fgmask = back_subtractor.apply(gray)
    _, thresh = cv2.threshold(fgmask, int(settings["threshold"]), 255, cv2.THRESH_BINARY)
    thresh = cv2.dilate(thresh, None, iterations=2)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    total_area = 0
    min_area = int(settings["min_area"])
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        boxes.append((x, y, w, h, area))
        total_area += area

    motion = len(boxes) > 0
    state["motion_now"] = motion
    if motion:
        state["last_motion_time"] = time.time()
        log_motion(int(total_area), len(boxes))

    display = frame.copy()
    for x, y, w, h, area in boxes:
        cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 120), 2)
        cv2.putText(display, f"Motion {int(area)}", (x, max(20, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 120), 2)

    header_color = (0, 40, 255) if motion else (0, 170, 80)
    status_text = "MOTION DETECTED" if motion else "NO MOTION"
    cv2.rectangle(display, (0, 0), (display.shape[1], 52), (8, 12, 22), -1)
    cv2.putText(display, status_text, (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.85, header_color, 2)
    cv2.putText(display, f"FPS {state['fps']:.1f}", (display.shape[1] - 130, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    if settings["show_mask"]:
        mask_bgr = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
        mask_bgr = cv2.resize(mask_bgr, (display.shape[1] // 4, display.shape[0] // 4))
        mh, mw = mask_bgr.shape[:2]
        display[display.shape[0]-mh-10:display.shape[0]-10, 10:10+mw] = mask_bgr
        cv2.rectangle(display, (10, display.shape[0]-mh-10), (10+mw, display.shape[0]-10), (255, 255, 255), 1)

    return display


def make_demo_frame(tick):
    frame = np.zeros((480, 760, 3), dtype=np.uint8)
    frame[:] = (17, 24, 39)
    for i in range(0, 760, 40):
        cv2.line(frame, (i, 0), (i, 480), (26, 36, 56), 1)
    for j in range(0, 480, 40):
        cv2.line(frame, (0, j), (760, j), (26, 36, 56), 1)
    x = int((math.sin(tick / 18) + 1) * 300) + 50
    y = int((math.cos(tick / 22) + 1) * 160) + 110
    cv2.rectangle(frame, (x, y), (x + 120, y + 80), (25, 180, 255), -1)
    cv2.circle(frame, (x + 60, y + 40), 26, (255, 255, 255), -1)
    cv2.putText(frame, "DEMO CAMERA - Motion Object", (25, 455), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (230, 230, 230), 2)
    return frame


def generate_frames():
    back_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=28, detectShadows=True)
    cap = None
    tick = 0
    while state["running"]:
        frame = None
        use_demo = bool(settings["demo_mode"])
        if not use_demo:
            if cap is None or not cap.isOpened():
                with camera_lock:
                    cap = cv2.VideoCapture(int(settings["camera_index"]), cv2.CAP_DSHOW)
                    if not cap.isOpened():
                        cap = cv2.VideoCapture(int(settings["camera_index"]))
                time.sleep(0.1)
            if cap is not None and cap.isOpened():
                ok, frame = cap.read()
                state["camera_ok"] = bool(ok)
                if not ok:
                    frame = None
            else:
                state["camera_ok"] = False

        if frame is None:
            settings["demo_mode"] = True
            frame = make_demo_frame(tick)
            state["camera_ok"] = False

        frame = cv2.resize(frame, (760, 480))
        display = detect_motion(frame, back_subtractor)
        state["frames"] += 1
        now = time.time()
        if now - state["last_frame_time"] >= 1:
            state["fps"] = state["frames"] / max(1, now - state["last_frame_time"])
            state["frames"] = 0
            state["last_frame_time"] = now

        ok, buffer = cv2.imencode(".jpg", display, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
        if not ok:
            continue
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
        tick += 1
        time.sleep(0.03)

    if cap is not None:
        cap.release()


HTML_PAGE = r'''
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Motion Detection System</title>
<style>
:root{--bg:#08111f;--card:#101b2e;--card2:#0d1628;--text:#e7eefc;--muted:#90a4c4;--accent:#34d399;--danger:#fb7185;--blue:#60a5fa;--border:#21314d;}
*{box-sizing:border-box} body{margin:0;font-family:Inter,Segoe UI,Arial,sans-serif;background:radial-gradient(circle at top,#12233e,#060b14 70%);color:var(--text)}
.header{padding:24px;display:flex;justify-content:space-between;align-items:center;border-bottom:1px solid var(--border);background:rgba(8,17,31,.8);backdrop-filter:blur(12px);position:sticky;top:0;z-index:5}
.logo{font-size:25px;font-weight:800}.logo span{color:var(--accent)}.badge{padding:9px 14px;border-radius:999px;background:#13233d;color:#c7d2fe;border:1px solid var(--border)}
.wrap{max-width:1220px;margin:0 auto;padding:24px}.grid{display:grid;grid-template-columns:2fr 1fr;gap:22px}.card{background:linear-gradient(180deg,var(--card),var(--card2));border:1px solid var(--border);border-radius:22px;box-shadow:0 20px 60px rgba(0,0,0,.28);overflow:hidden}.card h2{margin:0;padding:18px 20px;border-bottom:1px solid var(--border);font-size:18px}.video{width:100%;display:block;background:#000;min-height:320px}.controls{display:grid;grid-template-columns:repeat(2,1fr);gap:12px;padding:18px}.control{background:#0b1425;border:1px solid var(--border);border-radius:16px;padding:12px}.control label{display:block;font-size:13px;color:var(--muted);margin-bottom:8px}.control input,.control select{width:100%;background:#08111f;border:1px solid #283a5e;color:var(--text);border-radius:10px;padding:10px}button{border:0;border-radius:12px;padding:11px 14px;font-weight:700;color:#06131f;background:var(--accent);cursor:pointer}button.secondary{background:#243653;color:var(--text)}button.danger{background:var(--danger);color:white}.stats{display:grid;grid-template-columns:repeat(2,1fr);gap:12px;padding:18px}.stat{background:#0b1425;border:1px solid var(--border);border-radius:18px;padding:16px}.stat .n{font-size:28px;font-weight:900}.stat .t{color:var(--muted);font-size:13px;margin-top:6px}.live{display:inline-block;width:10px;height:10px;border-radius:50%;background:var(--accent);box-shadow:0 0 18px var(--accent);margin-right:8px}.live.red{background:var(--danger);box-shadow:0 0 18px var(--danger)}.list{padding:0 18px 18px;max-height:260px;overflow:auto}.event{display:flex;justify-content:space-between;gap:12px;padding:11px 0;border-bottom:1px solid var(--border);font-size:13px}.event small{color:var(--muted)}.chart{width:100%;min-height:230px;background:#0b1425;border:1px solid var(--border);border-radius:18px;padding:12px}.footer{padding:18px;color:var(--muted);font-size:13px;border-top:1px solid var(--border)}.upload{padding:18px;display:flex;gap:10px;flex-wrap:wrap}.toast{position:fixed;right:20px;bottom:20px;background:#101b2e;border:1px solid var(--border);padding:14px 18px;border-radius:14px;display:none} @media(max-width:900px){.grid{grid-template-columns:1fr}.controls,.stats{grid-template-columns:1fr}.header{display:block}.badge{display:inline-block;margin-top:10px}}
</style>
</head>
<body>
<div class="header"><div class="logo">🎥 Motion <span>Detection</span> System</div><div class="badge"><span id="dot" class="live"></span><span id="status">Loading...</span></div></div>
<div class="wrap"><div class="grid"><div class="card"><h2>Live Camera Feed</h2><img class="video" src="/video_feed" alt="Live motion detection"><div class="controls"><div class="control"><label>Minimum Motion Area</label><input id="min_area" type="number" value="850"></div><div class="control"><label>Threshold</label><input id="threshold" type="number" value="25"></div><div class="control"><label>Blur Size</label><input id="blur" type="number" value="21"></div><div class="control"><label>Camera Mode</label><select id="demo_mode"><option value="false">Real Webcam</option><option value="true">Demo Mode</option></select></div><button onclick="saveSettings()">Save Settings</button><button class="secondary" onclick="refreshNow()">Refresh Stats</button><button class="danger" onclick="clearLogs()">Clear Logs</button><button class="secondary" onclick="toggleMask()">Show/Hide Mask</button></div><div class="upload"><form id="uploadForm"><input type="file" name="video" accept="video/*"><button type="submit">Analyze Video File</button></form></div></div><div class="card"><h2>Dashboard</h2><div class="stats"><div class="stat"><div id="total" class="n">0</div><div class="t">Total Events</div></div><div class="stat"><div id="today" class="n">0</div><div class="t">Today Events</div></div><div class="stat"><div id="fps" class="n">0</div><div class="t">FPS</div></div><div class="stat"><div id="camera" class="n">--</div><div class="t">Camera</div></div></div><div style="padding:0 18px 18px"><img id="chart" class="chart" src="/chart.svg"></div></div></div><div class="grid" style="margin-top:22px"><div class="card"><h2>Recent Motion Logs</h2><div id="events" class="list"></div></div><div class="card"><h2>Project Features</h2><div class="footer">✅ Python Flask Web App<br>✅ OpenCV Motion Detection<br>✅ NumPy Frame Processing<br>✅ SQLite Database Logs<br>✅ Optional Pandas Analytics<br>✅ Optional Matplotlib Export without startup crash<br>✅ HTML CSS JavaScript Dashboard<br>✅ Tkinter Launcher using <b>python app.py --gui</b></div></div></div></div><div id="toast" class="toast"></div>
<script>
function toast(msg){let t=document.getElementById('toast');t.innerText=msg;t.style.display='block';setTimeout(()=>t.style.display='none',2500)}
async function refreshNow(){let r=await fetch('/stats');let d=await r.json();document.getElementById('total').innerText=d.total;document.getElementById('today').innerText=d.today;document.getElementById('fps').innerText=d.fps;document.getElementById('camera').innerText=d.camera_ok?'OK':'DEMO';document.getElementById('status').innerText=d.motion_now?'Motion detected':'No motion';document.getElementById('dot').className=d.motion_now?'live red':'live';document.getElementById('min_area').value=d.settings.min_area;document.getElementById('threshold').value=d.settings.threshold;document.getElementById('blur').value=d.settings.blur;document.getElementById('demo_mode').value=String(d.settings.demo_mode);let html='';d.recent.forEach(e=>{html+=`<div class="event"><div><b>${e.event_time}</b><br><small>${e.camera}</small></div><div>Area: ${e.motion_area}<br><small>Boxes: ${e.boxes}</small></div></div>`});document.getElementById('events').innerHTML=html||'<div class="event">No events yet</div>';document.getElementById('chart').src='/chart.svg?t='+Date.now()}
async function saveSettings(){let body={min_area:+min_area.value,threshold:+threshold.value,blur:+blur.value,demo_mode:demo_mode.value==='true'};await fetch('/settings',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});toast('Settings saved')}
async function toggleMask(){await fetch('/toggle_mask',{method:'POST'});toast('Mask setting changed')}
async function clearLogs(){if(!confirm('Delete all motion logs?'))return;await fetch('/clear',{method:'POST'});toast('Logs cleared');refreshNow()}
document.getElementById('uploadForm').addEventListener('submit',async e=>{e.preventDefault();let fd=new FormData(e.target);toast('Analyzing video...');let r=await fetch('/upload',{method:'POST',body:fd});let d=await r.json();toast(d.message);refreshNow()});
refreshNow();setInterval(refreshNow,3000);
</script></body></html>
'''


@app.route("/")
def index():
    return HTML_PAGE


@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/stats")
def stats():
    return jsonify(fetch_stats())


@app.route("/settings", methods=["POST"])
def update_settings():
    data = request.get_json(force=True, silent=True) or {}
    for key in ["min_area", "threshold", "blur", "camera_index"]:
        if key in data:
            try:
                settings[key] = int(data[key])
            except Exception:
                pass
    if "demo_mode" in data:
        settings["demo_mode"] = bool(data["demo_mode"])
    return jsonify({"ok": True, "settings": settings})


@app.route("/toggle_mask", methods=["POST"])
def toggle_mask():
    settings["show_mask"] = not settings["show_mask"]
    return jsonify({"ok": True, "show_mask": settings["show_mask"]})


@app.route("/clear", methods=["POST"])
def clear_logs():
    con = connect_db()
    con.execute("DELETE FROM motion_events")
    con.commit()
    con.close()
    return jsonify({"ok": True})


@app.route("/chart.svg")
def chart_svg():
    data = fetch_hourly_with_optional_pandas()
    if not data:
        data = [{"hour": f"{h:02d}:00", "count": 0} for h in range(0, 24, 3)]
    width, height = 780, 260
    pad = 42
    max_count = max([d["count"] for d in data] + [1])
    bar_gap = 8
    bar_w = max(12, int((width - 2 * pad) / max(1, len(data))) - bar_gap)
    bars = []
    labels = []
    for i, row in enumerate(data):
        x = pad + i * ((width - 2 * pad) / max(1, len(data)))
        h = int((height - 2 * pad) * row["count"] / max_count)
        y = height - pad - h
        bars.append(f'<rect x="{x:.1f}" y="{y}" width="{bar_w}" height="{h}" rx="7" fill="#34d399" opacity="0.9"/>')
        if i % max(1, len(data)//8) == 0:
            labels.append(f'<text x="{x:.1f}" y="{height-16}" fill="#90a4c4" font-size="12">{row["hour"]}</text>')
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}"><rect width="100%" height="100%" rx="18" fill="#0b1425"/><text x="26" y="28" fill="#e7eefc" font-size="18" font-family="Arial" font-weight="700">Today's hourly motion events</text><line x1="{pad}" y1="{height-pad}" x2="{width-pad}" y2="{height-pad}" stroke="#263957"/><line x1="{pad}" y1="{pad}" x2="{pad}" y2="{height-pad}" stroke="#263957"/>{''.join(bars)}{''.join(labels)}<text x="{width-170}" y="28" fill="#90a4c4" font-size="13" font-family="Arial">Max: {max_count}</text></svg>'''
    return Response(svg, mimetype="image/svg+xml")


@app.route("/matplotlib_chart")
def matplotlib_chart():
    """Optional Matplotlib route. If Matplotlib is broken, app still works."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        data = fetch_hourly_with_optional_pandas()
        labels = [d["hour"] for d in data]
        values = [d["count"] for d in data]
        fig, ax = plt.subplots(figsize=(9, 3))
        ax.bar(labels, values)
        ax.set_title("Motion Events by Hour")
        ax.set_xlabel("Hour")
        ax.set_ylabel("Count")
        plt.xticks(rotation=35, ha="right")
        plt.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        plt.close(fig)
        buf.seek(0)
        return send_file(buf, mimetype="image/png")
    except Exception as exc:
        return jsonify({"ok": False, "message": "Matplotlib optional chart failed", "error": str(exc)}), 200


@app.route("/upload", methods=["POST"])
def upload_video():
    if "video" not in request.files:
        return jsonify({"ok": False, "message": "No video selected"}), 400
    file = request.files["video"]
    if not file.filename:
        return jsonify({"ok": False, "message": "No file selected"}), 400
    safe_name = "uploaded_" + datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + os.path.basename(file.filename)
    path = UPLOAD_DIR / safe_name
    file.save(path)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return jsonify({"ok": False, "message": "Could not open video"}), 400
    subtractor = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=28, detectShadows=True)
    frames_checked = 0
    events = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames_checked += 1
        if frames_checked % 8 != 0:
            continue
        frame = cv2.resize(frame, (760, 480))
        before = state["last_log_time"]
        detect_motion(frame, subtractor)
        if state["last_log_time"] != before:
            events += 1
    cap.release()
    return jsonify({"ok": True, "message": f"Video analyzed: {events} motion events saved"})


def open_browser_later(port):
    time.sleep(1.4)
    webbrowser.open(f"http://127.0.0.1:{port}")


def start_gui():
    try:
        import tkinter as tk
        from tkinter import messagebox
    except Exception:
        print("Tkinter not available. Starting web app only.")
        run_server(open_browser=True)
        return

    def start_web():
        threading.Thread(target=lambda: run_server(open_browser=True), daemon=True).start()
        messagebox.showinfo(APP_NAME, "Server started. Browser will open automatically.")

    root = tk.Tk()
    root.title(APP_NAME)
    root.geometry("420x240")
    root.configure(bg="#08111f")
    tk.Label(root, text="Motion Detection System", fg="#e7eefc", bg="#08111f", font=("Segoe UI", 18, "bold")).pack(pady=24)
    tk.Label(root, text="Click Start and use the web dashboard", fg="#90a4c4", bg="#08111f", font=("Segoe UI", 11)).pack()
    tk.Button(root, text="Start Web App", command=start_web, bg="#34d399", fg="#06131f", font=("Segoe UI", 12, "bold"), width=18).pack(pady=24)
    tk.Label(root, text="URL: http://127.0.0.1:5000", fg="#60a5fa", bg="#08111f").pack()
    root.mainloop()


def run_server(open_browser=False):
    init_db()
    port = int(os.environ.get("PORT", "5000"))
    if open_browser:
        threading.Thread(target=open_browser_later, args=(port,), daemon=True).start()
    print("=" * 60)
    print(f"{APP_NAME} running")
    print(f"Open: http://127.0.0.1:{port}")
    print("Fixed build: Matplotlib/Pandas are optional and will not crash startup.")
    print("=" * 60)
    app.run(host="127.0.0.1", port=port, debug=False, threaded=True)


if __name__ == "__main__":
    if "--gui" in sys.argv:
        start_gui()
    else:
        run_server(open_browser=True)
