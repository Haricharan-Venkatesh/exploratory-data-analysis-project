"""
Capture authentic, high-resolution screenshots of the running dashboard
using Chrome DevTools Protocol (CDP) over WebSocket.
"""

import os
import sys
import json
import time
import base64
import subprocess
import urllib.request
import websocket

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SCREENSHOT_DIR = os.path.join(PROJECT_ROOT, "results", "screenshots")
os.makedirs(SCREENSHOT_DIR, exist_ok=True)

CHROME_PATH = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
if not os.path.exists(CHROME_PATH):
    CHROME_PATH = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"

_msg_id = 0

def kill_chrome():
    subprocess.run(["powershell", "-Command", "Stop-Process -Name chrome -Force -ErrorAction SilentlyContinue"], capture_output=True)

def start_chrome():
    kill_chrome()
    time.sleep(1)
    profile_dir = os.path.join(PROJECT_ROOT, "data", ".chrome_profile")
    cmd = [
        CHROME_PATH,
        "--headless=new",
        "--remote-debugging-port=9222",
        "--remote-allow-origins=*",
        "--window-size=1440,1100",
        "--disable-gpu",
        "--no-first-run",
        "--no-default-browser-check",
        f"--user-data-dir={profile_dir}",
        "about:blank"
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(2)
    return proc

def send_cdp(ws, method, params=None):
    global _msg_id
    _msg_id += 1
    msg_id = _msg_id
    payload = {"id": msg_id, "method": method, "params": params or {}}
    ws.send(json.dumps(payload))
    while True:
        raw = ws.recv()
        resp = json.loads(raw)
        if resp.get("id") == msg_id:
            if "error" in resp:
                print(f"CDP Error in {method}: {resp['error']}")
            return resp.get("result", {})

def capture_screenshot(ws, filename, clip=None):
    params = {"format": "png"}
    if clip:
        params["clip"] = clip
    res = send_cdp(ws, "Page.captureScreenshot", params)
    data = res.get("data")
    if data:
        img_bytes = base64.b64decode(data)
        out_path = os.path.join(SCREENSHOT_DIR, filename)
        with open(out_path, "wb") as f:
            f.write(img_bytes)
        print(f"Saved screenshot: {filename} ({len(img_bytes)} bytes)")
        return out_path
    else:
        print(f"Failed to capture: {filename}")
        return None

def eval_js(ws, expr):
    return send_cdp(ws, "Runtime.evaluate", {"expression": expr, "returnByValue": True})

def main():
    print(f"Starting browser from: {CHROME_PATH}")
    proc = start_chrome()
    
    try:
        # Get target WebSocket URL
        targets_json = urllib.request.urlopen("http://127.0.0.1:9222/json").read().decode("utf-8")
        targets = json.loads(targets_json)
        ws_url = targets[0]["webSocketDebuggerUrl"]
        print(f"Connecting to CDP WebSocket: {ws_url}")
        
        ws = websocket.create_connection(ws_url, timeout=30)
        
        # Navigate to application
        print("Navigating to http://127.0.0.1:8000 ...")
        send_cdp(ws, "Page.navigate", {"url": "http://127.0.0.1:8000"})
        time.sleep(4)  # Wait for charts & demo games to initialize
        
        # 1. Full Dashboard: Human Master Example
        print("\n[1] Capturing Human Master Example...")
        eval_js(ws, "goToPly(15);")
        time.sleep(1)
        capture_screenshot(ws, "01_dashboard_human_example.png")
        
        # 2. AI Bot Match Example
        print("\n[2] Capturing AI Bot Example...")
        eval_js(ws, "document.getElementById('btn-demo-bot').click();")
        time.sleep(2)
        eval_js(ws, "goToPly(20);")
        time.sleep(1)
        capture_screenshot(ws, "02_ai_bot_example.png")
        
        # 3. Human vs Bot Challenge Example
        print("\n[3] Capturing Human vs Bot Example...")
        eval_js(ws, "document.getElementById('btn-demo-hvb').click();")
        time.sleep(2)
        eval_js(ws, "goToPly(25);")
        time.sleep(1)
        capture_screenshot(ws, "03_human_vs_bot_example.png")
        
        # 4. Uploaded / Paste PGN Drawer View
        print("\n[4] Capturing Uploaded / Paste PGN View...")
        eval_js(ws, "document.getElementById('btn-toggle-paste').click();")
        time.sleep(1)
        capture_screenshot(ws, "04_uploaded_pgn_paste_drawer.png")
        eval_js(ws, "document.getElementById('btn-cancel-paste').click();")
        time.sleep(1)
        
        # Switch back to Human match for component crops
        eval_js(ws, "document.getElementById('btn-demo-human').click();")
        time.sleep(2)
        eval_js(ws, "goToPly(15);")
        time.sleep(1)
        
        # 5. Interactive Chessboard
        print("\n[5] Capturing Chessboard & Playback View...")
        capture_screenshot(ws, "05_chessboard_and_playback.png", clip={"x": 30, "y": 280, "width": 460, "height": 650, "scale": 1})
        
        # 6. Current Prediction Hero & Circular Gauge
        print("\n[6] Capturing Current Prediction & Gauge...")
        capture_screenshot(ws, "06_current_prediction_gauge.png", clip={"x": 510, "y": 280, "width": 880, "height": 220, "scale": 1})
        
        # 7. Behavioral Feature Meters Grid
        print("\n[7] Capturing Behavioral Feature Meters Grid...")
        capture_screenshot(ws, "07_behavioral_feature_meters.png", clip={"x": 510, "y": 500, "width": 880, "height": 380, "scale": 1})
        
        # 8. AI Probability Timeline Chart
        print("\n[8] Capturing Probability Timeline Chart...")
        capture_screenshot(ws, "08_probability_timeline_chart.png", clip={"x": 30, "y": 1050, "width": 680, "height": 280, "scale": 1})
        
        # 9. Move Thinking Time & Volatility Charts
        print("\n[9] Capturing Move-Time & Volatility Charts...")
        capture_screenshot(ws, "09_movetime_and_cv_charts.png", clip={"x": 720, "y": 1050, "width": 680, "height": 280, "scale": 1})
        
        # 10. Empirical Benchmarks & Scientific Limitations Section
        print("\n[10] Capturing Methodology & Limitations Section...")
        capture_screenshot(ws, "10_methodology_and_limitations.png", clip={"x": 30, "y": 1650, "width": 1380, "height": 300, "scale": 1})
        
        ws.close()
        print("\nAll 10 validation screenshots captured successfully!")
    finally:
        kill_chrome()

if __name__ == "__main__":
    main()
