"""
Capture full view of the Methodology and Limitations card
"""
import os
import time
import base64
import json
import urllib.request
import websocket
import subprocess

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SCREENSHOT_DIR = os.path.join(PROJECT_ROOT, "results", "screenshots")
CHROME_PATH = r"C:\Program Files\Google\Chrome\Application\chrome.exe"

def kill_chrome():
    subprocess.run(["powershell", "-Command", "Stop-Process -Name chrome -Force -ErrorAction SilentlyContinue"], capture_output=True)

def main():
    kill_chrome()
    time.sleep(1)
    profile_dir = os.path.join(PROJECT_ROOT, "data", ".chrome_profile")
    cmd = [
        CHROME_PATH,
        "--headless=new",
        "--remote-debugging-port=9222",
        "--remote-allow-origins=*",
        "--window-size=1440,1200",
        "--disable-gpu",
        "--no-first-run",
        "--no-default-browser-check",
        f"--user-data-dir={profile_dir}",
        "http://127.0.0.1:8000"
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(3)
    
    try:
        targets_json = urllib.request.urlopen("http://127.0.0.1:9222/json").read().decode("utf-8")
        targets = json.loads(targets_json)
        ws_url = targets[0]["webSocketDebuggerUrl"]
        ws = websocket.create_connection(ws_url, timeout=20)
        
        # Scroll to methodology card
        ws.send(json.dumps({"id": 1, "method": "Runtime.evaluate", "params": {"expression": "document.querySelector('.methodology-card').scrollIntoView();"}}))
        ws.recv()
        time.sleep(1)
        
        # Capture screenshot
        ws.send(json.dumps({"id": 2, "method": "Page.captureScreenshot", "params": {"format": "png"}}))
        resp = json.loads(ws.recv())
        data = resp.get("result", {}).get("data")
        if data:
            with open(os.path.join(SCREENSHOT_DIR, "10_methodology_and_limitations.png"), "wb") as f:
                f.write(base64.b64decode(data))
            print("Successfully updated 10_methodology_and_limitations.png!")
        ws.close()
    finally:
        kill_chrome()

if __name__ == "__main__":
    main()
