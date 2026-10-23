"""Headless-Chrome screenshots of the web demo (dev tool, used for docs/img).

Drives Chrome over the DevTools protocol so it can wait for real conditions
(streamed answer finished, PDF page image loaded) instead of fixed sleeps.

    python scripts/screenshot.py OUT.png URL [--wait JS_EXPR] [--width 1360] [--height 1000]
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import json
import shutil
import subprocess
import tempfile
import time
import urllib.request

import websockets

DEFAULT_WAIT = ("(()=>{const o=document.querySelector('#out');if(!o||!o.querySelector('.chips .chip'))return false;"
                "if(document.querySelector('#live'))return false;if(location.search.includes('cite=')&&!document.querySelector('#ctx'))return false;const img=document.querySelector('#pimg');"
                "return !img||(img.complete&&img.naturalWidth>0)})()")


async def shoot(url: str, out: str, wait: str, width: int, height: int, timeout: float, port: int) -> None:
    ver = None
    for _ in range(50):
        try:
            ver = json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}/json/version"))
            break
        except OSError:
            await asyncio.sleep(0.2)
    req = urllib.request.Request(f"http://127.0.0.1:{port}/json/new?about:blank", method="PUT")
    target = json.load(urllib.request.urlopen(req))
    async with websockets.connect(target["webSocketDebuggerUrl"], max_size=50_000_000) as ws:
        n = 0

        async def call(method, **params):
            nonlocal n
            n += 1
            await ws.send(json.dumps({"id": n, "method": method, "params": params}))
            while True:
                msg = json.loads(await ws.recv())
                if msg.get("id") == n:
                    return msg.get("result", {})

        await call("Emulation.setDeviceMetricsOverride", width=width, height=height, deviceScaleFactor=1, mobile=False)
        await call("Page.enable")
        await call("Page.navigate", url=url)
        t0 = time.time()
        while time.time() - t0 < timeout:
            r = await call("Runtime.evaluate", expression=wait, returnByValue=True)
            if r.get("result", {}).get("value") is True:
                break
            await asyncio.sleep(0.3)
        else:
            print("warning: wait condition not met before timeout")
        await asyncio.sleep(0.4)
        shot = await call("Page.captureScreenshot", format="png")
        open(out, "wb").write(base64.b64decode(shot["data"]))
        print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("url")
    ap.add_argument("--wait", default=DEFAULT_WAIT)
    ap.add_argument("--width", type=int, default=1360)
    ap.add_argument("--height", type=int, default=1000)
    ap.add_argument("--timeout", type=float, default=90)
    ap.add_argument("--port", type=int, default=9333)
    a = ap.parse_args()
    chrome = next(c for c in ("google-chrome", "chromium", "chromium-browser") if shutil.which(c))
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as prof:
        proc = subprocess.Popen([chrome, "--headless=new", "--no-sandbox", "--disable-gpu", "--hide-scrollbars",
                                 f"--remote-debugging-port={a.port}", f"--user-data-dir={prof}", "about:blank"],
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            asyncio.run(shoot(a.url, a.out, a.wait, a.width, a.height, a.timeout, a.port))
        finally:
            proc.terminate()
            proc.wait()


if __name__ == "__main__":
    main()
