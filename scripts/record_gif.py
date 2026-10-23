"""Record the demo GIF (docs/img/demo.gif) by driving the web UI in headless Chrome over the
DevTools protocol and stitching screenshots with Pillow. Requires a running server with the
sample documents indexed (see README "Demo GIF").

    python scripts/record_gif.py docs/img/demo.gif --base http://127.0.0.1:8000
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import shutil
import subprocess
import tempfile
import time
import urllib.request

import websockets
from PIL import Image

# (question, answerer, citation to open or 0) — an answer with PDF evidence, a heading-context
# answer, and a refusal
SCENES = [
    ("Whom do I notify after a security incident and how quickly?", "extractive", 1),
    ("When was the first X2 firmware released?", "extractive", 0),
    ("What is the price of the RoboArm X2?", "extractive", 0),
]
DONE = "(()=>!!document.querySelector('#out .chips .chip')&&!document.querySelector('#live')&&!document.querySelector('#go').disabled)()"
VIEWER = "(()=>{const img=document.querySelector('#pimg');return !!document.querySelector('#ctx')&&(!img||(img.complete&&img.naturalWidth>0))})()"


class Recorder:
    def __init__(self, ws, width: int):
        self.ws, self.n, self.frames, self.width = ws, 0, [], width

    async def call(self, method, **params):
        self.n += 1
        await self.ws.send(json.dumps({"id": self.n, "method": method, "params": params}))
        while True:
            msg = json.loads(await self.ws.recv())
            if msg.get("id") == self.n:
                return msg.get("result", {})

    async def js(self, expr):
        r = await self.call("Runtime.evaluate", expression=expr, returnByValue=True, awaitPromise=True)
        return r.get("result", {}).get("value")

    async def frame(self, ms: int):
        shot = await self.call("Page.captureScreenshot", format="png")
        img = Image.open(io.BytesIO(base64.b64decode(shot["data"]))).convert("RGB")
        img = img.resize((self.width, round(img.height * self.width / img.width)), Image.LANCZOS)
        if self.frames and list(img.getdata()) == list(self.frames[-1][0].getdata()):
            self.frames[-1] = (self.frames[-1][0], self.frames[-1][1] + ms)  # merge identical frames
        else:
            self.frames.append((img, ms))

    async def wait(self, expr, timeout=60, every_ms=0):
        t0 = time.time()
        while time.time() - t0 < timeout:
            if await self.js(expr):
                return True
            if every_ms:
                await self.frame(every_ms)
            await asyncio.sleep(0.15)
        return False


async def record(base: str, out: str, width: int, port: int, view_w: int, view_h: int):
    for _ in range(50):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/json/version")
            break
        except OSError:
            await asyncio.sleep(0.2)
    target = json.load(urllib.request.urlopen(urllib.request.Request(
        f"http://127.0.0.1:{port}/json/new?about:blank", method="PUT")))
    async with websockets.connect(target["webSocketDebuggerUrl"], max_size=50_000_000) as ws:
        r = Recorder(ws, width)
        await r.call("Emulation.setDeviceMetricsOverride", width=view_w, height=view_h, deviceScaleFactor=1, mobile=False)
        await r.call("Page.enable")
        await r.call("Page.navigate", url=base + "/")
        await r.wait("document.querySelector('#health')&&document.querySelector('#health').textContent.includes('chunks')")
        await r.frame(1200)
        for q, answerer, cite in SCENES:
            await r.js(f"document.querySelector('#answerer').value={json.dumps(answerer)};document.querySelector('#q').value=''")
            for i in range(0, len(q) + 1, 4):  # "typing"
                await r.js(f"document.querySelector('#q').value={json.dumps(q[:i])}")
                await r.frame(60)
            await r.js(f"document.querySelector('#q').value={json.dumps(q)}")
            await r.frame(400)
            await r.js("window.__last=null;document.querySelector('#askf').requestSubmit()")
            await asyncio.sleep(0.2)
            await r.wait(DONE, every_ms=200)
            await r.frame(2600)
            if cite:
                await r.js(f"openCitation({cite})")
                await r.wait(VIEWER)
                await asyncio.sleep(0.3)
                await r.js("window.scrollTo(0, document.querySelector('#viewer').offsetTop-20)")
                await r.frame(3200)
                await r.js("window.scrollTo(0,0)")
    imgs = [f.quantize(colors=128, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE) for f, _ in r.frames]
    imgs[0].save(out, save_all=True, append_images=imgs[1:], duration=[ms for _, ms in r.frames], loop=0, optimize=True)
    print(f"wrote {out}: {len(imgs)} frames, {sum(ms for _, ms in r.frames) / 1000:.1f} s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--width", type=int, default=900, help="GIF width in px")
    ap.add_argument("--view", default="1280x900", help="browser viewport")
    ap.add_argument("--port", type=int, default=9334)
    a = ap.parse_args()
    vw, vh = map(int, a.view.split("x"))
    chrome = next(c for c in ("google-chrome", "chromium", "chromium-browser") if shutil.which(c))
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as prof:
        proc = subprocess.Popen([chrome, "--headless=new", "--no-sandbox", "--disable-gpu", "--hide-scrollbars",
                                 f"--remote-debugging-port={a.port}", f"--user-data-dir={prof}", "about:blank"],
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            asyncio.run(record(a.base, a.out, a.width, a.port, vw, vh))
        finally:
            proc.terminate()
            proc.wait()


if __name__ == "__main__":
    main()
