"""Generate the Chinese sample PDF with real embedded CJK text.

Uses headless Chrome's print-to-PDF with the open-source Noto Sans CJK SC font
(SIL OFL). Each list entry becomes one PDF page. Dev-time only: the generated
PDF is committed, so neither Chrome nor the font is needed at runtime.

    python scripts/make_zh_pdf.py
"""
from __future__ import annotations

import html
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

PAGES = [
    ("第一章 开箱与安装",
     ["包装内含S7主机、充电座、电源适配器、边刷两个、备用滤网一个和清洁刷一把。",
      "充电座应放在靠墙的平整地面上，左右两侧各留出0.5米、前方留出1.5米的空间。",
      "首次使用前请将主机放在充电座上充电至少6小时。"]),
    ("第二章 连接App",
     ["请在手机应用商店下载“星河智家”App并注册账号。",
      "长按主机上的Wi-Fi键3秒，指示灯快速闪烁蓝色后即进入配网模式。",
      "配网完成后，App会自动推送最新固件，升级过程中请勿移动机器人。"]),
    ("第三章 清扫与地图",
     ["S7首次清扫时会自动建立地图，单层地图最多可保存4张。",
      "在App中可以设置禁区和虚拟墙，机器人不会进入禁区清扫。",
      "安静模式下噪音约为55分贝，适合夜间使用。",
      "地毯识别功能开启后，机器人在地毯上会自动将吸力提升到最大。"]),
    ("第四章 保养与故障排除",
     ["主刷建议每周清理一次缠绕的毛发，每6个月更换一次。",
      "边刷建议每3个月更换一次，传感器窗口每月用干布擦拭一次。",
      "故障代码E3表示驱动轮被卡住，请检查轮子周围是否有异物。",
      "故障代码E7表示尘盒未安装到位，请重新安装尘盒。",
      "如机器人长时间不使用，请充满电后关机，并每3个月补充一次电量。"]),
]

CSS = """@page { size: A4; margin: 25mm 20mm; }
body { font-family: 'Noto Sans CJK SC', sans-serif; font-size: 13pt; line-height: 1.8; }
h1 { font-size: 18pt; } h2 { font-size: 15pt; } .page { page-break-after: always; }
.page:last-child { page-break-after: auto; }"""


def build_html(title: str, pages) -> str:
    parts = []
    for i, (heading, paras) in enumerate(pages):
        head = f"<h1>{html.escape(title)}</h1>" if i == 0 else ""
        body = "".join(f"<p>{html.escape(p)}</p>" for p in paras)
        parts.append(f'<div class="page">{head}<h2>{html.escape(heading)}</h2>{body}</div>')
    return f"<!doctype html><html lang='zh-CN'><meta charset='utf-8'><style>{CSS}</style><body>{''.join(parts)}</body></html>"


def main(out: str = "sample_docs/zh/S7快速入门指南.pdf") -> None:
    chrome = next((c for c in ("google-chrome", "chromium", "chromium-browser") if shutil.which(c)), None)
    if not chrome:
        sys.exit("Chrome/Chromium not found")
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "doc.html"
        src.write_text(build_html("星河 S7 快速入门指南", PAGES), encoding="utf-8")
        subprocess.run([chrome, "--headless=new", "--no-sandbox", "--disable-gpu", "--no-pdf-header-footer",
                        f"--user-data-dir={tmp}/profile", f"--print-to-pdf={Path(out).resolve()}", src.as_uri()],
                       check=True, capture_output=True)
    print(f"wrote {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
