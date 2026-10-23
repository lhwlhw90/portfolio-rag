"""Dependency-free minimal PDF writer (one text block per page).
Used to generate the sample PDF and PDF test fixtures."""
from __future__ import annotations

import sys
from pathlib import Path


def _esc(s: str) -> str:
    return s.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def make_pdf(pages: list[list[str]]) -> bytes:
    objs: list[bytes] = []
    n = len(pages)
    font_id = 3 + 2 * n
    kids = " ".join(f"{3 + 2 * i} 0 R" for i in range(n))
    objs.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    objs.append(f"<< /Type /Pages /Kids [{kids}] /Count {n} >>".encode())
    for i, lines in enumerate(pages):
        content_id = 4 + 2 * i
        objs.append(f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 {font_id} 0 R >> >> /Contents {content_id} 0 R >>".encode())
        body = "BT /F1 11 Tf 14 TL 50 740 Td " + " ".join(f"({_esc(l)}) Tj T*" for l in lines) + " ET"
        objs.append(f"<< /Length {len(body)} >>\nstream\n{body}\nendstream".encode())
    objs.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, o in enumerate(objs, start=1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + o + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
    out += b"".join(f"{off:010d} 00000 n \n".encode() for off in offsets)
    out += f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    return bytes(out)


SAMPLE_PAGES = [
    ["Acme Robotics - Security Policy (2026)", "",
     "Section 1. Passwords",
     "All employee passwords must be at least 14 characters long.",
     "Passwords must be rotated every 180 days and may not be reused.",
     "Multi-factor authentication (MFA) is mandatory for VPN and email access.",
     "Accounts are locked for 15 minutes after 5 failed login attempts."],
    ["Section 2. Incident response",
     "Security incidents must be reported to security@acme.example within 1 hour of discovery.",
     "The on-call security engineer triages incidents and opens a post-mortem within 5 business days.",
     "Customer data breaches are escalated to the legal team immediately.",
     "Lost or stolen laptops must be reported to the IT helpdesk on the same day."],
    ["Section 3. Devices and data",
     "All laptops must use full-disk encryption and lock the screen after 5 minutes of inactivity.",
     "Confidential data may only be stored on company-managed devices and approved cloud storage.",
     "USB storage devices are blocked on company laptops unless an exception is approved by security.",
     "Data is classified as Public, Internal, Confidential or Strictly Confidential."],
    ["Section 4. Backups and vendors",
     "Production databases are backed up every 6 hours and backups are kept for 35 days.",
     "Backup restore tests are performed every quarter.",
     "Vendors receive time-limited access accounts that expire after 90 days.",
     "Vendor access to production systems requires approval from the system owner."],
]

INSTALL_PAGES = [
    ["RoboArm X2 Installation Manual", "",
     "Chapter 1. Site requirements",
     "The mounting surface must support a static load of at least 2,000 N.",
     "The operating temperature range is 0 to 45 degrees Celsius.",
     "Relative humidity must stay below 85 percent, non-condensing.",
     "Keep a free safety zone of at least 1.2 m around the arm during commissioning."],
    ["Chapter 2. Mounting",
     "The base is fixed with four M10 bolts tightened to a torque of 45 Nm.",
     "The X2 can be mounted on the floor, on a table or on a wall.",
     "Ceiling mounting is not supported because the joint brakes are not rated for it.",
     "The controller cable between arm and controller is 5 m long."],
    ["Chapter 3. Commissioning",
     "After power-on, run the joint calibration wizard in the web dashboard.",
     "The calibration wizard takes about 20 minutes and must be repeated after every collision above 150 N.",
     "Set the tool weight and center of gravity before running any program.",
     "The default dashboard password must be changed during commissioning."],
    ["Chapter 4. Maintenance and error codes",
     "Re-grease the joint gearboxes every 5,000 operating hours.",
     "Inspect the cable harness every 6 months for wear.",
     "Error code E101 indicates overcurrent in a joint motor.",
     "Error code E204 indicates that the emergency stop circuit is open.",
     "Error code E310 indicates a lost connection between controller and arm."],
]

DOCS = {
    "sample_docs/en/security_policy.pdf": SAMPLE_PAGES,
    "sample_docs/en/x2_installation_manual.pdf": INSTALL_PAGES,
}

if __name__ == "__main__":
    for path, pages in DOCS.items():
        Path(path).write_bytes(make_pdf(pages))
        print(f"wrote {path}")
