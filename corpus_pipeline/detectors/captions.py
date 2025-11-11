import re
from typing import List, Dict, Any

CAPTION_PAT = re.compile(r'(?i)\b(?:exhibit|figure|table|chart)\s*\d+\s*[:\-–]\s*')

def find_captions(blocks: List[Dict[str,Any]]) -> List[Dict[str,Any]]:
    caps = []
    for b in blocks:
        t = (b.get("text") or "").strip()
        if CAPTION_PAT.search(t):
            m = re.search(r'(?i)(exhibit|figure|table|chart)\s*(\d+)', t)
            ordn = int(m.group(2)) if m else None
            kind = m.group(1).lower() if m else "caption"
            caps.append({"bbox": (b["x0"], b["y0"], b["x1"], b["y1"]), "text": t, "kind": kind, "ordinal": ordn})
    caps.sort(key=lambda c: c["bbox"][1])
    return caps
