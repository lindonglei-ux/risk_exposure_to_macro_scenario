import re
from typing import List
TOC_PAT = re.compile(r'(?i)\b(table of contents|list of figures|list of tables)\b')
DOT_LEADER = re.compile(r'[._]{4,}\s*\d+$')
EXHIBIT_OF = re.compile(r'(?i)\bexhibit\s*<?\d+>?\s*of\s*<?\d+>?')
NOISE_PATTERNS = [
    re.compile(r'(?i)^web\s*<\d{4}>'),
    re.compile(r'(?i)mcKinsey\s*&\s*company'),
    re.compile(r'(?i)scan\s*•\s*download\s*•\s*personalize'),
]
def strip_toc_and_pagenums(lines: List[str]) -> List[str]:
    out = []
    for s in lines:
        t = s.strip()
        if not t: continue
        if TOC_PAT.search(t) or DOT_LEADER.search(t) or EXHIBIT_OF.search(t):
            continue
        if any(p.search(t) for p in NOISE_PATTERNS):
            continue
        if re.fullmatch(r'-?\s*\d{1,4}\s*-?', t):  # page numbers
            continue
        out.append(s)
    return out
