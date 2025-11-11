import re
from typing import List

def strip_toc_and_pagenums(lines: List[str]) -> List[str]:
    out = []
    for s in lines:
        t = s.strip()
        if not t:
            continue
        if re.search(r"(?i)\b(table of contents|contents|list of figures|list of tables)\b", t):
            continue
        if re.search(r"[\._]{4,}\s*\d+$", t):
            continue
        if re.fullmatch(r"-?\s*\d{1,4}\s*-?", t):
            continue
        out.append(s)
    return out
