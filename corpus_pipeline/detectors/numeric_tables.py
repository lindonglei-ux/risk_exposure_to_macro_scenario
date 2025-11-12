import re
def is_caption_line(s: str) -> bool:
    return bool(re.search(r'(?i)\b(?:exhibit|figure|table|chart)\s*\d+\s*[:\-–]\s*', s))
def keep_by_alpha_ratio(line: str, min_alpha_ratio: float = 0.2) -> bool:
    s = line.strip()
    if not s: return False
    if is_caption_line(s): return True
    alpha = sum(1 for c in s if c.isalpha())
    ratio = alpha / max(len(s), 1)
    return ratio >= min_alpha_ratio
