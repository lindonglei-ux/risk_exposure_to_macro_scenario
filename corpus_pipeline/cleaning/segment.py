import re
from typing import List
def naive_sentence_split(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z(0-9])", text); out = []
    for p in parts:
        if len(p) > 1200: out.extend(re.split(r";\s+", p))
        else: out.append(p)
    return [s.strip() for s in out if s and not s.isspace()]
def segment_text(s: str, mode: str="sentence", max_tokens: int=512) -> List[str]:
    s = s.strip()
    if not s: return []
    if mode == "paragraph":
        parts = [p.strip() for p in re.split(r"\n{2,}|(?<=\.)\n", s) if p.strip()]
    else:
        try:
            import nltk
            try: nltk.data.find("tokenizers/punkt")
            except LookupError: pass
            parts = nltk.sent_tokenize(s)
        except Exception:
            parts = naive_sentence_split(s)
    chunks = []
    for p in parts:
        if len(p) <= max_tokens*6: chunks.append(p)
        else:
            subs = re.split(r"(?<=[;:,])\s+", p); buf = ""
            for sp in subs:
                if len(buf)+len(sp) < max_tokens*6: buf = (buf+" "+sp).strip()
                else:
                    if buf: chunks.append(buf)
                    buf = sp
            if buf: chunks.append(buf)
    return chunks
