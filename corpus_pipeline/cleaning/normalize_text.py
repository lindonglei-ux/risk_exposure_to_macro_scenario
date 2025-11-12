import re
LIGATURES = {"\ufb01":"fi","\ufb02":"fl","\u00ad":""}
def normalize_ligatures(s:str)->str:
    for k,v in LIGATURES.items(): s = s.replace(k,v)
    return s
def fix_linebreak_hyphens(s:str)->str:
    return re.sub(r'(\w)-\s*\n\s*(\w)', r'\1\2', s)
def clean_text(t: str) -> str:
    if not t: return ""
    t = normalize_ligatures(t)
    t = fix_linebreak_hyphens(t)
    t = t.replace("\xa0", " ")
    t = re.sub(r"(?<=[a-z])(?=[A-Z][a-z])", " ", t)
    t = re.sub(r"(?<=[a-zA-Z])(?=[0-9])", " ", t)
    t = re.sub(r"(?<=[0-9])(?=[A-Za-z])", " ", t)
    t = re.sub(r"(?<=[A-Za-z0-9])(?=\()", " ", t)
    t = re.sub(r"(?<=\))(?=[A-Za-z0-9])", " ", t)
    t = re.sub(r"[‐–—]+", "-", t)
    t = re.sub(r"[•·∙◦*]+", "-", t)
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"\.\s*([A-Z])", r". \1", t)
    return t.strip()
