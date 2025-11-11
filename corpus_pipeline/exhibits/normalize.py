from typing import List, Dict

POSITIVE = {"positive","pos","increases","higher","rises","up"}
NEGATIVE = {"negative","neg","decreases","lower","falls","down"}

def normalize_exhibit(raw_text: str, bullets: List[str], macro_lex: Dict[str,list], micro_lex: Dict[str,list]) -> List[dict]:
    text = (raw_text or "") + "\n" + "\n".join(bullets or [])
    out = []
    macro_terms = {k: [k.lower()] + [s.lower() for s in v] for k,v in macro_lex.items()}
    micro_terms = {k: [k.lower()] + [s.lower() for s in v] for k,v in micro_lex.items()}
    low = text.lower()
    for M, m_alts in macro_terms.items():
        if any(a in low for a in m_alts):
            for C, c_alts in micro_terms.items():
                if any(a in low for a in c_alts):
                    rel = None
                    if any(w in low for w in POSITIVE):
                        rel = "positive"
                    if any(w in low for w in NEGATIVE):
                        rel = rel or "negative"
                    out.append({"macro": M, "micro": C, "relation": rel})
    return out
