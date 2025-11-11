from typing import List, Dict, Any, Tuple, Optional

def _intersects(bbox: Tuple[float,float,float,float], region: Tuple[float,float,float,float]) -> bool:
    x0,y0,x1,y1 = bbox
    rx0,ry0,rx1,ry1 = region
    return not (x1 < rx0 or x0 > rx1 or y1 < ry0 or y0 > ry1)

def harvest_exhibit_text(blocks: List[Dict[str,Any]], region: Tuple[float,float,float,float],
                         page_png_bytes: Optional[bytes] = None,
                         ocr_fn = None) -> dict:
    body_blocks = [b for b in blocks if _intersects((b["x0"],b["y0"],b["x1"],b["y1"]), region)]
    text = "\n".join((b.get("text") or "").strip() for b in body_blocks if (b.get("text") or "").strip())
    ocr_text = None
    if len(text) < 40 and page_png_bytes and ocr_fn:
        ocr_text = ocr_fn(page_png_bytes)
    src = text or (ocr_text or "")
    bullets = [ln.strip("•-–— ").strip() for ln in src.splitlines() if ln.strip().startswith(("•","-","–","—"))]
    return {"raw_text": text, "ocr_text": ocr_text, "bullets": bullets}
