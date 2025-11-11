from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path

have_fitz = False
try:
    import fitz  # PyMuPDF
    have_fitz = True
except Exception:
    pass

def num_pages(pdf_path: Path) -> int:
    if not have_fitz: return 0
    with fitz.open(str(pdf_path)) as doc:
        return doc.page_count

def get_page_pixmap(pdf_path: Path, page_idx: int, clip: Optional[Tuple[float,float,float,float]] = None, dpi: int = 300):
    if not have_fitz: return None
    doc = fitz.open(str(pdf_path))
    page = doc.load_page(page_idx)
    mat = fitz.Matrix(dpi/72, dpi/72)
    if clip:
        rect = fitz.Rect(*clip)
        pix = page.get_pixmap(matrix=mat, clip=rect, alpha=False)
    else:
        pix = page.get_pixmap(matrix=mat, alpha=False)
    doc.close()
    return pix

def extract_page_text_blocks(pdf_path: Path, page_idx: int) -> List[Dict[str, Any]]:
    if not have_fitz: return []
    blocks_out = []
    with fitz.open(str(pdf_path)) as doc:
        page = doc.load_page(page_idx)
        for i, b in enumerate(page.get_text("blocks")):
            x0, y0, x1, y1, text, *_ = b
            blocks_out.append({"x0":x0,"y0":y0,"x1":x1,"y1":y1,"text":text or "", "block_no": i})
    return blocks_out

def page_size(pdf_path: Path, page_idx: int) -> Tuple[float,float]:
    if not have_fitz: return (0.0,0.0)
    with fitz.open(str(pdf_path)) as doc:
        page = doc.load_page(page_idx)
        r = page.rect
        return (r.width, r.height)
