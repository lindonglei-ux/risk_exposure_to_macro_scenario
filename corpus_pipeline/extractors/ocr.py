from typing import Optional
try:
    import pytesseract
    from PIL import Image
    have_ocr = True
except Exception:
    have_ocr = False

def ocr_region_from_bytes(png_bytes: bytes, psm: int = 6) -> str:
    if not have_ocr:
        return ""
    cfg = f"--psm {psm}"
    try:
        import io
        img = Image.open(io.BytesIO(png_bytes))
        return pytesseract.image_to_string(img, config=cfg)
    except Exception:
        return ""
