import argparse, json
from pathlib import Path
from typing import List, Dict, Any
from .extractors.pdf_text import extract_page_text_blocks, num_pages, page_size, get_page_pixmap
from .detectors.captions import find_captions
from .detectors.columns import reorder_two_columns
from .detectors.numeric_tables import keep_by_alpha_ratio
from .detectors.text_tables import mask_text_table_lines
from .detectors.toc_footer import strip_toc_and_pagenums
from .exhibits.regions import compute_exhibit_region
from .exhibits.harvest import harvest_exhibit_text
from .extractors.ocr import ocr_region_from_bytes
from .exhibits.normalize import normalize_exhibit
from .cleaning.normalize_text import clean_text
from .cleaning.segment import segment_text
from .io_utils import write_jsonl

def blocks_to_lines(blocks: List[Dict[str,Any]]) -> List[str]:
    out = []
    for b in blocks:
        for ln in (b.get("text") or "").splitlines():
            if ln.strip():
                out.append(ln)
    return out

def process_pdf(pdf_path: Path, args):
    pages = []
    n = num_pages(pdf_path)
    for i in range(n):
        blocks = extract_page_text_blocks(pdf_path, i)
        if args.columns == 2:
            blocks = reorder_two_columns(blocks)
        lines = blocks_to_lines(blocks)
        if args.drop_tables:
            lines = [ln for ln in lines if keep_by_alpha_ratio(ln, args.alpha_ratio)]
        if args.drop_toc:
            lines = strip_toc_and_pagenums(lines)
        if args.drop_text_tables:
            mask = mask_text_table_lines(lines, args.text_table_min_cols, args.text_table_min_block)
            lines = [ln for ln, m in zip(lines, mask) if not m]
        cleaned = clean_text("\n".join(lines))
        exhibits = []
        if args.extract_exhibits:
            caps = find_captions(blocks)
            width, height = page_size(pdf_path, i)
            for idx, cap in enumerate(caps):
                nxt = caps[idx+1]["bbox"] if idx+1 < len(caps) else None
                region = compute_exhibit_region(cap["bbox"], page_height=height, next_caption_bbox=nxt, min_height=args.exhibit_min_height, page_width=width)
                pix = get_page_pixmap(pdf_path, i, clip=region, dpi=args.ocr_dpi) if args.ocr_exhibits else None
                pngb = pix.tobytes("png") if pix else None
                harvested = harvest_exhibit_text(blocks, region, page_png_bytes=pngb,
                                                ocr_fn=(lambda b: ocr_region_from_bytes(b, psm=args.ocr_psm)) if args.ocr_exhibits else None)
                rec = {"doc": pdf_path.name, "page": i, "exhibit_id": cap.get("ordinal"), "caption": cap.get("text"), "region": region, **harvested}
                if args.normalize_exhibits:
                    macro_lex = json.loads(Path(args.macro_lexicon).read_text(encoding="utf-8")) if args.macro_lexicon and Path(args.macro_lexicon).exists() else {}
                    micro_lex = json.loads(Path(args.micro_lexicon).read_text(encoding="utf-8")) if args.micro_lexicon and Path(args.micro_lexicon).exists() else {}
                    rec["triples"] = normalize_exhibit((harvested.get("raw_text") or "") + "\n" + (harvested.get("ocr_text") or ""), harvested.get("bullets") or [], macro_lex, micro_lex)
                exhibits.append(rec)
        segs = segment_text(cleaned, mode=args.segment_by, max_tokens=args.max_len)
        pages.append({"page": i, "cleaned": cleaned, "segments": segs, "exhibits": exhibits})
    return pages

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_pdf", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--segment_by", choices=["sentence","paragraph"], default="sentence")
    ap.add_argument("--max_len", type=int, default=512)

    ap.add_argument("--drop_tables", action="store_true")
    ap.add_argument("--alpha_ratio", type=float, default=0.2)
    ap.add_argument("--drop_toc", action="store_true")
    ap.add_argument("--drop_text_tables", action="store_true")
    ap.add_argument("--text_table_min_cols", type=int, default=3)
    ap.add_argument("--text_table_min_block", type=int, default=3)
    ap.add_argument("--columns", type=int, default=1)

    ap.add_argument("--extract_exhibits", action="store_true")
    ap.add_argument("--exhibits_dir", default=None)
    ap.add_argument("--exhibit_min_height", type=float, default=120.0)
    ap.add_argument("--ocr_exhibits", action="store_true")
    ap.add_argument("--ocr_dpi", type=int, default=300)
    ap.add_argument("--ocr_psm", type=int, default=6)
    ap.add_argument("--normalize_exhibits", action="store_true")
    ap.add_argument("--macro_lexicon", default=None)
    ap.add_argument("--micro_lexicon", default=None)

    args = ap.parse_args()
    pdf = Path(args.input_pdf)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    pages = process_pdf(pdf, args)
    # narrative
    (out_dir / (pdf.stem + ".txt")).write_text("\n".join(p["cleaned"] for p in pages if p["cleaned"]), encoding="utf-8")
    write_jsonl(out_dir / (pdf.stem + ".jsonl"), [{"page":p["page"], "segment_index":i, "text":s} for p in pages for i,s in enumerate(p["segments"])])
    # exhibits
    if args.extract_exhibits:
        ex_dir = Path(args.exhibits_dir) if args.exhibits_dir else out_dir
        write_jsonl(ex_dir / (pdf.stem + "_exhibits.jsonl"), [e for p in pages for e in p["exhibits"]])
    print("Done.")
if __name__ == "__main__":
    main()
