import argparse
from pathlib import Path
from typing import List, Dict, Any
from .extractors.pdf_text import extract_page_text_blocks, num_pages
from .detectors.columns import reorder_two_columns
from .detectors.numeric_tables import keep_by_alpha_ratio
from .detectors.text_tables import mask_text_table_lines
from .detectors.toc_footer import strip_toc_and_pagenums
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
        segs = segment_text(cleaned, mode=args.segment_by, max_tokens=args.max_len)
        pages.append({"page": i, "cleaned": cleaned, "segments": segs})
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

    args = ap.parse_args()
    pdf = Path(args.input_pdf)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    pages = process_pdf(pdf, args)
    # narrative
    (out_dir / (pdf.stem + ".txt")).write_text("\n".join(p["cleaned"] for p in pages if p["cleaned"]), encoding="utf-8")
    write_jsonl(out_dir / (pdf.stem + ".jsonl"), [{"page":p["page"], "segment_index":i, "text":s} for p in pages for i,s in enumerate(p["segments"])])
    print("Done.")
if __name__ == "__main__":
    main()
