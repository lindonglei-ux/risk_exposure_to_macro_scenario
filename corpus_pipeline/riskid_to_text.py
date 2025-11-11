#!/usr/bin/env python3
"""
riskid_to_text.py — Convert a RiskID attributes table (tabular, keywords/phrases + a free-text Summary)
into tokenizer-ready text fields for Sentence Transformer training.

Inputs:
  - CSV/Excel with columns like: ["RiskID","Summary","Basel L1","Basel L2","TX3","ERH","Drivers","Controls","Keywords", ...]
    Only "Summary" is full sentences; others are short keywords/phrases (including acronyms).

Outputs:
  - JSONL with multiple text renderings per row (choose what works best in your pipeline):
      * text_phrases:   canonical bag-of-phrases string (order-stable)
      * text_template:  templated, grammatical sentences using the fields
      * text_mwe:       same as phrases but multi-word expressions protected with underscores (optional)
    and metadata fields: {riskid, fields_used, macro_tags (optional placeholder)}

Tokenization notes:
  - We preserve case and punctuation (CPI, QE, S&P).
  - We generate hyphen/space variants for matching (e.g., "Third-Party Risk" vs "Third Party Risk").
  - We keep acronyms and spell-outs (if provided via --expand_map).

Usage:
  python riskid_to_text.py \\
      --input_file RiskID_Attributes.xlsx \\
      --sheet RiskIDs \\
      --id_col RiskID \\
      --summary_col Summary \\
      --phrase_cols "Basel L1,Basel L2,TX3,ERH,Drivers,Controls,Keywords" \\
      --output_jsonl riskid_corpus.jsonl \\
      --make_csv \\
      --protect_mwe \\
      --tokenizer sentence-transformers/all-MiniLM-L6-v2 \\
      --expand_map expand_map.json

expand_map.json example:
{
  "EDPM": ["Enterprise Data & Process Management"],
  "CISO": ["Chief Information Security Officer"]
}
"""
import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Optional deps
have_pandas = False
have_transformers = False
try:
    import pandas as pd
    have_pandas = True
except Exception:
    pass

tokenizer = None
try:
    from transformers import AutoTokenizer
    have_transformers = True
except Exception:
    pass


def normalize_phrase(p: str) -> str:
    """Trim, collapse whitespace, normalize dashes."""
    p = (p or "").strip()
    p = re.sub(r"\s+", " ", p)
    p = re.sub(r"[‐–—]+", "-", p)
    return p


def hyphen_space_variants(p: str) -> List[str]:
    """Generate simple hyphen/space variants for matching consistency."""
    if "-" in p:
        return [p, p.replace("-", " ")]
    return [p]


def protect_mwe(p: str) -> str:
    """Replace internal spaces with underscores for MWEs, keep hyphens as-is."""
    if not p:
        return p
    if " " not in p:
        return p
    return re.sub(r"\s+", "_", p)


def tokenize_len(text: str, model_name: Optional[str]) -> int:
    """Return token count using HF tokenizer if available, else heuristic."""
    global tokenizer
    if have_transformers and model_name:
        if tokenizer is None or getattr(tokenizer, "name_or_path", None) != model_name:
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_name)
            except Exception:
                tokenizer = None
    if tokenizer:
        return len(tokenizer.encode(text, add_special_tokens=False))
    words = re.findall(r"\w+(\.\w+)?", text)
    return int(len(words) * 1.3)


def render_phrases_row(row_dict: Dict[str, str], phrase_cols: List[str], expand_map: Dict[str, List[str]], protect: bool) -> Tuple[str, List[str]]:
    """Build a canonical bag-of-phrases string and track used fields.

    - Dedup phrases per row, stable sort by (len desc, then alpha) to put stronger phrases earlier.
    - Apply acronym expansions if provided.
    - Include hyphen/space variants to improve matching robustness.
    """
    phrases: List[str] = []
    used: List[str] = []
    seen = set()
    for col in phrase_cols:
        val = row_dict.get(col, "")
        if not isinstance(val, str):
            continue
        parts = [normalize_phrase(x) for x in re.split(r"[;,/|]", val) if x and x.strip()]
        if not parts and val.strip():
            parts = [normalize_phrase(val)]
        for p in parts:
            if not p:
                continue
            exp = expand_map.get(p, [])
            cands = [p] + exp
            out_cands: List[str] = []
            for c in cands:
                out_cands.extend(hyphen_space_variants(c))
            for c in out_cands:
                c = normalize_phrase(c)
                key = c.lower()
                if c and key not in seen:
                    seen.add(key)
                    phrases.append(c)
                    used.append(col)
    phrases_sorted = sorted(phrases, key=lambda x: (-len(x), x.lower()))
    if protect:
        phrases_sorted = [protect_mwe(p) for p in phrases_sorted]
    text = " | ".join(phrases_sorted)
    return text, used


def render_template_row(row_dict: Dict[str, str], id_col: str, summary_col: str, phrase_cols: List[str]) -> str:
    """Render a grammatical sentence template combining summary + fields."""
    rid = str(row_dict.get(id_col, "")).strip()
    summary = str(row_dict.get(summary_col, "") or "").strip()
    fields_str = []
    for col in phrase_cols:
        val = row_dict.get(col, "")
        if isinstance(val, str) and val.strip():
            fields_str.append(f"{col}: {normalize_phrase(val)}")
    fields_render = "; ".join(fields_str)
    if summary and fields_render:
        return f"RiskID {rid}. Summary: {summary} Related attributes — {fields_render}."
    elif fields_render:
        return f"RiskID {rid}. Attributes — {fields_render}."
    elif summary:
        return f"RiskID {rid}. Summary: {summary}"
    return f"RiskID {rid}."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_file", required=True, help="RiskID table (.csv or .xlsx)")
    ap.add_argument("--sheet", default=None, help="Sheet name for Excel")
    ap.add_argument("--id_col", default="RiskID")
    ap.add_argument("--summary_col", default="Summary")
    ap.add_argument("--phrase_cols", required=True, help="Comma-separated list of columns with keywords/phrases")
    ap.add_argument("--output_jsonl", required=True, help="Output JSONL path")
    ap.add_argument("--make_csv", action="store_true", help="Also write a flat CSV with rendered text")
    ap.add_argument("--protect_mwe", action="store_true", help="Underscore multi-word expressions in text_mwe")
    ap.add_argument("--tokenizer", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--expand_map", default=None, help="JSON mapping of acronym->list of expansions")
    args = ap.parse_args()

    phrase_cols = [c.strip() for c in args.phrase_cols.split(",") if c.strip()]

    expand_map: Dict[str, List[str]] = {}
    if args.expand_map and Path(args.expand_map).exists():
        expand_map = json.loads(Path(args.expand_map).read_text(encoding="utf-8"))

    # Load
    if args.input_file.lower().endswith(".xlsx"):
        if not have_pandas:
            raise RuntimeError("pandas is required for Excel input")
        df = pd.read_excel(args.input_file, sheet_name=args.sheet or 0, dtype=str)
    else:
        if not have_pandas:
            # fallback CSV reader
            rows = []
            with open(args.input_file, newline="", encoding="utf-8") as f:
                r = csv.DictReader(f)
                for row in r:
                    rows.append({k: (v if v is not None else "") for k, v in row.items()})
            class _DF:
                def __init__(self, rows): self.rows = rows
                def to_dict(self, orient): return self.rows
                def __iter__(self): return iter(self.rows)
                def __len__(self): return len(self.rows)
            df = _DF(rows)  # type: ignore
        else:
            df = pd.read_csv(args.input_file, dtype=str)

    out_jsonl = Path(args.output_jsonl)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    flat_rows = []
    count = 0
    with out_jsonl.open("w", encoding="utf-8") as f:
        iterator = df.to_dict(orient="records") if have_pandas else df  # type: ignore
        for row in iterator:
            row = {k: ("" if v is None else str(v)) for k, v in row.items()}
            rid = row.get(args.id_col, f"row_{count}")
            text_phrases, used_fields = render_phrases_row(row, phrase_cols, expand_map, protect=args.protect_mwe)
            text_template = render_template_row(row, args.id_col, args.summary_col, phrase_cols)
            text_mwe = text_phrases  # already protected if flag used

            rec = {
                "riskid": rid,
                "text_phrases": text_phrases,
                "text_template": text_template,
                "text_mwe": text_mwe,
                "fields_used": used_fields,
                "summary": row.get(args.summary_col, ""),
                "tokens_phrases": tokenize_len(text_phrases, args.tokenizer),
                "tokens_template": tokenize_len(text_template, args.tokenizer),
                "tokens_mwe": tokenize_len(text_mwe, args.tokenizer)
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            flat_rows.append(rec)
            count += 1

    if args.make_csv:
        out_csv = out_jsonl.with_suffix(".csv")
        fieldnames = ["riskid","text_phrases","text_template","text_mwe","fields_used","summary","tokens_phrases","tokens_template","tokens_mwe"]
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in flat_rows:
                w.writerow(r)

    print(f"Wrote {count} records to {out_jsonl}")
    if args.make_csv:
        print(f"Also wrote {out_csv}")
    total_tokens = sum(r["tokens_template"] for r in flat_rows)
    print(f"Total tokens (template view) ≈ {total_tokens:,}")

if __name__ == "__main__":
    main()
