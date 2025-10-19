"""Map scenario sentences to taxonomy labels with lightweight fallbacks.

This script originally depended on heavy third-party libraries such as
``pandas`` and ``sentence_transformers``. Those packages are not available in
the execution environment used for automated testing, so the import failures
prevented the script from running at all.  To keep the workflow functional we
now fall back to standard-library implementations whenever the optional
dependencies are missing.  When the libraries are available locally the
behaviour remains unchanged.

In particular, the script can ingest tabular inputs (``.csv`` or ``.xlsx``)
without ``pandas`` by relying on the standard ``csv`` module and a very small
OpenXML parser for Excel workbooks.  The richer dependency stack is still used
automatically whenever it is available locally.
"""

from __future__ import annotations

import csv
import math
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, List, Sequence
from xml.etree import ElementTree as ET

try:  # Optional dependency
    import pandas as pd  # type: ignore
    _HAS_PANDAS = True
except ImportError:  # pragma: no cover - triggered in minimal environments
    pd = None  # type: ignore
    _HAS_PANDAS = False

try:  # Optional dependency
    import numpy as np  # type: ignore
    _HAS_NUMPY = True
except ImportError:  # pragma: no cover - triggered in minimal environments
    np = None  # type: ignore
    _HAS_NUMPY = False

try:  # Optional dependency
    from sentence_transformers import SentenceTransformer, util  # type: ignore
    _HAS_ST = True
except ImportError:  # pragma: no cover - triggered in minimal environments
    SentenceTransformer = None  # type: ignore
    util = None  # type: ignore
    _HAS_ST = False

# ======== INPUT FILES (update path if needed) ========
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

scenario_path = DATA_DIR / "3Q 2025 BAC Global Debt Threat Scenario.txt"
taxonomy_candidates = [
    DATA_DIR / "BAC_GDT_2025_Risk_Taxonomy.xlsx",
    DATA_DIR / "Primary Risk Taxonomy.txt",
]
output_path = DATA_DIR / "GDT_Taxonomy_Similarity.xlsx"

for candidate in taxonomy_candidates:
    if candidate.exists():
        taxonomy_path = candidate
        break
else:  # pragma: no cover - defensive guard
    raise FileNotFoundError("No taxonomy source file found in the data directory.")

# ======== LOAD FILES ========
scenario_text = scenario_path.read_text(encoding="utf-8")


def _excel_column_index(cell_ref: str) -> int:
    letters = "".join(ch for ch in cell_ref if ch.isalpha())
    index = 0
    for ch in letters:
        index = index * 26 + (ord(ch.upper()) - 64)
    return max(index - 1, 0)


def _read_xlsx_without_pandas(path: Path) -> List[List[str]]:
    """Parse the first worksheet of ``path`` into a list-of-lists structure."""

    with zipfile.ZipFile(path) as zf:
        main_ns = {"main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
        rel_ns = {
            "rel": "http://schemas.openxmlformats.org/package/2006/relationships"
        }

        shared_strings: List[str] = []
        if "xl/sharedStrings.xml" in zf.namelist():
            shared_root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
            for si in shared_root.findall("main:si", main_ns):
                text = "".join(
                    t.text or "" for t in si.findall(".//main:t", main_ns)
                )
                shared_strings.append(text)

        rel_root = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
        rel_map = {
            rel.attrib["Id"]: rel.attrib["Target"]
            for rel in rel_root.findall("rel:Relationship", rel_ns)
        }

        workbook_root = ET.fromstring(zf.read("xl/workbook.xml"))
        sheet_paths: List[str] = []
        for sheet in workbook_root.findall("main:sheets/main:sheet", main_ns):
            rel_id = sheet.attrib.get(
                "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"
            )
            if not rel_id:
                continue
            target = rel_map.get(rel_id)
            if not target:
                continue
            sheet_paths.append("xl/" + target if not target.startswith("/") else target.lstrip("/"))
        if not sheet_paths:
            raise ValueError(f"Unable to locate worksheets in {path}.")

        sheet_root = ET.fromstring(zf.read(sheet_paths[0]))

        rows: List[List[str]] = []
        for row_el in sheet_root.findall("main:sheetData/main:row", main_ns):
            row_values: dict[int, str] = {}
            max_idx = -1
            for cell in row_el.findall("main:c", main_ns):
                ref = cell.attrib.get("r", "A")
                idx = _excel_column_index(ref)
                max_idx = max(max_idx, idx)
                value_el = cell.find("main:v", main_ns)
                if value_el is None:
                    continue
                raw = value_el.text or ""
                if cell.attrib.get("t") == "s":
                    try:
                        raw = shared_strings[int(raw)]
                    except (IndexError, ValueError):
                        raw = ""
                row_values[idx] = raw
            if max_idx >= 0:
                row = [row_values.get(i, "") for i in range(max_idx + 1)]
            else:
                row = []
            rows.append(row)
    return rows


def _load_tabular_column(path: Path, column_index: int = 0) -> List[str]:
    """Return the non-empty values from ``column_index`` in ``path``."""

    suffix = path.suffix.lower()
    if suffix == ".txt":
        return [
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not any(
                header in line for header in ["Distinct", "Risk", "Standardized"]
            )
        ]

    if suffix == ".csv":
        if _HAS_PANDAS:
            series = pd.read_csv(path, usecols=[column_index]).iloc[:, 0]
            return [
                str(value).strip()
                for value in series.dropna()
                if str(value).strip()
            ]
        with path.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.reader(fh)
            rows = list(reader)
        if rows:
            rows = rows[1:]
        values: List[str] = []
        for row in rows:
            if column_index < len(row):
                value = row[column_index].strip()
                if value:
                    values.append(value)
        return values

    if suffix in {".xlsx", ".xlsm", ".xltx", ".xltm"}:
        if _HAS_PANDAS:
            frame = pd.read_excel(path, usecols=[column_index])
            series = frame.iloc[:, 0]
            return [
                str(value).strip()
                for value in series.dropna()
                if str(value).strip()
            ]
        rows = _read_xlsx_without_pandas(path)
        if rows:
            rows = rows[1:]
        values = []
        for row in rows:
            if column_index < len(row):
                value = str(row[column_index]).strip()
                if value:
                    values.append(value)
        return values

    raise ValueError(f"Unsupported file type for {path}")


taxonomy_labels = _load_tabular_column(taxonomy_path, column_index=0)

# Split scenario into sentences (long lines become separate analysis units)
sentences = re.split(r'(?<=[.!?])\s+', scenario_text)
sentences = [s.strip() for s in sentences if len(s.split()) > 4]

print(f"{len(sentences)} scenario sentences, {len(taxonomy_labels)} taxonomy labels loaded.")


_TOKEN_PATTERN = re.compile(r"\b\w+\b")


def _tokenize(text: str) -> List[str]:
    """Lowercase tokenization used by the fallback similarity routine."""

    return _TOKEN_PATTERN.findall(text.lower())


def _vectorize(text: str) -> tuple[Counter[str], float]:
    """Return bag-of-words counts and their Euclidean norm."""

    counts: Counter[str] = Counter(_tokenize(text))
    norm = math.sqrt(sum(value * value for value in counts.values()))
    return counts, norm


def _bow_cosine_similarities(
    source_sentences: Sequence[str],
    target_labels: Sequence[str],
) -> List[List[float]]:
    """Compute cosine similarity on bag-of-words vectors as a fallback."""

    sent_vectors = [_vectorize(sentence) for sentence in source_sentences]
    label_vectors = [_vectorize(label) for label in target_labels]

    scores: List[List[float]] = []
    for sent_counts, sent_norm in sent_vectors:
        row: List[float] = []
        for label_counts, label_norm in label_vectors:
            if not sent_norm or not label_norm:
                row.append(0.0)
                continue
            overlap = sum(
                sent_counts[token] * label_counts[token]
                for token in sent_counts.keys() & label_counts.keys()
            )
            row.append(overlap / (sent_norm * label_norm))
        scores.append(row)
    return scores


if _HAS_ST:
    # ======== EMBEDDING MODEL ========
    model = SentenceTransformer("all-mpnet-base-v2")
    scenario_emb = model.encode(sentences, convert_to_tensor=True)
    taxonomy_emb = model.encode(taxonomy_labels, convert_to_tensor=True)

    # ======== COSINE SIMILARITY ========
    cosine_scores_matrix = util.cos_sim(scenario_emb, taxonomy_emb)
    if _HAS_NUMPY:
        cosine_scores = cosine_scores_matrix.cpu().numpy()
    else:
        cosine_scores = cosine_scores_matrix.cpu().tolist()
else:
    print(
        "sentence-transformers not available; using bag-of-words cosine"
        " similarity scores instead."
    )
    cosine_scores = _bow_cosine_similarities(sentences, taxonomy_labels)

# ======== BUILD OUTPUT ========
records = []
for i, sent in enumerate(sentences):
    scores = cosine_scores[i]
    indexed_scores = list(enumerate(scores))
    indexed_scores.sort(key=lambda item: item[1], reverse=True)
    top_entries = indexed_scores[:10]
    for j, score in top_entries:
        records.append({
            "Scenario_Sentence": sent,
            "Risk_Taxonomy": taxonomy_labels[j],
            "Similarity": round(float(score), 3),
        })

filtered_records = [r for r in records if r["Similarity"] > 0.55]

summary_map: defaultdict[str, List[float]] = defaultdict(list)
for rec in filtered_records:
    summary_map[rec["Risk_Taxonomy"]].append(rec["Similarity"])

summary_records = [
    {
        "Risk_Taxonomy": taxonomy,
        "Count": len(scores),
        "Mean": round(sum(scores) / len(scores), 3) if scores else math.nan,
        "Max": round(max(scores), 3) if scores else math.nan,
    }
    for taxonomy, scores in summary_map.items()
]
summary_records.sort(key=lambda item: item["Mean"], reverse=True)

if _HAS_PANDAS:
    df = pd.DataFrame(filtered_records)
    summary = pd.DataFrame(summary_records)

    # ======== WRITE RESULTS ========
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Sentence-Level Mapping", index=False)
        summary.to_excel(writer, sheet_name="Aggregated Summary", index=False)
    print(f"✅ Completed. Results saved to {output_path}")
else:
    # Fallback: write CSV reports when pandas/openpyxl are unavailable.
    csv_path = output_path.with_suffix(".csv")
    summary_path = output_path.with_name(f"{output_path.stem}_summary.csv")

    def _write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[dict]):
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    _write_csv(
        csv_path,
        ["Scenario_Sentence", "Risk_Taxonomy", "Similarity"],
        filtered_records,
    )
    _write_csv(summary_path, ["Risk_Taxonomy", "Count", "Mean", "Max"], summary_records)
    print(
        "✅ Completed. Results saved to",
        f"{csv_path} and {summary_path} (CSV fallback; install pandas for XLSX)",
    )
