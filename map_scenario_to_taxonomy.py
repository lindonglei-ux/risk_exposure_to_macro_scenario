"""Map scenario sentences to taxonomy labels with lightweight fallbacks.

Running the module as a script accepts a ``--similarity-mode`` flag that selects
between a ``sentence-transformers`` backend, a TF–IDF lexical model, and the
lightweight bag-of-words routine.  The default ``auto`` mode uses the
best-available option depending on the locally installed dependencies.

The script retains the original goal of functioning in restricted execution
environments.  Heavy optional dependencies such as ``pandas``, ``numpy``,
``sentence_transformers``, and ``scikit-learn`` are detected dynamically and the
workflow downgrades gracefully when they are missing.

When the richer stack *is* available the script provides a more detailed macro
theme bridge analysis by reusing the logic that previously lived in
``scenario_to_taxonomy_mapping_lw.py``.  This additional stage scores scenario
sentences against macro themes using TF–IDF cosine similarity, expands the
themes into linked taxonomy entries, and records the intermediate and direct
similarity scores in dedicated Excel sheets.

In particular, the script can ingest tabular inputs (``.csv`` or ``.xlsx``)
without ``pandas`` by relying on the standard ``csv`` module and a very small
OpenXML parser for Excel workbooks.  The richer dependency stack is still used
automatically whenever it is available locally.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence
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
    from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
    from sklearn.metrics.pairwise import cosine_similarity  # type: ignore

    _HAS_SKLEARN = True
except ImportError:  # pragma: no cover - triggered in minimal environments
    TfidfVectorizer = None  # type: ignore
    cosine_similarity = None  # type: ignore
    _HAS_SKLEARN = False

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

DEFAULT_SCENARIO_PATH = DATA_DIR / "3Q 2025 BAC Global Debt Threat Scenario.txt"
DEFAULT_TAXONOMY_CANDIDATES = [
    DATA_DIR / "BAC_GDT_2025_Risk_Taxonomy.xlsx",
    DATA_DIR / "Primary Risk Taxonomy.txt",
]
DEFAULT_OUTPUT_PATH = DATA_DIR / "GDT_Taxonomy_Similarity.xlsx"
DEFAULT_BRIDGE_PATH = DATA_DIR / "Macro_to_OpRisk_Bridge.xlsx"


def _discover_default_taxonomy_path() -> Path:
    for candidate in DEFAULT_TAXONOMY_CANDIDATES:
        if candidate.exists():
            return candidate
    return DEFAULT_TAXONOMY_CANDIDATES[0]

# ======== ARGUMENTS ========


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Map scenario sentences to taxonomy labels using either "
            "sentence-transformers embeddings, a TF-IDF lexical model, "
            "or a lightweight bag-of-words similarity routine."
        )
    )
    parser.add_argument(
        "--similarity-mode",
        choices=("auto", "sentence-transformer", "tfidf", "bow"),
        default="auto",
        help=(
            "Select the similarity backend. 'auto' prefers sentence-transformers "
            "when installed, then TF-IDF if scikit-learn is available, "
            "otherwise falls back to the bag-of-words model."
        ),
    )
    parser.add_argument(
        "--scenario-path",
        type=Path,
        default=DEFAULT_SCENARIO_PATH,
        help="Path to the scenario narrative text file.",
    )
    parser.add_argument(
        "--taxonomy-path",
        type=Path,
        default=_discover_default_taxonomy_path(),
        help="Path to the taxonomy source file (TXT, CSV, or XLSX).",
    )
    parser.add_argument(
        "--bridge-path",
        type=Path,
        default=DEFAULT_BRIDGE_PATH if DEFAULT_BRIDGE_PATH.exists() else None,
        help=(
            "Optional path to the Macro-to-OpRisk bridge file. Provide when "
            "macro theme analysis should be executed."
        ),
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Destination Excel path for the similarity results.",
    )
    return parser.parse_args()


args = _parse_args()

scenario_path = args.scenario_path
taxonomy_path = args.taxonomy_path
output_path = args.output_path
bridge_path: Optional[Path]
if args.bridge_path and args.bridge_path.exists():
    bridge_path = args.bridge_path
else:
    bridge_path = None
    if args.bridge_path:
        print(f"⚠️ Bridge file not found at {args.bridge_path}; skipping macro analysis.")

if not scenario_path.exists():  # pragma: no cover - defensive guard
    raise FileNotFoundError(f"Scenario narrative not found at {scenario_path}.")

if not taxonomy_path.exists():  # pragma: no cover - defensive guard
    raise FileNotFoundError(f"Taxonomy file not found at {taxonomy_path}.")

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


def _tfidf_cosine_similarities(
    source_sentences: Sequence[str],
    target_labels: Sequence[str],
) -> List[List[float]]:
    """Compute cosine similarity scores using TF-IDF vectors."""

    if not _HAS_SKLEARN:  # pragma: no cover - guarded by callers
        raise RuntimeError("scikit-learn is required for TF-IDF similarity")

    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    target_matrix = vectorizer.fit_transform([label.lower() for label in target_labels])
    source_matrix = vectorizer.transform([sentence.lower() for sentence in source_sentences])
    scores_matrix = cosine_similarity(source_matrix, target_matrix)
    if _HAS_NUMPY:
        return scores_matrix.tolist()
    return [list(map(float, row)) for row in scores_matrix]


def _run_macro_bridge_pipeline(
    sentences: Sequence[str],
    taxonomy_labels: Sequence[str],
    bridge_path: Path,
) -> Dict[str, "pd.DataFrame"]:
    """Execute the macro-theme bridge pipeline when dependencies are available."""

    if not _HAS_PANDAS:
        print(
            "⚠️ Macro theme bridge requested but pandas is unavailable; "
            "skipping macro analysis."
        )
        return {}

    if not _HAS_SKLEARN:
        print(
            "⚠️ Macro theme bridge requested but scikit-learn is unavailable; "
            "skipping macro analysis."
        )
        return {}

    suffix = bridge_path.suffix.lower()
    if suffix in {".xlsx", ".xlsm", ".xltx", ".xltm"}:
        loader = pd.read_excel
    elif suffix == ".csv":
        loader = pd.read_csv
    else:
        print(
            f"⚠️ Unsupported macro bridge file type '{bridge_path.suffix}'. "
            "Skipping macro analysis."
        )
        return {}

    try:
        bridge_df = loader(bridge_path)
    except Exception as exc:  # pragma: no cover - defensive guard
        print(f"⚠️ Unable to load macro bridge file: {exc}; skipping macro analysis.")
        return {}

    required_cols = {
        "Macro_Theme_ID",
        "Macro_Theme_Name",
        "Macro_Theme_Description",
        "Trigger_Keywords_Examples",
        "Linked_Taxonomies",
    }
    missing = [col for col in required_cols if col not in bridge_df.columns]
    if missing:
        print(
            "⚠️ Macro bridge file missing required columns "
            f"{', '.join(missing)}; skipping macro analysis."
        )
        return {}

    bridge_df = bridge_df.copy()
    bridge_df["Macro_Text"] = (
        bridge_df["Macro_Theme_Name"].astype(str)
        + " "
        + bridge_df["Macro_Theme_Description"].astype(str)
        + " "
        + bridge_df["Trigger_Keywords_Examples"].fillna("").astype(str)
    )
    macro_themes = bridge_df["Macro_Text"].tolist()

    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    macro_vecs = vectorizer.fit_transform([theme.lower() for theme in macro_themes])
    scenario_vecs = vectorizer.transform([sentence.lower() for sentence in sentences])
    macro_scores = cosine_similarity(scenario_vecs, macro_vecs)

    macro_records: List[Dict[str, object]] = []
    for sent_idx, sent in enumerate(sentences):
        scores = macro_scores[sent_idx]
        ranked_indices = sorted(
            range(len(scores)), key=lambda idx: scores[idx], reverse=True
        )[:3]
        for theme_idx in ranked_indices:
            score = float(scores[theme_idx])
            if score <= 0.05:
                continue
            row = bridge_df.iloc[theme_idx]
            macro_records.append(
                {
                    "Scenario_Sentence": sent,
                    "Macro_Theme_ID": row["Macro_Theme_ID"],
                    "Macro_Theme_Name": row["Macro_Theme_Name"],
                    "Similarity_Macro": round(score, 3),
                }
            )

    if macro_records:
        macro_df = pd.DataFrame(macro_records)
    else:
        macro_df = pd.DataFrame(
            columns=[
                "Scenario_Sentence",
                "Macro_Theme_ID",
                "Macro_Theme_Name",
                "Similarity_Macro",
            ]
        )

    if macro_df.empty:
        macro_expanded = pd.DataFrame(
            columns=[
                "Scenario_Sentence",
                "Macro_Theme_ID",
                "Macro_Theme_Name",
                "Similarity_Macro",
                "Risk_Taxonomy",
            ]
        )
    else:
        macro_expanded = (
            macro_df.merge(
                bridge_df[["Macro_Theme_ID", "Linked_Taxonomies"]],
                on="Macro_Theme_ID",
                how="left",
            )
            .assign(
                Linked_Taxonomies=lambda d: d["Linked_Taxonomies"]
                .fillna("")
                .astype(str)
                .str.split(",")
            )
            .explode("Linked_Taxonomies")
            .rename(columns={"Linked_Taxonomies": "Risk_Taxonomy"})
        )
        macro_expanded["Risk_Taxonomy"] = (
            macro_expanded["Risk_Taxonomy"].fillna("").astype(str).str.strip()
        )
        macro_expanded = macro_expanded[macro_expanded["Risk_Taxonomy"] != ""]

    tfidf_scores = _tfidf_cosine_similarities(sentences, taxonomy_labels)
    direct_records: List[Dict[str, object]] = []
    for sent_idx, sent in enumerate(sentences):
        scores = tfidf_scores[sent_idx]
        ranked_indices = sorted(
            range(len(scores)), key=lambda idx: scores[idx], reverse=True
        )[:5]
        for tax_idx in ranked_indices:
            score = float(scores[tax_idx])
            if score <= 0.05:
                continue
            direct_records.append(
                {
                    "Scenario_Sentence": sent,
                    "Risk_Taxonomy": taxonomy_labels[tax_idx],
                    "Similarity_Taxonomy": round(score, 3),
                }
            )

    if direct_records:
        direct_df = pd.DataFrame(direct_records)
    else:
        direct_df = pd.DataFrame(
            columns=["Scenario_Sentence", "Risk_Taxonomy", "Similarity_Taxonomy"]
        )

    if macro_expanded.empty:
        agg = pd.DataFrame(columns=["Risk_Taxonomy", "Count", "Mean", "Max"])
    else:
        agg = (
            macro_expanded.groupby("Risk_Taxonomy")["Similarity_Macro"]
            .agg(Count="count", Mean="mean", Max="max")
            .reset_index()
            .sort_values("Mean", ascending=False)
        )
        agg["Mean"] = agg["Mean"].round(3)
        agg["Max"] = agg["Max"].round(3)

    return {
        "Scenario→MacroThemes": macro_df,
        "MacroThemes→Taxonomies": macro_expanded,
        "Direct_Scenario→Taxonomy": direct_df,
        "Macro_Aggregated_Summary": agg,
    }


mode = args.similarity_mode
if mode == "sentence-transformer":
    if _HAS_ST:
        backend = "sentence-transformer"
    else:
        if _HAS_SKLEARN:
            backend = "tfidf"
        else:
            backend = "bow"
        print(
            "sentence-transformers requested but unavailable; "
            f"falling back to {backend} similarity."
        )
elif mode == "tfidf":
    if _HAS_SKLEARN:
        backend = "tfidf"
    else:
        backend = "bow"
        print(
            "TF-IDF similarity requested but scikit-learn is unavailable; "
            "falling back to bag-of-words."
        )
elif mode == "bow":
    backend = "bow"
else:  # auto
    if _HAS_ST:
        backend = "sentence-transformer"
    elif _HAS_SKLEARN:
        backend = "tfidf"
    else:
        backend = "bow"
    if backend != "sentence-transformer":
        print(
            "sentence-transformers not available; "
            f"using {backend} cosine similarity instead."
        )


if backend == "sentence-transformer":
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
elif backend == "tfidf":
    cosine_scores = _tfidf_cosine_similarities(sentences, taxonomy_labels)
else:
    if mode == "bow":
        print("Bag-of-words similarity requested; skipping sentence-transformers.")
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

macro_sheets: Dict[str, "pd.DataFrame"] = {}
if bridge_path:
    macro_sheets = _run_macro_bridge_pipeline(sentences, taxonomy_labels, bridge_path)

if _HAS_PANDAS:
    df = pd.DataFrame(filtered_records)
    summary = pd.DataFrame(summary_records)

    # ======== WRITE RESULTS ========
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Sentence-Level Mapping", index=False)
        summary.to_excel(writer, sheet_name="Aggregated Summary", index=False)
        for sheet_name, sheet_df in macro_sheets.items():
            sheet_df.to_excel(writer, sheet_name=sheet_name, index=False)
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
