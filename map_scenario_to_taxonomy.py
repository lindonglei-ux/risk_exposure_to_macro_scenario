"""Map scenario sentences to taxonomy labels with lightweight fallbacks.

This script originally depended on heavy third-party libraries such as
``pandas`` and ``sentence_transformers``. Those packages are not available in
the execution environment used for automated testing, so the import failures
prevented the script from running at all.  To keep the workflow functional we
now fall back to standard-library implementations whenever the optional
dependencies are missing.  When the libraries are available locally the
behaviour remains unchanged.
"""

from __future__ import annotations

import csv
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, List, Sequence

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
taxonomy_path = DATA_DIR / "Primary Risk Taxonomy.txt"
output_path = DATA_DIR / "GDT_Taxonomy_Similarity.xlsx"

# ======== LOAD FILES ========
scenario_text = scenario_path.read_text(encoding="utf-8")
taxonomy_text = taxonomy_path.read_text(encoding="utf-8")

# Extract taxonomy labels (remove header lines)
taxonomy_labels = [
    line.strip() for line in taxonomy_text.splitlines()
    if line.strip() and not any(h in line for h in ["Distinct", "Risk", "Standardized"])
]

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
