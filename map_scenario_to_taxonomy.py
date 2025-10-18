import pandas as pd
import numpy as np
import re
from sentence_transformers import SentenceTransformer, util
from pathlib import Path

# ======== INPUT FILES (update path if needed) ========
scenario_path = Path("/data/3Q 2025 BAC Global Debt Threat Scenario.txt")
taxonomy_path = Path("/data/Primary Risk Taxonomy.txt")
output_path   = Path("/data/GDT_Taxonomy_Similarity.xlsx")

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

# ======== EMBEDDING MODEL ========
model = SentenceTransformer("all-mpnet-base-v2")
scenario_emb = model.encode(sentences, convert_to_tensor=True)
taxonomy_emb = model.encode(taxonomy_labels, convert_to_tensor=True)

# ======== COSINE SIMILARITY ========
cosine_scores = util.cos_sim(scenario_emb, taxonomy_emb).cpu().numpy()

# ======== BUILD OUTPUT ========
records = []
for i, sent in enumerate(sentences):
    scores = cosine_scores[i]
    top_idx = np.argsort(scores)[::-1][:10]
    for j in top_idx:
        records.append({
            "Scenario_Sentence": sent,
            "Risk_Taxonomy": taxonomy_labels[j],
            "Similarity": round(float(scores[j]), 3)
        })

df = pd.DataFrame(records).query("Similarity > 0.55")

# Optional aggregation: average score per taxonomy (for summary view)
summary = (
    df.groupby("Risk_Taxonomy")["Similarity"]
      .agg(["count","mean","max"])
      .reset_index()
      .sort_values("mean", ascending=False)
)

# ======== WRITE RESULTS ========
with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
    df.to_excel(writer, sheet_name="Sentence-Level Mapping", index=False)
    summary.to_excel(writer, sheet_name="Aggregated Summary", index=False)

print(f"✅ Completed. Results saved to {output_path}")
