"""
Scenario → Macro Theme → OpRisk Taxonomy Mapping
------------------------------------------------
Lightweight, fully auditable version using TF-IDF cosine similarity.
Adds an intermediate economist reasoning layer ("Macro Theme Bridge").
"""

import pandas as pd, re, numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path

# ======== INPUT FILES ========
scenario_path = Path("data/3Q 2025 BAC Global Debt Threat Scenario.txt")
taxonomy_path = Path("data/Primary Risk Taxonomy.txt")
bridge_path   = Path("data/Macro_to_OpRisk_Bridge.xlsx")   # or .csv
output_path   = Path("data/GDT_Taxonomy_Similarity_withMacro.xlsx")

# ======== LOAD & CLEAN DATA ========
scenario_text = scenario_path.read_text(encoding="utf-8")
taxonomy_text = taxonomy_path.read_text(encoding="utf-8")

taxonomy_labels = [
    line.strip() for line in taxonomy_text.splitlines()
    if line.strip() and not any(h in line for h in ["Distinct", "Risk", "Standardized"])
]

sentences = re.split(r'(?<=[.!?])\s+', scenario_text)
sentences = [s.strip().lower() for s in sentences if len(s.split()) > 4]

# Load Macro-Theme Bridge
bridge_df = pd.read_excel(bridge_path)
bridge_df["Macro_Text"] = (
    bridge_df["Macro_Theme_Name"].astype(str) + " " +
    bridge_df["Macro_Theme_Description"].astype(str) + " " +
    bridge_df["Trigger_Keywords_Examples"].fillna("")
)
macro_themes = bridge_df["Macro_Text"].tolist()

# ======== STAGE 1 — Scenario → Macro Themes ========
vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1,2))
macro_vecs = vectorizer.fit_transform(macro_themes)
scenario_vecs = vectorizer.transform(sentences)
macro_scores = cosine_similarity(scenario_vecs, macro_vecs)

macro_records = []
for i, sent in enumerate(sentences):
    scores = macro_scores[i]
    top_idx = np.argsort(scores)[::-1][:3]
    for j in top_idx:
        if scores[j] > 0.05:
            macro_records.append({
                "Scenario_Sentence": sent,
                "Macro_Theme_ID": bridge_df.iloc[j]["Macro_Theme_ID"],
                "Macro_Theme_Name": bridge_df.iloc[j]["Macro_Theme_Name"],
                "Similarity_Macro": round(float(scores[j]), 3)
            })
macro_df = pd.DataFrame(macro_records)

# ======== STAGE 2 — Macro Themes → Taxonomies ========
macro_expanded = (
    macro_df
    .merge(bridge_df[["Macro_Theme_ID","Linked_Taxonomies"]], on="Macro_Theme_ID", how="left")
    .assign(Linked_Taxonomies=lambda d: d["Linked_Taxonomies"].fillna("").str.split(","))
    .explode("Linked_Taxonomies")
    .rename(columns={"Linked_Taxonomies":"Risk_Taxonomy"})
)

# ======== STAGE 3 — Optional direct Scenario→Taxonomy Similarity ========
vectorizer2 = TfidfVectorizer(stop_words="english", ngram_range=(1,2))
taxonomy_vecs = vectorizer2.fit_transform(taxonomy_labels)
scenario_vecs2 = vectorizer2.transform(sentences)
direct_scores = cosine_similarity(scenario_vecs2, taxonomy_vecs)

records_direct = []
for i, sent in enumerate(sentences):
    scores = direct_scores[i]
    top_idx = np.argsort(scores)[::-1][:5]
    for j in top_idx:
        if scores[j] > 0.05:
            records_direct.append({
                "Scenario_Sentence": sent,
                "Risk_Taxonomy": taxonomy_labels[j],
                "Similarity_Taxonomy": round(float(scores[j]), 3)
            })
direct_df = pd.DataFrame(records_direct)

# ======== AGGREGATED OUTPUT ========
agg = (
    macro_expanded.groupby("Risk_Taxonomy")["Similarity_Macro"]
    .agg(["count","mean","max"])
    .reset_index()
    .sort_values("mean", ascending=False)
)

# ======== WRITE TO EXCEL ========
with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
    macro_df.to_excel(writer, sheet_name="Scenario→MacroThemes", index=False)
    macro_expanded.to_excel(writer, sheet_name="MacroThemes→Taxonomies", index=False)
    direct_df.to_excel(writer, sheet_name="Direct_Scenario→Taxonomy", index=False)
    agg.to_excel(writer, sheet_name="Aggregated_Summary", index=False)

print(f"✅ Mapping pipeline complete. Results saved to {output_path}")