Risk Exposure to Macro Scenario Pipeline

This repository provides a prototype implementation of a semantic exposure
pipeline for Bank of America’s Global Control and Operations Risk (GCORE)
project. Each quarter, GCORE publishes a macroeconomic scenario narrative
— for example, BAC Global Debt Threat — and needs to assess how that
scenario impacts the bank’s internal operational risks. The pipeline
implemented in risk_exposure_pipeline.py automates three key tasks:

Text extraction: Convert PDFs or images containing the macro
narrative into machine‑readable text. The module uses
pdfplumber for PDFs and pytesseract for images, so you can feed in
scanned documents or screenshots.

Data loading: Read the risk inventory from an Excel workbook.
Each row should include a narrative description of the risk along with
any taxonomies, drivers, triggers or emerging risks. These extra
columns are preserved and passed through unmodified.

Semantic matching: Compare the macro narrative to each RiskID
narrative using state‑of‑the‑art sentence embeddings (sentence‑transformers) and
rank the risks by cosine similarity. The result is a dataframe with
an exposure_score and a categorical exposure_rank (very weak → very strong).
