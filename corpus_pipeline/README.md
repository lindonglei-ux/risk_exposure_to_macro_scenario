# corpus_pipeline
Modular PDF → text pipeline for extracting and segmenting narrative content.

## Examples

Process a single PDF:

```bash
python -m corpus_pipeline.cli --input_pdf "./Operational-Risk-Stress-Testing.pdf" --out_dir "./out_demo" --segment_by sentence --drop_toc --drop_tables --alpha_ratio 0.2 --drop_text_tables --columns 2
```

Process every PDF under a directory:

```bash
python -m corpus_pipeline.cli --input_pdf "./pdf_inputs" --out_dir "./out_demo" --segment_by sentence --drop_toc --drop_tables --alpha_ratio 0.2 --drop_text_tables --columns 2
```
