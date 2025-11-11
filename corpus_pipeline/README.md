# corpus_pipeline
Modular PDF → text pipeline with exhibit capture.

## Example
python -m corpus_pipeline.cli --input_pdf "./Operational-Risk-Stress-Testing.pdf" --out_dir "./out_demo" --segment_by sentence --drop_toc --drop_tables --alpha_ratio 0.2 --drop_text_tables --columns 2 --extract_exhibits --ocr_exhibits --exhibit_min_height 140 --ocr_psm 6 --normalize_exhibits --macro_lexicon macro_lex.json --micro_lexicon micro_lex.json
