# Ancient Greek ↔ English NMT (Low‑Resource Recipe)

This repository contains a **full, end‑to‑end pipeline** for training machine translation models between **Ancient Greek (grc)** and **English (en)**. It follows the practical strategy we discussed: transfer learning from a multilingual parent, orthography‑aware normalization, back‑translation, and robust evaluation (BLEU + chrF).

You get:
- `grc_nmt.py` — a single Python CLI that does **preprocess → split → train → generate → back‑translation → evaluate** with strong defaults (mBART‑50). It’s also NLLB‑ready.
- `grc_nmt_opus.py` — the earlier **OPUS/Marian‑default** variant for reference.
- `notebooks/grc_en_demo.ipynb` — a **tiny demo notebook** that runs a toy experiment end‑to‑end.
- `sample_data/` — a handful of toy lines to sanity‑check the pipeline wiring.
- `requirements.txt` — versions that work well together.

> **Heads‑up on data**: Real training requires your own corpora. The `sample_data` is intentionally tiny just to verify the plumbing.

---

## 1) Environment

**Python**: 3.10+ recommended.  
**Hardware**: A single 16–24GB GPU is enough for small runs; multi‑GPU speeds things up (configure with `accelerate`).

```bash
python -m pip install -r requirements.txt
# (optional) python -m accelerate config
```

Key packages:
- `transformers>=4.42`, `datasets>=2.16`, `accelerate>=0.33`
- `sacrebleu`, `ftfy`, `unidecode`, `regex`

---

## 2) Project layout

```
ancient-greek-nmt/
├─ grc_nmt.py                 # mBART/NLLB‑ready pipeline (default mBART)
├─ grc_nmt_opus.py            # reference version with OPUS/Marian defaults
├─ notebooks/
│  └─ grc_en_demo.ipynb       # tiny end‑to‑end demo
├─ sample_data/
│  ├─ grc.txt                 # toy Greek
│  └─ en.txt                  # toy English
├─ requirements.txt
└─ README.md
```

---

## 3) Quickstart (tiny toy run)

```bash
# 1) Preprocess to JSONL
python grc_nmt.py preprocess   --src_file sample_data/grc.txt   --tgt_file sample_data/en.txt   --out_jsonl data/bitext.jsonl   --lang src_grc --keep_diacritics --lowercase

# 2) Split
python grc_nmt.py make_splits   --bitext data/bitext.jsonl   --train data/train.jsonl --dev data/dev.jsonl --test data/test.jsonl

# 3) Train GRC→EN (mBART)
python grc_nmt.py train   --train_jsonl data/train.jsonl --dev_jsonl data/dev.jsonl   --direction grc2en   --model_name facebook/mbart-large-50-many-to-many-mmt   --save_dir runs/grc2en.mbart   --num_epochs 1 --lr 5e-5 --batch 4   --src_lang_tok_code el_GR --tgt_lang_tok_code en_XX --forced_bos_lang en_XX

# 4) Evaluate
python grc_nmt.py evaluate   --model_dir runs/grc2en.mbart   --test_jsonl data/test.jsonl   --src_lang grc --tgt_lang en   --src_lang_tok_code el_GR --tgt_lang_tok_code en_XX
```

> For a proper run, increase epochs to ~20–30, use larger batch (or gradient accumulation), and real data.

---

## 4) Switching models (mBART ↔ NLLB ↔ Marian)

- **mBART‑50 (default)**: `facebook/mbart-large-50-many-to-many-mmt`  
  Language codes: `el_GR` (Greek script), `en_XX` (English). Ancient Greek isn’t a separate code, but script‑level tokenization works fine.
- **NLLB‑200**: e.g., `facebook/nllb-200-distilled-600M` (smaller) or `facebook/nllb-200-1.3B` (larger)  
  Language codes: `ell_Grek` (Greek script), `eng_Latn` (English Latin script).
- **Marian (OPUS)**: See `grc_nmt_opus.py` (simpler, strong baseline).

**Tip:** If you omit `--src_lang_tok_code`/`--tgt_lang_tok_code`, the script **auto‑guesses** sensible codes based on model name and direction.

---

## 5) Back‑translation (improves EN→GRC)

1. Train a **GRC→EN** model.
2. Use it to translate **English monolingual** into **synthetic Greek**:
   ```bash
   python grc_nmt.py generate      --model_dir runs/grc2en.mbart      --in_txt data/mono.en.txt --out_txt data/mono.en.synthetic.grc      --src_lang en --tgt_lang grc      --src_lang_tok_code en_XX --tgt_lang_tok_code el_GR --forced_bos_lang el_GR
   ```
3. Build synthetic bitext and **mix** with real parallel:
   ```bash
   python grc_nmt.py build_bt      --source_txt data/mono.en.txt      --synthetic_txt data/mono.en.synthetic.grc      --out_jsonl data/bt.en_grc.jsonl --direction en2grc

   python grc_nmt.py train      --train_jsonl data/train.jsonl data/bt.en_grc.jsonl      --dev_jsonl data/dev.jsonl      --direction en2grc      --model_name facebook/mbart-large-50-many-to-many-mmt      --save_dir runs/en2grc.mbart      --num_epochs 24 --lr 5e-5 --batch 64      --src_lang_tok_code en_XX --tgt_lang_tok_code el_GR --forced_bos_lang el_GR
   ```

---

## 6) Evaluation

We report **SacreBLEU** and **chrF**:
```bash
python grc_nmt.py evaluate   --model_dir runs/grc2en.mbart   --test_jsonl data/test.jsonl   --src_lang grc --tgt_lang en   --src_lang_tok_code el_GR --tgt_lang_tok_code en_XX
```

**Recommendations**
- Keep **author/work‑level splits** to avoid leakage.
- Report scores **by register/dialect** (Homeric / Attic / Koine).
- Include **round‑trip chrF** if you want an extra robustness signal.

---

## 7) Data normalization tips

- Normalize: NFC, unify quotes/punctuation, **ς→σ** in mid‑word, optional diacritics stripping (keep a second “raw” copy for output).
- Consider domain tags like `<BIBLE>`, `<HIST>`, `<PHIL>` as **prefix tokens** to guide style.

> ⚖️ **Licensing**: Some aligned corpora are **non‑commercial**. Always verify licenses for your use case.

---

## 8) Troubleshooting

- **Repeated phrases / hallucinations**: increase `no_repeat_ngram_size`, use length penalty (`--length_penalty 1.0–1.5`), try sampling with `--do_sample` for EN→GRC name rendering.
- **Wrong style**: add domain tags; fine‑tune adapters by sub‑domain; mix broader bitext.
- **Tokenizer errors**: set `--src_lang_tok_code`/`--tgt_lang_tok_code` explicitly; confirm codes for mBART vs NLLB.
- **OOM**: lower `--batch`, raise `--grad_accum`, reduce `--max_src_len`, or use a smaller parent (distilled NLLB).

---

## 9) Notebook

See `notebooks/grc_en_demo.ipynb` for a self‑contained toy run. It uses relative paths (`../grc_nmt.py`).

---

## 10) Attribution

This codebase adapts standard low‑resource NMT practices (transfer learning, back‑translation, chrF/BLEU) to Ancient Greek. It was inspired by https://polytranslator.com/paper.pdf. 
