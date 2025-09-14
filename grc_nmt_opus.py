#!/usr/bin/env python3
"""
Ancient Greek ⟷ English NMT — end‑to‑end pipeline

One file to:
  • preprocess & normalize
  • split datasets (train/dev/test with author/work‑level splits if you provide metadata)
  • fine‑tune a seq2seq model (Marian / mBART / NLLB via Hugging Face Transformers)
  • generate (translation or back‑translation)
  • evaluate (SacreBLEU + chrF)

Usage (examples):

# 0) Install deps (Python ≥3.10 recommended)
python -m pip install --upgrade "transformers>=4.42" "datasets>=2.16" "accelerate>=0.33" sacrebleu ftfy unidecode regex

# 1) Normalize raw parallel files into JSONL (columns: src,tgt)
python grc_nmt.py preprocess \
  --src_file data/raw/grc.txt \
  --tgt_file data/raw/en.txt \
  --out_jsonl data/bitext.jsonl \
  --lang src_grc --keep_diacritics --lowercase

# 2) Make splits (80/10/10 by work if a metadata TSV is provided; else random by line)
python grc_nmt.py make_splits \
  --bitext data/bitext.jsonl \
  --train data/train.jsonl --dev data/dev.jsonl --test data/test.jsonl \
  --split_by_meta data/meta.tsv  # optional: columns id,work,author (id aligns to line index or a pair id)

# 3) Fine‑tune GRC→EN (default: Helsinki-NLP/opus-mt-mul-en)
python grc_nmt.py train \
  --train_jsonl data/train.jsonl --dev_jsonl data/dev.jsonl \
  --direction grc2en \
  --model_name Helsinki-NLP/opus-mt-mul-en \
  --save_dir runs/grc2en.opus \
  --num_epochs 24 --lr 5e-5 --batch 64 --grad_accum 1

# 4) Generate translations (e.g., EN→GRC back‑translation over monolingual EN)
python grc_nmt.py generate \
  --model_dir runs/en2grc.opus \
  --in_txt data/mono.en.txt --out_txt data/mono.en.synthetic.grc \
  --src_lang en --tgt_lang grc --num_beams 5 --max_new_tokens 256

# 5) Build back‑translated parallel for EN→GRC training
python grc_nmt.py build_bt \
  --source_txt data/mono.en.txt \
  --synthetic_txt data/mono.en.synthetic.grc \
  --out_jsonl data/bt.en_grc.jsonl --direction en2grc

# 6) Train EN→GRC with real + synthetic data
python grc_nmt.py train \
  --train_jsonl data/train.jsonl data/bt.en_grc.jsonl \
  --dev_jsonl data/dev.jsonl \
  --direction en2grc \
  --model_name Helsinki-NLP/opus-mt-en-mul \
  --save_dir runs/en2grc.opus

# 7) Evaluate on test set (BLEU + chrF)
python grc_nmt.py evaluate \
  --model_dir runs/grc2en.opus \
  --test_jsonl data/test.jsonl \
  --src_lang grc --tgt_lang en

Notes
— Ancient Greek handling
  • Normalization includes NFC, optional lowercasing, final‑sigma mapping, punctuation unification, and optional diacritics stripping.
  • Keep two corpora if desired: diacritics‑kept and diacritics‑stripped. Use --keep_diacritics or omit it.
— Models
  • Marian (Helsinki‑NLP/opus‑mt‑*) is the simplest to start with.
  • mBART‑50 or NLLB‑200 can be used; set --model_name accordingly. For mBART/NLLB you may need to set language codes (see code comments below).
— Repro tips
  • Always evaluate with SacreBLEU signatures and also chrF.
  • Consider domain tags (<BIBLE>, <HIST>, …) as prefixes to the source during preprocessing.
"""

from __future__ import annotations
import argparse
import json
import os
import random
import sys
import math
import regex as re
import unicodedata as ud
from dataclasses import dataclass
from typing import Iterable, List, Tuple, Dict, Optional

# Optional but helpful
try:
    from ftfy import fix_text
except Exception:  # pragma: no cover
    def fix_text(x: str) -> str:
        return x

from datasets import load_dataset, Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
import sacrebleu

# -----------------------------
# Utilities for normalization
# -----------------------------
GREEK_FINAL_SIGMA = "ς"  # ς
GREEK_SIGMA = "σ"        # σ
GREEK_ANO_TELEIA = "·"    # ·
GREEK_QUESTION = ";"      # ; (greek question mark variant, now deprecated but appears)

COMBINING_MARKS_RE = re.compile(r"\p{M}+")
WHITESPACE_RE = re.compile(r"\s+")

PUNCT_MAP = {
    "\u0387": GREEK_ANO_TELEIA,  # Greek ano teleia variant → ·
    ";": ";",  # keep ASCII semicolon
    "\u037E": ";",  # Greek question mark → ;
    "\u2019": "'",
    "\u2018": "'",
    "\u201C": '"',
    "\u201D": '"',
}


def strip_diacritics(text: str) -> str:
    # Remove all combining marks (accents, breathings, iota subscripts, etc.)
    return COMBINING_MARKS_RE.sub("", ud.normalize("NFD", text))


def norm_greek(text: str, keep_diacritics: bool = True, lowercase: bool = True) -> str:
    if not text:
        return text
    t = fix_text(text)
    t = ud.normalize("NFC", t)
    # unify punctuation
    t = "".join(PUNCT_MAP.get(ch, ch) for ch in t)
    # map final sigma to base sigma in the middle of words; restore at word ends later if you want
    t = t.replace(GREEK_FINAL_SIGMA, GREEK_SIGMA)
    if lowercase:
        t = t.lower()
    if not keep_diacritics:
        t = strip_diacritics(t)
        t = ud.normalize("NFC", t)
    # collapse whitespace
    t = WHITESPACE_RE.sub(" ", t).strip()
    return t


def norm_english(text: str, lowercase: bool = False) -> str:
    t = fix_text(text)
    t = ud.normalize("NFC", t)
    t = "".join(PUNCT_MAP.get(ch, ch) for ch in t)
    if lowercase:
        t = t.lower()
    t = WHITESPACE_RE.sub(" ", t).strip()
    return t

# -----------------------------
# I/O helpers
# -----------------------------

def read_lines(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


def write_jsonl(path: str, rows: Iterable[Dict]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(x) for x in f]

# -----------------------------
# Preprocess
# -----------------------------

def cmd_preprocess(args: argparse.Namespace):
    src = read_lines(args.src_file) if args.src_file else []
    tgt = read_lines(args.tgt_file) if args.tgt_file else []
    if src and tgt and len(src) != len(tgt):
        raise ValueError(f"Mismatched line counts: src={len(src)} tgt={len(tgt)}")

    out_rows = []
    if src and tgt:
        for s, t in zip(src, tgt):
            s2 = norm_greek(s, keep_diacritics=args.keep_diacritics, lowercase=args.lowercase) if args.lang.startswith("src_grc") else norm_english(s, lowercase=args.lowercase)
            t2 = norm_english(t, lowercase=False) if args.lang.startswith("src_grc") else norm_greek(t, keep_diacritics=args.keep_diacritics, lowercase=args.lowercase)
            out_rows.append({"src": s2, "tgt": t2})
    elif src and not tgt:
        # monolingual normalization; store as src only
        for s in src:
            s2 = norm_greek(s, keep_diacritics=args.keep_diacritics, lowercase=args.lowercase) if args.lang in ("grc", "src_grc") else norm_english(s, lowercase=args.lowercase)
            out_rows.append({"src": s2})
    else:
        raise ValueError("Provide at least --src_file; --tgt_file is required for parallel.")

    write_jsonl(args.out_jsonl, out_rows)
    print(f"Wrote {len(out_rows)} rows → {args.out_jsonl}")

# -----------------------------
# Make splits
# -----------------------------

def cmd_make_splits(args: argparse.Namespace):
    rows = read_jsonl(args.bitext)
    idx = list(range(len(rows)))

    # If metadata provided, try to group by work/author to split without leakage
    groups: Dict[str, List[int]] = {}
    if args.split_by_meta and os.path.exists(args.split_by_meta):
        meta = {}
        with open(args.split_by_meta, "r", encoding="utf-8") as f:
            # expect TSV: id \t work \t author
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t")
                if len(parts) < 2:
                    continue
                _id, work = parts[0], parts[1]
                meta[_id] = work
        # assume the jsonl rows have implicit ids 0..N-1 unless a field "id" exists
        for i, r in enumerate(rows):
            key = r.get("id", str(i))
            work = meta.get(str(key), f"_unk_{i}")
            groups.setdefault(work, []).append(i)
        # shuffle groups
        work_keys = list(groups.keys())
        random.Random(args.seed).shuffle(work_keys)
        n = len(work_keys)
        n_train = int(0.8 * n)
        n_dev = int(0.1 * n)
        train_idx = [i for k in work_keys[:n_train] for i in groups[k]]
        dev_idx = [i for k in work_keys[n_train:n_train + n_dev] for i in groups[k]]
        test_idx = [i for k in work_keys[n_train + n_dev:] for i in groups[k]]
    else:
        random.Random(args.seed).shuffle(idx)
        n = len(idx)
        n_train = int(0.8 * n)
        n_dev = int(0.1 * n)
        train_idx = idx[:n_train]
        dev_idx = idx[n_train:n_train + n_dev]
        test_idx = idx[n_train + n_dev:]

    def pick(idxs):
        return [rows[i] for i in idxs]

    write_jsonl(args.train, pick(train_idx))
    write_jsonl(args.dev, pick(dev_idx))
    write_jsonl(args.test, pick(test_idx))

    print(f"Splits → train={len(train_idx)} dev={len(dev_idx)} test={len(test_idx)}")

# -----------------------------
# Training
# -----------------------------

LANG_CODES_HINT = {
    # Helpful mBART‑50 codes. For NLLB, codes differ (e.g., eng_Latn, ell_Grek for Modern Greek; Ancient Greek not explicit).
    "en": "en_XX",
    # Ancient Greek lacks a standard mBART code; models trained on script‑agnostic tokenizers (Marian/NLLB byte‑level) work fine.
}


def load_bitext(*jsonl_paths: str) -> Dataset:
    rows: List[Dict] = []
    for p in jsonl_paths:
        rows.extend(read_jsonl(p))
    return Dataset.from_list(rows)


def tok_fn_builder(tokenizer, max_source_length: int, max_target_length: int, src_field: str, tgt_field: str):
    def _fn(batch):
        model_inputs = tokenizer(batch[src_field], max_length=max_source_length, truncation=True)
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(batch[tgt_field], max_length=max_target_length, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    return _fn


def cmd_train(args: argparse.Namespace):
    assert args.direction in {"grc2en", "en2grc"}

    # Build datasets
    train_ds = load_bitext(*args.train_jsonl)
    dev_ds = load_bitext(args.dev_jsonl)

    # Load model/tokenizer
    model_name = args.model_name
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # If model requires forced language tokens (e.g., mBART), you can adjust here based on direction
    forced_bos_token_id = None
    if hasattr(tokenizer, "lang_code_to_id") and args.forced_bos_lang:
        forced_bos_token_id = tokenizer.lang_code_to_id.get(args.forced_bos_lang)
        if forced_bos_token_id is None:
            print(f"Warning: lang code {args.forced_bos_lang} not in tokenizer.lang_code_to_id")

    preprocess_fn = tok_fn_builder(
        tokenizer,
        max_source_length=args.max_src_len,
        max_target_length=args.max_tgt_len,
        src_field="src",
        tgt_field="tgt",
    )

    train_ds = train_ds.map(preprocess_fn, batched=True, remove_columns=train_ds.column_names)
    dev_ds = dev_ds.map(preprocess_fn, batched=True, remove_columns=dev_ds.column_names)

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.save_dir,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=min(64, args.batch),
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.num_epochs,
        bf16=args.bf16,
        logging_steps=100,
        evaluation_strategy="steps",
        eval_steps=max(50, args.eval_every),
        save_steps=max(50, args.eval_every),
        save_total_limit=3,
        predict_with_generate=True,
        generation_max_length=args.max_tgt_len,
        load_best_model_at_end=True,
        metric_for_best_model="bleu",
        greater_is_better=True,
        report_to=["none"],
    )

    # Simple BLEU metric for dev during training
    def postprocess_text(preds, labels):
        preds = [p.strip() for p in preds]
        labels = [[l.strip()] for l in labels]
        return preds, labels

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels = [[l for l in label if l != -100] for l in labels]
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
        bleu = sacrebleu.corpus_bleu(decoded_preds, [decoded_labels]).score
        chrf = sacrebleu.corpus_chrf(decoded_preds, [decoded_labels]).score
        return {"bleu": bleu, "chrf": chrf}

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    if forced_bos_token_id is not None:
        model.config.forced_bos_token_id = forced_bos_token_id

    trainer.train()
    trainer.save_model(args.save_dir)
    tokenizer.save_pretrained(args.save_dir)
    print(f"Saved fine‑tuned model → {args.save_dir}")

# -----------------------------
# Generation (translate / back‑translate)
# -----------------------------

def batched(iterable: Iterable, n: int):
    buf = []
    for x in iterable:
        buf.append(x)
        if len(buf) >= n:
            yield buf
            buf = []
    if buf:
        yield buf


def load_model_and_tok(model_dir: str):
    tok = AutoTokenizer.from_pretrained(model_dir)
    mdl = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
    return mdl, tok


def cmd_generate(args: argparse.Namespace):
    model, tok = load_model_and_tok(args.model_dir)

    inputs = read_lines(args.in_txt)
    os.makedirs(os.path.dirname(args.out_txt), exist_ok=True)
    out_f = open(args.out_txt, "w", encoding="utf-8")

    gen_kwargs = dict(
        num_beams=args.num_beams,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        length_penalty=args.length_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
    )

    for batch in batched(inputs, args.batch):
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=args.max_src_len)
        with torch.no_grad():
            out_ids = model.generate(**enc, **gen_kwargs)
        decoded = tok.batch_decode(out_ids, skip_special_tokens=True)
        for line in decoded:
            out_f.write(line.strip() + "\n")

    out_f.close()
    print(f"Wrote {args.out_txt}")

# -----------------------------
# Build back‑translation JSONL
# -----------------------------

def cmd_build_bt(args: argparse.Namespace):
    src_lines = read_lines(args.source_txt)
    hyp_lines = read_lines(args.synthetic_txt)
    if len(src_lines) != len(hyp_lines):
        raise ValueError("source_txt and synthetic_txt must have same #lines")

    rows = []
    if args.direction == "en2grc":
        for en, grc in zip(src_lines, hyp_lines):
            rows.append({"src": norm_english(en), "tgt": norm_greek(grc)})
    elif args.direction == "grc2en":
        for grc, en in zip(src_lines, hyp_lines):
            rows.append({"src": norm_greek(grc), "tgt": norm_english(en)})
    else:
        raise ValueError("direction must be en2grc or grc2en")

    write_jsonl(args.out_jsonl, rows)
    print(f"Wrote BT parallel → {args.out_jsonl} ({len(rows)} pairs)")

# -----------------------------
# Evaluate
# -----------------------------

def cmd_evaluate(args: argparse.Namespace):
    import torch
    model, tok = load_model_and_tok(args.model_dir)

    data = read_jsonl(args.test_jsonl)
    src = [r["src"] for r in data]
    refs = [[r["tgt"] for r in data]]

    hyps: List[str] = []
    gen_kwargs = dict(
        num_beams=args.num_beams,
        max_new_tokens=args.max_new_tokens,
        length_penalty=args.length_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
    )

    for batch in batched(src, args.batch):
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=args.max_src_len)
        with torch.no_grad():
            out_ids = model.generate(**enc, **gen_kwargs)
        decoded = tok.batch_decode(out_ids, skip_special_tokens=True)
        hyps.extend([d.strip() for d in decoded])

    bleu = sacrebleu.corpus_bleu(hyps, refs)
    chrf = sacrebleu.corpus_chrf(hyps, refs)

    print("SacreBLEU:", bleu.format())
    print("chrF:", chrf.format())

# -----------------------------
# CLI
# -----------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Ancient Greek ⟷ English NMT pipeline")
    sp = p.add_subparsers(dest="cmd", required=True)

    # preprocess
    pp = sp.add_parser("preprocess", help="Normalize raw text and build JSONL bitext/monotext")
    pp.add_argument("--src_file", required=True)
    pp.add_argument("--tgt_file", default=None, help="Optional target file for parallel")
    pp.add_argument("--out_jsonl", required=True)
    pp.add_argument("--lang", default="src_grc", choices=["src_grc", "src_en", "grc", "en"], help="Which side is in --src_file")
    pp.add_argument("--keep_diacritics", action="store_true")
    pp.add_argument("--lowercase", action="store_true")
    pp.set_defaults(func=cmd_preprocess)

    # splits
    ms = sp.add_parser("make_splits", help="Create train/dev/test JSONL files")
    ms.add_argument("--bitext", required=True)
    ms.add_argument("--train", required=True)
    ms.add_argument("--dev", required=True)
    ms.add_argument("--test", required=True)
    ms.add_argument("--split_by_meta", default=None, help="Optional TSV with id,work,author for leakage‑free splits")
    ms.add_argument("--seed", type=int, default=13)
    ms.set_defaults(func=cmd_make_splits)

    # train
    tr = sp.add_parser("train", help="Fine‑tune a seq2seq model on bitext")
    tr.add_argument("--train_jsonl", nargs='+', required=True, help="One or more JSONL files (merged)")
    tr.add_argument("--dev_jsonl", required=True)
    tr.add_argument("--direction", choices=["grc2en", "en2grc"], required=True)
    tr.add_argument("--model_name", default="Helsinki-NLP/opus-mt-mul-en")
    tr.add_argument("--save_dir", required=True)
    tr.add_argument("--num_epochs", type=int, default=24)
    tr.add_argument("--lr", type=float, default=5e-5)
    tr.add_argument("--batch", type=int, default=64)
    tr.add_argument("--grad_accum", type=int, default=1)
    tr.add_argument("--bf16", action="store_true")
    tr.add_argument("--max_src_len", type=int, default=512)
    tr.add_argument("--max_tgt_len", type=int, default=512)
    tr.add_argument("--eval_every", type=int, default=500)
    tr.add_argument("--forced_bos_lang", default=None, help="e.g., en_XX for mBART; optional")
    tr.set_defaults(func=cmd_train)

    # generate
    ge = sp.add_parser("generate", help="Translate a text file line‑by‑line")
    ge.add_argument("--model_dir", required=True)
    ge.add_argument("--in_txt", required=True)
    ge.add_argument("--out_txt", required=True)
    ge.add_argument("--src_lang", required=True, choices=["en", "grc"])  # informational
    ge.add_argument("--tgt_lang", required=True, choices=["en", "grc"])  # informational
    ge.add_argument("--batch", type=int, default=32)
    ge.add_argument("--num_beams", type=int, default=5)
    ge.add_argument("--do_sample", action="store_true")
    ge.add_argument("--temperature", type=float, default=1.0)
    ge.add_argument("--top_p", type=float, default=0.9)
    ge.add_argument("--no_repeat_ngram_size", type=int, default=4)
    ge.add_argument("--length_penalty", type=float, default=1.0)
    ge.add_argument("--max_new_tokens", type=int, default=256)
    ge.add_argument("--max_src_len", type=int, default=512)
    ge.set_defaults(func=cmd_generate)

    # build_bt
    bt = sp.add_parser("build_bt", help="Create JSONL parallel from source + synthetic translations")
    bt.add_argument("--source_txt", required=True)
    bt.add_argument("--synthetic_txt", required=True)
    bt.add_argument("--out_jsonl", required=True)
    bt.add_argument("--direction", choices=["en2grc", "grc2en"], required=True)
    bt.set_defaults(func=cmd_build_bt)

    # evaluate
    ev = sp.add_parser("evaluate", help="Evaluate a model on a JSONL test set with SacreBLEU + chrF")
    ev.add_argument("--model_dir", required=True)
    ev.add_argument("--test_jsonl", required=True)
    ev.add_argument("--src_lang", required=True, choices=["en", "grc"])  # informational
    ev.add_argument("--tgt_lang", required=True, choices=["en", "grc"])  # informational
    ev.add_argument("--batch", type=int, default=32)
    ev.add_argument("--num_beams", type=int, default=5)
    ev.add_argument("--no_repeat_ngram_size", type=int, default=4)
    ev.add_argument("--length_penalty", type=float, default=1.0)
    ev.add_argument("--max_new_tokens", type=int, default=256)
    ev.add_argument("--max_src_len", type=int, default=512)
    ev.set_defaults(func=cmd_evaluate)

    return p


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    # Lazy import torch only when needed (generation/eval)
    if args.cmd in {"generate", "evaluate"}:
        global torch
        import torch  # noqa: F401

    args.func(args)


if __name__ == "__main__":
    main()
