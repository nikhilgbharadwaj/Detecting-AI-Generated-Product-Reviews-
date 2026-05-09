"""Phase 2 Step 2: Compute stylometric features for val + test sets.

Reads the DistilBERT prediction parquets you downloaded from Colab
(val_distilbert_preds.parquet, test_distilbert_preds.parquet), adds 16
numeric stylometric features per review, and writes new parquet files
that the fusion layer will consume.

Inputs (place in ./data/):
    val_distilbert_preds.parquet
    test_distilbert_preds.parquet

Outputs:
    data/val_features.parquet   (text, label, distilbert_prob, + 16 features)
    data/test_features.parquet  (same schema)

Runs on CPU. ~5-10 minutes for ~9k reviews because of GPT-2 perplexity.
GPU is used automatically if available (much faster).
"""
import os
import re
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

# -------------------------------------------------------------------------
# Setup
# -------------------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

DATA = Path("data")
DATA.mkdir(exist_ok=True)

print("Loading GPT-2 small for perplexity...")
gpt2_tok = GPT2TokenizerFast.from_pretrained("gpt2")
gpt2 = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE).eval()

# -------------------------------------------------------------------------
# Lexicons
# -------------------------------------------------------------------------
GENERIC_ADJ = {"great", "amazing", "perfect", "wonderful", "excellent",
               "fantastic", "awesome", "incredible", "outstanding", "superb",
               "impressive", "remarkable", "exceptional", "stunning", "lovely"}

HEDGING = {"perhaps", "somewhat", "quite", "rather", "pretty", "fairly",
           "relatively", "arguably", "presumably", "supposedly", "seemingly"}

FIRST_PERSON = {"i", "me", "my", "mine", "myself", "i'm", "i've", "i'd", "i'll"}

WORD_RE = re.compile(r"[A-Za-z']+")
SENT_RE = re.compile(r"(?<=[.!?])\s+")

# -------------------------------------------------------------------------
# Perplexity
# -------------------------------------------------------------------------
@torch.no_grad()
def perplexity(text: str, max_tokens: int = 512) -> float:
    """Cross-entropy perplexity under GPT-2. Lower = more predictable text."""
    if not text or not text.strip():
        return float("nan")
    enc = gpt2_tok(text, return_tensors="pt", truncation=True, max_length=max_tokens)
    input_ids = enc["input_ids"].to(DEVICE)
    if input_ids.size(1) < 2:
        return float("nan")
    out = gpt2(input_ids, labels=input_ids)
    # Convert nats -> perplexity. Clip to avoid inf on very short snippets.
    return float(min(math.exp(out.loss.item()), 1e6))

def sentence_perplexities(text: str):
    sents = [s.strip() for s in SENT_RE.split(text) if s.strip()]
    sents = [s for s in sents if len(s.split()) >= 2]
    if not sents:
        return []
    return [perplexity(s) for s in sents]

# -------------------------------------------------------------------------
# Per-review feature extraction
# -------------------------------------------------------------------------
def extract_features(text: str) -> dict:
    text = text or ""
    words = WORD_RE.findall(text.lower())
    n_words = len(words) or 1
    n_chars = len(text) or 1

    # Sentence-level
    sents = [s.strip() for s in SENT_RE.split(text) if s.strip()]
    sent_lens = [len(WORD_RE.findall(s)) for s in sents] or [n_words]

    # Perplexity (whole + per-sentence)
    full_pp = perplexity(text)
    sent_pps = sentence_perplexities(text)
    if sent_pps:
        mean_sent_pp = float(np.nanmean(sent_pps))
        burstiness = float(np.nanstd(sent_pps))
    else:
        mean_sent_pp = full_pp
        burstiness = 0.0

    # Lexical diversity
    type_token = len(set(words)) / n_words

    # Surface features
    punct_chars = sum(1 for c in text if c in ".,!?;:'\"()-")
    excl = text.count("!")
    quest = text.count("?")
    caps_words = sum(1 for w in text.split() if len(w) > 1 and w.isupper())
    digits = sum(1 for c in text if c.isdigit())
    mean_word_len = float(np.mean([len(w) for w in words])) if words else 0.0

    # Lexicon-based ratios
    generic_adj = sum(1 for w in words if w in GENERIC_ADJ)
    hedging = sum(1 for w in words if w in HEDGING)
    first_person = sum(1 for w in words if w in FIRST_PERSON)

    return {
        "perplexity": full_pp,
        "mean_sentence_perplexity": mean_sent_pp,
        "burstiness": burstiness,
        "type_token_ratio": type_token,
        "mean_sentence_length": float(np.mean(sent_lens)),
        "std_sentence_length": float(np.std(sent_lens)),
        "mean_word_length": mean_word_len,
        "punctuation_density": punct_chars / n_chars,
        "exclamation_ratio": excl / n_words,
        "question_ratio": quest / n_words,
        "caps_ratio": caps_words / max(len(text.split()), 1),
        "generic_adjective_ratio": generic_adj / n_words,
        "hedging_ratio": hedging / n_words,
        "first_person_ratio": first_person / n_words,
        "digit_ratio": digits / n_chars,
        "word_count": float(n_words),
    }

# -------------------------------------------------------------------------
# Process a split
# -------------------------------------------------------------------------
def process(in_path: Path, out_path: Path):
    df = pd.read_parquet(in_path)
    print(f"\n{in_path.name}: {len(df)} rows")

    feats = []
    for text in tqdm(df["text"].tolist(), desc=f"  features"):
        feats.append(extract_features(text))

    feat_df = pd.DataFrame(feats)
    out = pd.concat([df.reset_index(drop=True), feat_df], axis=1)

    # Sanity: replace any inf/nan with median of column (robust default)
    feat_cols = list(feat_df.columns)
    for c in feat_cols:
        col = out[c].replace([np.inf, -np.inf], np.nan)
        out[c] = col.fillna(col.median())

    out.to_parquet(out_path)
    print(f"  -> {out_path}  ({len(out.columns)} cols)")

    # Quick separability sanity check (mean by label)
    print("\n  Mean feature values by label:")
    summary = out.groupby("label")[feat_cols].mean().T
    summary.columns = [f"label={int(c)}" for c in summary.columns]
    summary["diff"] = summary.iloc[:, 1] - summary.iloc[:, 0]
    print(summary.round(3).to_string())

# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------
if __name__ == "__main__":
    val_in = DATA / "val_distilbert_preds.parquet"
    test_in = DATA / "test_distilbert_preds.parquet"

    assert val_in.exists(), f"Missing {val_in}. Download from Colab Cell 10."
    assert test_in.exists(), f"Missing {test_in}. Download from Colab Cell 10."

    process(val_in, DATA / "val_features.parquet")
    process(test_in, DATA / "test_features.parquet")

    print("\nDone. Next step: Phase 2 Step 3 (rule-based features).")
