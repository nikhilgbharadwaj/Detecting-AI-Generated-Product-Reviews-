"""Phase 2 Step 3: Rule-based red-flag features for val + test sets.

Reads:  data/val_features.parquet, data/test_features.parquet
Writes: data/val_full.parquet,     data/test_full.parquet
        (original cols + DistilBERT prob + 16 stylometric + 11 rule features)

Pure Python + datasketch. Runs in seconds. No GPU, no model loading.

The 11 rule features (all numeric so XGBoost can use them directly):
  - rule_template_hits           # how many known AI template phrases match
  - rule_starts_with_rating      # opens with "5/5", "★★★★★", "Rating:", etc.
  - rule_markdown_artifacts      # leftover **bold**, ##, bullet markers
  - rule_pros_cons_structure     # contains "Pros:" / "Cons:" sections
  - rule_superlative_density     # ratio of generic-adj clusters
  - rule_sentiment_rating_mismatch  # 5-star review with negative words, etc.
  - rule_emoji_count             # AI rarely uses emojis; humans sometimes do
  - rule_avg_paragraph_perfection  # extremely uniform paragraph lengths
  - rule_buyer_persona_phrases   # "as a parent", "as a college student", etc.
  - rule_near_duplicate          # near-duplicate of another review in the set
  - rule_max_jaccard             # max Jaccard similarity to any other review
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd
from datasketch import MinHash, MinHashLSH
from tqdm import tqdm

DATA = Path("data")

# ---------------------------------------------------------------------------
# Lexicons & patterns
# ---------------------------------------------------------------------------

# Phrases observed disproportionately in LLM-generated reviews.
# Compiled from your HF dataset preview + general LLM tells.
TEMPLATE_PHRASES = [
    r"\bi recently purchased\b",
    r"\bi'm blown away\b",
    r"\bblown away by\b",
    r"\boverall,? (i|this|it)\b",
    r"\bi highly recommend\b",
    r"\bhighly recommended\b",
    r"\bi was impressed by\b",
    r"\bi was pleasantly surprised\b",
    r"\bexceeded my expectations\b",
    r"\bgame[- ]changer\b",
    r"\bworth every penny\b",
    r"\bthe perfect (addition|solution|gift|choice)\b",
    r"\bi can('?t| not) recommend.*enough\b",
    r"\bin (conclusion|summary)\b",
    r"\bnot only.*but also\b",
    r"\bsleek (and|design|aesthetic)\b",
    r"\bintuitive (design|interface|features)\b",
    r"\buser[- ]friendly\b",
    r"\btop[- ]notch\b",
    r"\bvalue for money\b",
    r"\bmust[- ]have\b",
]

# Persona-prompt leakage — these strongly suggest "write as X" prompting
PERSONA_PHRASES = [
    r"\bas a (busy )?parent\b",
    r"\bas a (college )?student\b",
    r"\bas a (tech )?enthusiast\b",
    r"\bas a (first[- ]time )?buyer\b",
    r"\bas a frequent shopper\b",
    r"\bas a retiree\b",
    r"\bas a gift giver\b",
]

# Rating-style openers
RATING_OPENER = re.compile(
    r"^\s*(?:"
    r"\d\s*[/\\]\s*\d\s*stars?|"     # 5/5 stars
    r"\d\s*stars?\s*[!.:]?|"         # 5 stars!
    r"★+|"                            # ★★★★★
    r"\*+|"                           # ***
    r"rating\s*[:\-]\s*\d|"           # Rating: 5
    r"⭐+"
    r")",
    re.IGNORECASE,
)

# Markdown artifacts that survived from LLM output
MARKDOWN_RE = re.compile(r"\*\*[^*]+\*\*|^#{1,3}\s|^\s*[-*]\s")

PROS_CONS_RE = re.compile(r"\bpros\s*[:\-]|\bcons\s*[:\-]", re.IGNORECASE)

GENERIC_ADJ = {"great", "amazing", "perfect", "wonderful", "excellent",
               "fantastic", "awesome", "incredible", "outstanding", "superb",
               "impressive", "remarkable", "exceptional", "stunning", "lovely"}

POSITIVE_WORDS = GENERIC_ADJ | {"love", "loved", "best", "happy", "satisfied",
                                 "recommend", "good", "nice"}
NEGATIVE_WORDS = {"bad", "worst", "terrible", "awful", "disappointed",
                  "broken", "useless", "waste", "hate", "horrible", "poor",
                  "defective", "refund", "returned", "frustrating"}

EMOJI_RE = re.compile(
    "[" "\U0001F600-\U0001F64F" "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF" "\U0001F1E0-\U0001F1FF" "\u2600-\u26FF\u2700-\u27BF" "]",
    flags=re.UNICODE,
)

WORD_RE = re.compile(r"[A-Za-z']+")

# Pre-compile template patterns
TEMPLATE_REGEXES = [re.compile(p, re.IGNORECASE) for p in TEMPLATE_PHRASES]
PERSONA_REGEXES = [re.compile(p, re.IGNORECASE) for p in PERSONA_PHRASES]


# ---------------------------------------------------------------------------
# Per-review rule features (no inter-review computation here)
# ---------------------------------------------------------------------------
def per_review_rules(text: str, rating: float | None) -> dict:
    text = text or ""
    low = text.lower()
    words = WORD_RE.findall(low)
    n_words = max(len(words), 1)

    template_hits = sum(1 for r in TEMPLATE_REGEXES if r.search(low))
    persona_hits = sum(1 for r in PERSONA_REGEXES if r.search(low))

    starts_rating = 1 if RATING_OPENER.match(text) else 0
    md_hits = len(MARKDOWN_RE.findall(text))
    pros_cons = 1 if PROS_CONS_RE.search(text) else 0

    # Superlative density: clusters of generic adjectives within 5 words
    word_seq = words
    super_clusters = 0
    last_idx = -10
    for i, w in enumerate(word_seq):
        if w in GENERIC_ADJ:
            if i - last_idx <= 5:
                super_clusters += 1
            last_idx = i
    super_density = super_clusters / n_words

    # Sentiment vs rating mismatch
    pos = sum(1 for w in words if w in POSITIVE_WORDS)
    neg = sum(1 for w in words if w in NEGATIVE_WORDS)
    mismatch = 0
    if rating is not None and not (isinstance(rating, float) and np.isnan(rating)):
        if rating >= 4 and neg > pos and neg >= 2:
            mismatch = 1
        elif rating <= 2 and pos > neg and pos >= 2:
            mismatch = 1

    emoji_n = len(EMOJI_RE.findall(text))

    # Paragraph uniformity (very uniform paragraphs => AI-like)
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    if len(paragraphs) >= 2:
        para_lens = [len(p.split()) for p in paragraphs]
        # coefficient of variation; lower = more uniform = more AI-like
        mean_pl = np.mean(para_lens)
        cv = (np.std(para_lens) / mean_pl) if mean_pl > 0 else 0.0
        para_perfection = 1.0 / (1.0 + cv)   # in (0, 1], higher = more uniform
    else:
        para_perfection = 0.0

    return {
        "rule_template_hits": float(template_hits),
        "rule_buyer_persona_phrases": float(persona_hits),
        "rule_starts_with_rating": float(starts_rating),
        "rule_markdown_artifacts": float(md_hits),
        "rule_pros_cons_structure": float(pros_cons),
        "rule_superlative_density": float(super_density),
        "rule_sentiment_rating_mismatch": float(mismatch),
        "rule_emoji_count": float(emoji_n),
        "rule_avg_paragraph_perfection": float(para_perfection),
    }


# ---------------------------------------------------------------------------
# Cross-review rules: near-duplicate detection via MinHash + LSH
# ---------------------------------------------------------------------------
def shingles(text: str, k: int = 5) -> set:
    """Word-level k-shingles for MinHash."""
    words = WORD_RE.findall(text.lower())
    if len(words) < k:
        return {" ".join(words)} if words else {""}
    return {" ".join(words[i:i + k]) for i in range(len(words) - k + 1)}

def compute_dup_features(texts: list[str], num_perm: int = 64,
                          threshold: float = 0.6):
    """For each text, return (is_near_dup, max_jaccard_to_any_other).

    Uses MinHash LSH for fast lookup. Self-matches are ignored.
    """
    minhashes = []
    for t in texts:
        m = MinHash(num_perm=num_perm)
        for s in shingles(t):
            m.update(s.encode("utf-8"))
        minhashes.append(m)

    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    for i, m in enumerate(minhashes):
        lsh.insert(str(i), m)

    is_dup = np.zeros(len(texts), dtype=np.float32)
    max_jacc = np.zeros(len(texts), dtype=np.float32)

    for i, m in enumerate(tqdm(minhashes, desc="  near-dup")):
        candidates = [int(j) for j in lsh.query(m) if int(j) != i]
        if not candidates:
            continue
        is_dup[i] = 1.0
        # Compute exact Jaccard for the top candidate (cheap; few candidates)
        best = 0.0
        for j in candidates[:10]:
            best = max(best, m.jaccard(minhashes[j]))
        max_jacc[i] = best

    return is_dup, max_jacc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def process(in_path: Path, out_path: Path):
    df = pd.read_parquet(in_path)
    print(f"\n{in_path.name}: {len(df)} rows")

    # Per-review features
    rule_rows = []
    ratings = df["rating"].tolist() if "rating" in df.columns else [None] * len(df)
    for text, rating in tqdm(zip(df["text"], ratings), total=len(df), desc="  rules"):
        rule_rows.append(per_review_rules(text, rating))
    rule_df = pd.DataFrame(rule_rows)

    # Near-duplicate (cross-review)
    is_dup, max_jacc = compute_dup_features(df["text"].tolist())
    rule_df["rule_near_duplicate"] = is_dup
    rule_df["rule_max_jaccard"] = max_jacc

    out = pd.concat([df.reset_index(drop=True), rule_df], axis=1)
    out.to_parquet(out_path)
    print(f"  -> {out_path}  ({len(out.columns)} cols)")

    # Sanity: mean by label
    rule_cols = list(rule_df.columns)
    print("\n  Mean rule-feature values by label:")
    summary = out.groupby("label")[rule_cols].mean().T
    summary.columns = [f"label={int(c)}" for c in summary.columns]
    summary["diff"] = summary.iloc[:, 1] - summary.iloc[:, 0]
    print(summary.round(4).to_string())


if __name__ == "__main__":
    val_in = DATA / "val_features.parquet"
    test_in = DATA / "test_features.parquet"
    assert val_in.exists() and test_in.exists(), \
        "Missing inputs. Run 7_stylometric.py first."

    process(val_in,  DATA / "val_full.parquet")
    process(test_in, DATA / "test_full.parquet")

    print("\nDone. Next: Phase 2 Step 4 (XGBoost fusion layer).")
