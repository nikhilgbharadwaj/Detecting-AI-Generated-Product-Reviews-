"""Step 3: Clean augmented data — filter refusals, dedupe, length-check."""
from pathlib import Path
import pandas as pd

IN = Path("data/augmented_raw.csv")
OUT = Path("data/augmented_clean.csv")
assert IN.exists(), f"Run 2_generate.py first. Missing {IN}"

df = pd.read_csv(IN)
print(f"Raw samples: {len(df)}")

# Filter refusals and LLM artifacts
refusal_patterns = (
    "as an ai|i cannot|i can't write|i'm unable|i am unable|"
    "here's a|here is a|sure,|certainly,|i don't feel comfortable|"
    "i do not feel comfortable|of course,"
)
bad = df["text"].str.lower().str.contains(refusal_patterns, regex=True, na=False)
print(f"Refusals filtered: {bad.sum()}")
df = df[~bad]

# Strip common LLM preambles/wrappers
df["text"] = (df["text"]
    .str.replace(r"^\*+\s*", "", regex=True)   # leading asterisks
    .str.replace(r"^#+\s*", "", regex=True)    # leading markdown headers
    .str.strip())

# Dedupe + length sanity
before = len(df)
df = df.drop_duplicates(subset="text")
print(f"Duplicates removed: {before - len(df)}")

wc = df["text"].str.split().str.len()
df = df[(wc >= 5) & (wc <= 300)]
print(f"After length filter [5-300 words]: {len(df)}")

# Per-model breakdown
print("\nPer-model distribution:")
print(df["_model"].value_counts())

df.to_csv(OUT, index=False)
print(f"\nSaved to {OUT}")
