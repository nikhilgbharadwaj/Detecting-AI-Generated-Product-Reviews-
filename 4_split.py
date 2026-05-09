"""Step 4: Merge base + augmented, build train/val/test + held-out generator set."""
import os
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
HELDOUT_MODEL = os.getenv("HELDOUT_MODEL", "qwen2.5:7b")

base = pd.read_parquet("data/base.parquet")
aug_full = pd.read_csv("data/augmented_clean.csv")

print(f"Base: {len(base)}  |  Augmented (clean): {len(aug_full)}")
print(f"Holding out generator: {HELDOUT_MODEL}")

heldout = aug_full[aug_full["_model"] == HELDOUT_MODEL][["category","rating","text","label"]]
aug_train = aug_full[aug_full["_model"] != HELDOUT_MODEL][["category","rating","text","label"]]

combined = (pd.concat([base, aug_train], ignore_index=True)
              .sample(frac=1, random_state=42)
              .reset_index(drop=True))

n = len(combined)
train = combined[:int(0.8*n)]
val   = combined[int(0.8*n):int(0.9*n)]
test  = combined[int(0.9*n):]

train.to_parquet("data/train.parquet")
val.to_parquet("data/val.parquet")
test.to_parquet("data/test.parquet")
heldout.to_parquet("data/test_heldout_generator.parquet")

print(f"\ntrain: {len(train)}  |  val: {len(val)}  |  test: {len(test)}  |  heldout: {len(heldout)}")
print("\nLabel balance in train:")
print(train["label"].value_counts(normalize=True))
