"""Step 1: Sanity check the base Hugging Face dataset."""
from pathlib import Path
from datasets import load_dataset
import pandas as pd

Path("data").mkdir(exist_ok=True)

print("Loading theArijitDas/Fake-Reviews-Dataset ...")
ds = load_dataset("theArijitDas/Fake-Reviews-Dataset", split="train")
df = ds.to_pandas()

print(f"\nShape: {df.shape}")
print(f"\nLabel balance:\n{df['label'].value_counts()}")
print(f"\nCategories:\n{df['category'].value_counts()}")
print(f"\nWord count stats:\n{df['text'].str.split().str.len().describe()}")
print(f"\nMean char length per label:\n{df.groupby('label')['text'].apply(lambda x: x.str.len().mean())}")

df.to_parquet("data/base.parquet")
print("\nSaved base dataset to data/base.parquet")
