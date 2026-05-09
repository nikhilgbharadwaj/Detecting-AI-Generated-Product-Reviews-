"""Step 2: Generate AI-labeled reviews via Ollama. Resumable. Batched by model.

Key design:
- Processes all samples for one model before switching to the next.
  On 8GB VRAM this means ~4 model loads instead of thousands of swaps.
- Checkpoints by # of successful samples (not loop iteration).
- Auto-resumes from data/augmented_raw.csv if it exists. Already-generated
  samples per model are counted and only the remaining quota is generated.
- Graceful Ctrl-C saves before exit.
"""
import os, random, time, signal, sys
from pathlib import Path
from collections import Counter
import pandas as pd
import ollama
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()
TARGET = int(os.getenv("TARGET_SAMPLES", "6000"))
OUT = Path("data/augmented_raw.csv")
Path("data").mkdir(exist_ok=True)

CATEGORIES = ["Home_and_Kitchen", "Electronics", "Books", "Clothing_Shoes_and_Jewelry",
              "Sports_and_Outdoors", "Toys_and_Games", "Beauty", "Pet_Supplies",
              "Grocery_and_Gourmet_Food", "Office_Products"]

MODELS = ["llama3.1:8b", "mistral:7b", "qwen2.5:7b", "phi3:mini"]

PROMPT_TEMPLATES = [
    "Write a {stars}-star Amazon review for a {category} product. {length_hint}. Just the review, no preamble.",
    "You bought a {category} item. Write a {stars}-star review as a {persona}. {length_hint}.",
    "Write a genuine-sounding {stars}-star product review for something in {category}. {length_hint}. Do not start with 'I'.",
    "Review a {category} product you recently purchased. Rating: {stars}/5. {length_hint}.",
]

PERSONAS = ["busy parent", "college student", "retiree", "tech enthusiast",
            "first-time buyer", "gift giver", "frequent shopper"]
LENGTH_HINTS = ["Keep it under 20 words", "Around 30-50 words",
                "Around 80-120 words", "A detailed 150-200 word review"]

# ---- Resume support ------------------------------------------------------
results = []
if OUT.exists():
    existing = pd.read_csv(OUT)
    results = existing.to_dict("records")
    print(f"Resuming from checkpoint: {len(results)} samples already collected.")
else:
    print("Starting fresh.")

def save():
    pd.DataFrame(results).to_csv(OUT, index=False)

def handler(sig, frame):
    print("\nInterrupted. Saving checkpoint...")
    save()
    sys.exit(0)
signal.signal(signal.SIGINT, handler)

# ---- Per-model quotas ----------------------------------------------------
per_model_target = TARGET // len(MODELS)
already = Counter(r["_model"] for r in results)

print(f"\nTarget per model: {per_model_target}")
print("Already collected per model:")
for m in MODELS:
    print(f"  {m:20s} {already.get(m, 0)}")

def generate_one(model, category, stars, persona, length_hint, template, temperature):
    """Single generation call. Returns cleaned text or None if too short."""
    prompt = template.format(stars=stars, category=category,
                             persona=persona, length_hint=length_hint)
    resp = ollama.generate(model=model, prompt=prompt,
                           options={"temperature": temperature, "num_predict": 300})
    text = resp["response"].strip().strip('"').strip()
    if len(text.split()) < 5:
        return None
    return text

# ---- Main loop: one model at a time --------------------------------------
total_start = time.time()

for model in MODELS:
    needed = per_model_target - already.get(model, 0)
    if needed <= 0:
        print(f"\n[{model}] quota already met, skipping.")
        continue

    print(f"\n{'='*60}")
    print(f"Loading {model} - generating {needed} samples")
    print(f"{'='*60}")

    # Warm up: one throwaway call forces the model into VRAM before the bar starts
    warm_start = time.time()
    try:
        ollama.generate(model=model, prompt="ok", options={"num_predict": 1})
        print(f"Model loaded in {time.time() - warm_start:.1f}s")
    except Exception as e:
        print(f"Failed to load {model}: {e}  - skipping.")
        continue

    pbar = tqdm(total=needed, desc=model, unit="sample")
    collected = 0
    attempts = 0
    model_start = time.time()

    while collected < needed:
        attempts += 1
        category = random.choice(CATEGORIES)
        stars = random.choices([1, 2, 3, 4, 5], weights=[1, 1, 1, 3, 6])[0]
        persona = random.choice(PERSONAS)
        length_hint = random.choice(LENGTH_HINTS)
        template = random.choice(PROMPT_TEMPLATES)
        temperature = round(random.uniform(0.5, 1.2), 2)

        try:
            text = generate_one(model, category, stars, persona,
                                length_hint, template, temperature)
            if text is None:
                continue
            results.append({
                "category": category, "rating": stars, "text": text,
                "label": 1, "_model": model, "_temp": temperature,
            })
            collected += 1
            pbar.update(1)
            if len(results) % 50 == 0:
                save()
                pbar.set_postfix(last=text[:40])
        except Exception as e:
            pbar.write(f"error [{model}]: {e}")
            time.sleep(2)

    pbar.close()
    elapsed = time.time() - model_start
    rate = collected / elapsed if elapsed else 0
    print(f"[{model}] done: {collected} samples in {elapsed/60:.1f} min "
          f"({rate:.2f} samples/s, {attempts} attempts)")
    save()

# ---- Done -----------------------------------------------------------------
save()
total_elapsed = time.time() - total_start
print(f"\n{'='*60}")
print(f"All done. Total samples: {len(results)}  in  {total_elapsed/60:.1f} min")
print(f"Saved to {OUT}")
print("\nPer-model breakdown:")
for m, c in Counter(r["_model"] for r in results).items():
    print(f"  {m:20s} {c}")
