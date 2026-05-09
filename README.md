# AI Review Detector — Data Pipeline

Local VS Code project to build training data for the AI review detector.
Replaces the Colab notebook with a proper Python project.

## One-time setup

### 1. Install Ollama (once)

Download from https://ollama.com and install. Then pull the models:

```bash
ollama pull llama3.1:8b
ollama pull mistral:7b
ollama pull qwen2.5:7b
ollama pull phi3:mini
```

Models live in `~/.ollama/models` and persist across reboots.

### 2. Python environment

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

pip install -r requirements.txt
```

### 3. Verify GPU + Ollama

```bash
ollama list                    # should show 4 models
ollama run llama3.1:8b "hi"    # should respond quickly, GPU usage spikes
```

## Pipeline (run in order)

```bash
python 1_explore.py         # sanity check the base dataset
python 2_generate.py        # generate augmented AI reviews (long, resumable)
python 3_clean.py           # filter refusals, dedupe, length-check
python 4_split.py           # merge, split train/val/test, hold out one generator
python 5_upload.py          # (optional) push splits to your HF dataset repo
```

Output files land in `./data/`.

## Resuming generation

`2_generate.py` checkpoints every 50 successful samples to `data/augmented_raw.csv`.
If it crashes or you stop it, re-run — it resumes from the last checkpoint
automatically.

To force a fresh run: delete `data/augmented_raw.csv`.

## Target

6,000 samples across 4 models, 10 categories, varied prompts/temps/lengths.
Expect ~4-8 hours on an RTX 5070 depending on length mix.

