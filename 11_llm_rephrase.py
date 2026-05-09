"""Phase 4 — Step 1: LLM rephraser.

Takes the structured Explanation from 10_explain.py and asks gemma2:2b to
rewrite the technical reasons into a 1-2 sentence natural-language summary.

The LLM is strictly downstream of the deterministic pipeline:
  - It does NOT decide the verdict (XGBoost did that).
  - It does NOT re-rank reasons (SHAP did that).
  - It ONLY rephrases existing facts into prose.

Temperature = 0 for determinism. Same input -> same output.
"""
from __future__ import annotations
import ollama

MODEL = "gemma2:2b"

SYSTEM_PROMPT = """You are a writing assistant. You will receive a list of technical reasons explaining why a product review was flagged as AI-generated. Your job is to rewrite these reasons into 1 or 2 short sentences of plain English that a normal shopper can understand.

STRICT RULES:
- Use ONLY the facts in the reasons provided. Do NOT invent new claims.
- Do NOT mention any technical terms (perplexity, burstiness, SHAP, etc.).
- Do NOT use numbers or statistics from the reasons.
- Do NOT add disclaimers, opinions, or recommendations.
- Do NOT start with "This review" or "The review". Start directly with the explanation.
- Maximum 2 sentences. Keep it conversational.
- Output ONLY the rewritten explanation, nothing else."""

def rephrase(verdict: str, ai_probability: float, reasons: list) -> str:
    """Returns a 1-2 sentence plain-English explanation, or a fallback string."""

    # Skip the LLM entirely for human verdicts — nothing to explain.
    if verdict == "likely_human":
        return "This review reads like a normal human-written review. No suspicious patterns were detected."

    if not reasons:
        if verdict == "uncertain":
            return "The signals are mixed — this review has some AI-like qualities but isn't a clear case either way."
        return "This review shows several patterns common in AI-generated text."

    # Build a clean bullet list for the LLM
    reason_lines = "\n".join(f"- {r.text}" for r in reasons[:5])
    confidence_word = "very likely" if ai_probability >= 0.85 else "likely" if ai_probability >= 0.65 else "possibly"

    user_prompt = f"""The review was flagged as {confidence_word} AI-generated. Technical reasons:
{reason_lines}

Rewrite into 1-2 plain-English sentences for a regular shopper."""

    try:
        resp = ollama.chat(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            options={"temperature": 0, "num_predict": 120},
        )
        text = resp["message"]["content"].strip().strip('"').strip()
        # Defensive trimming: strip leading "Here's...:" / "Sure...:" preambles
        for prefix in ("here's", "here is", "sure,", "okay,", "ok,"):
            if text.lower().startswith(prefix):
                # Find the first sentence-ending colon or period and skip past it
                for sep in (":\n", ": ", ".\n"):
                    idx = text.find(sep)
                    if 0 < idx < 40:
                        text = text[idx + len(sep):].strip()
                        break
        return text
    except Exception as e:
        # Graceful fallback: return the first reason verbatim if Ollama is down.
        return reasons[0].text


if __name__ == "__main__":
    # Quick smoke test using the dataclasses from 10_explain.py
    import importlib.util, sys
    spec = importlib.util.spec_from_file_location("expl", "10_explain.py")
    expl_mod = importlib.util.module_from_spec(spec); sys.modules["expl"] = expl_mod
    spec.loader.exec_module(expl_mod)

    fake_reasons = [
        expl_mod.Reason("structural", "Contains 5 phrase(s) commonly seen in AI-generated reviews.", 0.9),
        expl_mod.Reason("statistical", "Sentence rhythm is unnaturally uniform (burstiness 34.1 vs ~230); humans alternate predictable and surprising phrasing.", 0.6),
        expl_mod.Reason("statistical", "Statistically too predictable (perplexity 16.7 vs ~86 for typical human reviews) — the wording follows common LLM patterns.", 0.5),
    ]
    print("AI sample:")
    print(rephrase("likely_ai", 0.99, fake_reasons))
    print("\nHuman sample:")
    print(rephrase("likely_human", 0.02, []))
