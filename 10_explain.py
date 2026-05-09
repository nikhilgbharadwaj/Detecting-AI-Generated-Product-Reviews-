"""Phase 3: Explainability layer for the AI review detector.

Combines four explanation sources, all deterministic, no LLMs:

  1. Stylometric SHAP on the XGBoost fusion model (via XGBoost's native
     pred_contribs to bypass shap library compatibility issues).
  2. Rule callouts — direct factual statements when rule features fire.
  3. Token-level SHAP on DistilBERT — highlighted spans in the review text.
     Computed only for high-confidence-AI verdicts (>= EXPLAIN_THRESHOLD)
     and cached by SHA-1 of the text.
  4. Final verdict with reasons attached.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import shap
import torch
import xgboost as xgb
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                           GPT2LMHeadModel, GPT2TokenizerFast)

import importlib.util, sys
import warnings, logging
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

def _import(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); sys.modules[name] = mod
    spec.loader.exec_module(mod); return mod

styl  = _import("styl",  "7_stylometric.py")
rules = _import("rules", "8_rules.py")

# ---------------------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA = Path("data")
MODEL_DIR = Path("distilbert_ai_review_detector")
EXPLAIN_THRESHOLD = 0.65
TOKEN_TOPK = 5
MIN_TOKEN_LEN = 3
CACHE_DB = DATA / "shap_cache.sqlite"

HUMAN_NORMS = {
    "perplexity": 86.0,
    "mean_sentence_perplexity": 240.0,
    "burstiness": 230.0,
    "type_token_ratio": 0.81,
}

TEMPLATES = {
    "perplexity": (-1, "Statistically too predictable (perplexity {v:.1f} vs ~{n:.0f} for typical human reviews) — the wording follows common LLM patterns."),
    "mean_sentence_perplexity": (-1, "Sentences are individually too predictable ({v:.1f} vs ~{n:.0f}) — characteristic of generated text."),
    "burstiness": (-1, "Sentence rhythm is unnaturally uniform (burstiness {v:.1f} vs ~{n:.0f}); humans alternate predictable and surprising phrasing."),
    "type_token_ratio": (-1, "Vocabulary is less varied than typical human reviews ({v:.2f} vs ~{n:.2f})."),
    "word_count":   (+1, "Length and structure resemble templated AI output ({v:.0f} words)."),
    "first_person_ratio": (+1, "Heavy use of first-person pronouns ({v:.0%}) — common when an LLM is prompted to write 'as a buyer'."),
    "mean_sentence_length": (-1, "Sentence length pattern matches typical AI output."),
    "std_sentence_length":  (-1, "Sentence-length variance is unusually low for human writing."),
    "mean_word_length":     (+1, "Word-length distribution skews longer than typical for human reviews."),
    "punctuation_density":  (+1, "Punctuation usage is unusually formal for a customer review."),
    "exclamation_ratio":    (-1, "Exclamation usage is muted in a way typical of LLM output."),
    "caps_ratio":           (-1, "Lacks the casual capitalization patterns humans typically use."),
    "generic_adjective_ratio": (+1, "Heavy use of generic praise words like \"amazing\" / \"perfect\" / \"great\"."),
    "digit_ratio":          (-1, "No specific numbers (sizes, dates, prices) — humans usually mention these."),
    "hedging_ratio":        (+1, "Frequent hedging language (\"perhaps\", \"quite\", \"rather\") typical of polished LLM prose."),
    "question_ratio":       (-1, "No rhetorical questions — humans often use these."),
}

RULE_TEMPLATES = {
    "rule_template_hits":          (lambda v: v >= 1, "Contains {v:.0f} phrase(s) commonly seen in AI-generated reviews."),
    "rule_buyer_persona_phrases":  (lambda v: v >= 1, "Contains a 'buyer persona' phrase (e.g. 'as a busy parent') typical of LLM prompts."),
    "rule_starts_with_rating":     (lambda v: v >= 1, "Opens with a rating-style header ('5/5 stars', '★★★★★')."),
    "rule_markdown_artifacts":     (lambda v: v >= 1, "Contains leftover markdown formatting from LLM output."),
    "rule_pros_cons_structure":    (lambda v: v >= 1, "Uses formal Pros/Cons section structure — unusual for casual reviews."),
    "rule_emoji_count":            (lambda v: v >= 1, "Uses emojis in a pattern typical of recent LLM outputs."),
    "rule_near_duplicate":         (lambda v: v >= 1, "Closely matches another review in this dataset (potential template reuse)."),
    "rule_avg_paragraph_perfection": (lambda v: v >= 0.6, "Paragraph lengths are unusually uniform — characteristic of generated text."),
    "rule_superlative_density":    (lambda v: v >= 0.01, "Clusters of superlatives close together — typical AI praise pattern."),
    "rule_sentiment_rating_mismatch": (lambda v: v >= 1, "Sentiment of the text doesn't match the star rating."),
}


def _cache():
    conn = sqlite3.connect(CACHE_DB)
    conn.execute("CREATE TABLE IF NOT EXISTS shap_cache "
                 "(hash TEXT PRIMARY KEY, payload TEXT)")
    return conn

def _hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


@dataclass
class Reason:
    category: str
    text: str
    weight: float

@dataclass
class Span:
    start: int
    end: int
    text: str
    score: float

@dataclass
class Explanation:
    ai_probability: float
    verdict: str
    top_reasons: list[Reason]
    highlighted_spans: list[Span]

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


class Explainer:
    def __init__(self):
        print(f"Loading models on {DEVICE}...")

        self.tok = AutoTokenizer.from_pretrained(MODEL_DIR)
        self.bert = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(DEVICE).eval()

        self.gpt2_tok = GPT2TokenizerFast.from_pretrained("gpt2")
        self.gpt2 = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE).eval()
        styl.gpt2_tok, styl.gpt2 = self.gpt2_tok, self.gpt2
        styl.DEVICE = DEVICE

        self.xgb = xgb.XGBClassifier()
        self.xgb.load_model(DATA / "fusion_model.json")
        meta = json.loads((DATA / "fusion_metadata.json").read_text())
        self.feat_names: list[str] = meta["features"]
        self.threshold: float = meta["best_threshold"]
        self._booster = self.xgb.get_booster()

        from transformers import pipeline
        self.bert_pipe = pipeline("text-classification", model=self.bert,
                                   tokenizer=self.tok,
                                   device=0 if DEVICE == "cuda" else -1,
                                   top_k=None)
        self.text_explainer = shap.Explainer(self.bert_pipe)

        print("Ready.")

    def _features(self, text: str, rating: Optional[float]) -> dict:
        feats = styl.extract_features(text)
        feats.update(rules.per_review_rules(text, rating))
        feats["rule_near_duplicate"] = 0.0
        feats["rule_max_jaccard"] = 0.0
        feats["distilbert_prob"] = self._bert_prob(text)
        return feats

    @torch.no_grad()
    def _bert_prob(self, text: str) -> float:
        enc = self.tok(text, return_tensors="pt", truncation=True, max_length=256).to(DEVICE)
        logits = self.bert(**enc).logits
        return float(torch.softmax(logits, dim=-1)[0, 1].item())

    def _token_spans(self, text: str) -> list[Span]:
        h = _hash(text)
        conn = sqlite3.connect(CACHE_DB, check_same_thread=False)
        conn.execute("CREATE TABLE IF NOT EXISTS shap_cache (hash TEXT PRIMARY KEY, payload TEXT)")
        cur = conn.execute("SELECT payload FROM shap_cache WHERE hash=?", (h,))

        row = cur.fetchone()
        if row:
            return [Span(**s) for s in json.loads(row[0])]

        sv = self.text_explainer([text])
        labels = list(sv.output_names) if hasattr(sv, "output_names") else []
        try:
            ai_idx = labels.index("ai")
        except (ValueError, AttributeError):
            ai_idx = 1
        values = sv.values[0]
        ai_vals = values[:, ai_idx] if values.ndim == 2 else values
        tokens = sv.data[0]

        spans = []
        cursor = 0
        for tok, score in zip(tokens, ai_vals):
            if not tok or not tok.strip():
                continue
            idx = text.lower().find(tok.lower().lstrip("##").strip(), cursor)
            if idx == -1:
                continue
            spans.append(Span(start=idx, end=idx + len(tok),
                               text=text[idx:idx + len(tok)], score=float(score)))
            cursor = idx + len(tok)

        # Fix 2: keep only meaningful word-like tokens — drop punctuation,
        # short fragments, and anything that's mostly non-letters.
        def _is_meaningful(s: Span) -> bool:
            t = s.text.strip()
            if len(t) < MIN_TOKEN_LEN:
                return False
            letters = sum(1 for c in t if c.isalpha())
            return letters >= max(MIN_TOKEN_LEN, int(0.6 * len(t)))

        spans = [s for s in spans if s.score > 0 and _is_meaningful(s)]
        spans = sorted(spans, key=lambda s: -s.score)[:TOKEN_TOPK]

        conn.execute("INSERT OR REPLACE INTO shap_cache VALUES (?,?)",
              (h, json.dumps([asdict(s) for s in spans])))
        conn.commit()
        conn.close()
        return spans

    def explain(self, text: str, rating: Optional[float] = None) -> Explanation:
        feats = self._features(text, rating)
        x = np.array([[feats[f] for f in self.feat_names]])
        prob = float(self.xgb.predict_proba(x)[0, 1])

        if prob >= self.threshold:        verdict = "likely_ai"
        elif prob >= 0.35:                verdict = "uncertain"
        else:                              verdict = "likely_human"

        import xgboost as _xgb
        reasons: list[Reason] = []

        # Fix 1: only attribute reasons when verdict supports them.
        if verdict != "likely_human":
            shap_vals = self._booster.predict(_xgb.DMatrix(x), pred_contribs=True)[0, :-1]
            for fname, contrib in sorted(zip(self.feat_names, shap_vals),
                                          key=lambda kv: -abs(kv[1])):
                if contrib <= 0:
                    continue
                v = feats[fname]

                if fname.startswith("rule_") and fname in RULE_TEMPLATES:
                    cond, tmpl = RULE_TEMPLATES[fname]
                    if cond(v):
                        reasons.append(Reason("structural", tmpl.format(v=v),
                                                float(contrib)))
                    continue

                if fname in TEMPLATES:
                    sign, tmpl = TEMPLATES[fname]
                    norm = HUMAN_NORMS.get(fname, 0.0)
                    if sign == -1 and v >= norm * 0.9 and norm > 0:  continue
                    if sign == +1 and norm > 0 and v <= norm * 1.1:  continue
                    cat = "statistical" if "perplex" in fname or "burst" in fname else "linguistic"
                    reasons.append(Reason(cat, tmpl.format(v=v, n=norm),
                                            float(contrib)))
                if len(reasons) >= 5:
                    break

        if not reasons and verdict == "likely_ai":
            reasons.append(Reason("statistical",
                                    "Combined statistical signals match patterns "
                                    "common in AI-generated reviews.", float(prob)))

        spans: list[Span] = []
        if prob >= EXPLAIN_THRESHOLD:
            try:
                spans = self._token_spans(text)
            except Exception as e:
                print(f"[token SHAP failed: {e}]")

        return Explanation(ai_probability=prob, verdict=verdict,
                            top_reasons=reasons, highlighted_spans=spans)


if __name__ == "__main__":
    samples = [
        ("5/5 stars! As a busy parent, I was blown away by the sleek design and "
         "intuitive features of this product. It exceeded my expectations and "
         "I highly recommend it to anyone looking for a reliable solution.", 5.0),
        ("Got this last Tuesday. The latch is a bit stiff but works fine after "
         "a week. My cat ignored it for two days then finally claimed it. "
         "Decent for the price.", 4.0),
        ("broken on arrival. seller refunded fast tho", 1.0),
        ("I'm thoroughly impressed with the Office Products \"Ultimate Notebook "
         "Set\" I recently purchased. The set includes a high-quality notebook, "
         "smooth pens, and elegant bookmarks — truly a must-have for any "
         "professional.", 5.0),
        ("Bought it for my dad's birthday. He's been using it every morning for "
         "coffee and says the handle gets a little warm but he likes the size. "
         "Came in a dented box but the mug itself was fine.", 4.0),
    ]

    ex = Explainer()
    for i, (text, rating) in enumerate(samples, 1):
        print("\n" + "=" * 78)
        print(f"SAMPLE {i}  rating={rating}")
        print("=" * 78)
        print(text)
        result = ex.explain(text, rating)
        print(f"\nVerdict: {result.verdict}  (p={result.ai_probability:.3f})")
        print("Top reasons:")
        for r in result.top_reasons:
            print(f"  [{r.category}] {r.text}")
        if result.highlighted_spans:
            print("Highlights:")
            for s in result.highlighted_spans:
                print(f"  '{s.text}' (score={s.score:.3f})")

