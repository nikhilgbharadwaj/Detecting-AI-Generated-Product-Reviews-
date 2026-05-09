"""Phase 4 — Step 2: Streamlit UI.

Run with:
    streamlit run 12_app.py

The user pastes a review, picks a star rating, hits Analyze, and gets:
  1. Verdict badge (Likely AI / Uncertain / Likely Human) + confidence
  2. Plain-English explanation from gemma2:2b
  3. Highlighted phrases (for AI verdicts)
  4. Collapsible technical breakdown with all SHAP-ranked reasons

Loads the Explainer once via @st.cache_resource so model loading happens
on first request only, not on every interaction.
"""
import importlib.util, sys
from pathlib import Path

import streamlit as st

# ---------------------------------------------------------------------------
# Page config — must be the first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Review Detector",
    page_icon="🔍",
    layout="centered",
)


# ---------------------------------------------------------------------------
# Lazy module loading (so the heavy imports happen inside cached resources)
# ---------------------------------------------------------------------------
def _import(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); sys.modules[name] = mod
    spec.loader.exec_module(mod); return mod


@st.cache_resource(show_spinner="Loading models (DistilBERT, GPT-2, XGBoost, Gemma 2B). First time only...")
def get_explainer():
    expl_mod = _import("expl", "10_explain.py")
    return expl_mod.Explainer()


@st.cache_resource(show_spinner=False)
def get_rephraser():
    return _import("rephraser", "11_llm_rephrase.py")


# ---------------------------------------------------------------------------
# Highlight rendering
# ---------------------------------------------------------------------------
def render_highlighted(text: str, spans) -> str:
    """Inserts <mark> tags around span ranges in the text. Higher score = darker."""
    if not spans:
        return text

    # Normalize scores to [0.3, 1.0] for opacity scaling
    scores = [s.score for s in spans]
    max_s = max(scores) if scores else 1.0
    intervals = []
    for s in spans:
        opacity = 0.3 + 0.7 * (s.score / max_s if max_s > 0 else 0)
        intervals.append((s.start, s.end, opacity))

    # Merge / sort, build output
    intervals.sort()
    out, cursor = [], 0
    for start, end, opacity in intervals:
        if start < cursor:   # overlaps prior — skip
            continue
        out.append(text[cursor:start])
        color = f"rgba(255, 99, 99, {opacity:.2f})"
        out.append(f'<mark style="background-color:{color}; padding:1px 2px; border-radius:3px;">{text[start:end]}</mark>')
        cursor = end
    out.append(text[cursor:])
    return "".join(out)


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.title("🔍 AI Review Detector")
st.caption("Hybrid system: DistilBERT + stylometric features + rule-based heuristics, fused with XGBoost. Plain-English explanations by Gemma 2B.")

with st.sidebar:
    st.header("How it works")
    st.markdown("""
This tool combines four signals to detect AI-generated product reviews:

1. **DistilBERT** fine-tuned on ~45k labeled reviews
2. **Stylometric features** — perplexity, burstiness, vocabulary diversity (16 features)
3. **Rule-based detectors** — template phrases, persona leakage, formatting (11 features)
4. **XGBoost fusion layer** combines all 28 signals

Explanations are deterministic — they come from SHAP feature attributions, not from an LLM. The LLM only rephrases the technical findings into plain English.

**Test set performance:**
- Accuracy: 97.3%
- F1: 0.975
- False positive rate: 3.3%
- Held-out generator recall: 95.5%
    """)
    st.divider()
    st.caption("Master's project · NLP · 2026")


review_text = st.text_area(
    "Paste a product review:",
    height=180,
    placeholder="Copy and paste any product review from Amazon, Flipkart, or any e-commerce site...",
)

col_a, col_b = st.columns([1, 3])
with col_a:
    rating = st.selectbox("Star rating", [5, 4, 3, 2, 1], index=0)
with col_b:
    st.write("")  # spacer
    analyze_clicked = st.button("Analyze review", type="primary", use_container_width=True)


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------
if analyze_clicked:
    if not review_text or len(review_text.strip()) < 5:
        st.warning("Please paste a review with at least a few words.")
        st.stop()

    explainer = get_explainer()
    rephraser = get_rephraser()

    with st.spinner("Analyzing..."):
        result = explainer.explain(review_text.strip(), float(rating))
        plain_summary = rephraser.rephrase(
            result.verdict, result.ai_probability, result.top_reasons
        )

    # ------------------ Verdict banner ------------------
    pct = int(round(result.ai_probability * 100))
    if result.verdict == "likely_ai":
        st.error(f"### 🤖 Likely AI-generated  ·  {pct}% confidence")
    elif result.verdict == "uncertain":
        st.warning(f"### ❓ Uncertain  ·  {pct}% AI probability")
    else:
        st.success(f"### 👤 Likely human-written  ·  {100 - pct}% confidence")

    st.progress(result.ai_probability,
                text=f"AI probability: {result.ai_probability:.3f}")

    # ------------------ Plain-English explanation ------------------
    st.markdown("#### Why")
    st.markdown(f"> {plain_summary}")

    # ------------------ Highlights ------------------
    if result.highlighted_spans:
        st.markdown("#### Suspicious phrases")
        st.caption("Phrases the model found most indicative of AI generation. Darker red = stronger signal.")
        highlighted = render_highlighted(review_text.strip(), result.highlighted_spans)
        st.markdown(
            f'<div style="line-height:1.6; padding:12px; background:#1e1e1e; border-radius:6px; '
            f'border:1px solid #333;">{highlighted}</div>',
            unsafe_allow_html=True,
        )

    # ------------------ Technical breakdown ------------------
    if result.top_reasons:
        with st.expander("Show technical details", expanded=False):
            st.markdown("**Reasons ranked by SHAP contribution to the AI verdict:**")
            for r in result.top_reasons:
                badge_color = {
                    "statistical": "🔵",
                    "linguistic": "🟢",
                    "structural": "🟡",
                }.get(r.category, "⚪")
                st.markdown(f"{badge_color} *{r.category}* · {r.text}")
            st.divider()
            st.markdown(
                f"**Final probability:** {result.ai_probability:.4f}  ·  "
                f"**Decision threshold:** {explainer.threshold:.2f}  ·  "
                f"**Verdict:** `{result.verdict}`"
            )


# ---------------------------------------------------------------------------
# Footer with example reviews
# ---------------------------------------------------------------------------
st.divider()
st.markdown("##### Try these examples")

examples = {
    "Obvious AI 🤖": (
        "5/5 stars! As a busy parent, I was blown away by the sleek design and "
        "intuitive features of this product. It exceeded my expectations and I "
        "highly recommend it to anyone looking for a reliable solution.", 5,
    ),
    "Obvious human 👤": (
        "Got this last Tuesday. The latch is a bit stiff but works fine after a "
        "week. My cat ignored it for two days then finally claimed it. Decent "
        "for the price.", 4,
    ),
    "Short human 👤": (
        "broken on arrival. seller refunded fast tho", 1,
    ),
    "Subtle AI 🤖": (
        "I'm thoroughly impressed with the Office Products \"Ultimate Notebook "
        "Set\" I recently purchased. The set includes a high-quality notebook, "
        "smooth pens, and elegant bookmarks — truly a must-have for any "
        "professional.", 5,
    ),
}

cols = st.columns(len(examples))
for col, (label, (text, r)) in zip(cols, examples.items()):
    with col:
        if st.button(label, use_container_width=True):
            st.session_state["_example_text"] = text
            st.session_state["_example_rating"] = r
            st.rerun()

# Auto-fill from example button click
if "_example_text" in st.session_state:
    st.info(f"**Example loaded** — copy this into the box above:\n\n{st.session_state['_example_text']}\n\n*(rating: {st.session_state['_example_rating']})*")
    if st.button("Clear example"):
        del st.session_state["_example_text"]
        del st.session_state["_example_rating"]
        st.rerun()
