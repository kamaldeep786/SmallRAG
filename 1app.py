# app.py
# pip install sentence-transformers numpy google-generativeai streamlit

import os
import streamlit as st
import google.generativeai as genai
import numpy as np
import time
import traceback
import importlib
from html import escape

st.set_page_config(page_title="Small RAG — Streamlit + Gemini (clean answers)", layout="wide")

# Local uploaded file path (available in this session)
# We'll expose this path in the UI (deployment will handle making it a URL)
FILE_URL = "/mnt/data/edf991c1-66d2-49aa-9bfc-f30bdcf212c6.png"

# ---- Configuration / secrets ----
API_KEY = st.secrets.get("GENAI_API_KEY", os.environ.get("GENAI_API_KEY"))
if not API_KEY:
    st.warning("No Gemini API key found. Set GENAI_API_KEY in Streamlit Secrets or environment.")
else:
    try:
        genai.configure(api_key=API_KEY)
    except Exception as e:
        st.error(f"Failed to configure Gemini SDK: {e}")

# ---- session state for lazy loading ----
if "models_loaded" not in st.session_state:
    st.session_state["models_loaded"] = False
if "embedder" not in st.session_state:
    st.session_state["embedder"] = None
if "documents" not in st.session_state:
    st.session_state["documents"] = None
if "doc_embs" not in st.session_state:
    st.session_state["doc_embs"] = None
if "last_error" not in st.session_state:
    st.session_state["last_error"] = None

# ---- function to lazy load heavy deps ----
def lazy_load_models():
    """
    Load SentenceTransformer and prepare the small KB.
    This is called only when user clicks Ask.
    """
    try:
        SentenceTransformer = importlib.import_module("sentence_transformers").SentenceTransformer
    except Exception as e:
        st.session_state["last_error"] = f"Failed to import sentence_transformers: {e}\n\n{traceback.format_exc()}"
        raise

    try:
        embedder = SentenceTransformer("all-MiniLM-L6-v2")
    except Exception as e:
        st.session_state["last_error"] = f"Failed to load model all-MiniLM-L6-v2: {e}\n\n{traceback.format_exc()}"
        raise

    # sample in-memory KB: keep items reasonably short or treat them as long documents shown in an expander
    documents = [
        "AI automates repetitive tasks and saves time.",
        "AI works continuously without breaks.",
        "AI quickly detects patterns in large datasets.",
        "AI improves decision-making with data insights.",
        "AI personalizes user experiences across apps.",
        "AI reduces human error in routine processes.",
        "AI scales services to millions of users.",
        "AI speeds up support with smart chatbots.",
        "AI lowers operational costs via efficiency.",
        "AI enables rapid prototyping and experimentation."
    ]
    doc_embs = embedder.encode(documents, normalize_embeddings=True)
    # store in session state
    st.session_state["embedder"] = embedder
    st.session_state["documents"] = documents
    st.session_state["doc_embs"] = doc_embs
    st.session_state["models_loaded"] = True
    st.session_state["last_error"] = None

# ---- retrieval function (uses session state) ----
def retrieve(query, k=3):
    embedder = st.session_state.get("embedder")
    documents = st.session_state.get("documents")
    doc_embs = st.session_state.get("doc_embs")
    if not embedder or documents is None or doc_embs is None:
        raise RuntimeError("Models not loaded. Call lazy_load_models() first.")
    qvec = embedder.encode([query], normalize_embeddings=True)[0]
    sims = doc_embs @ qvec
    top_idx = np.argsort(sims)[-k:][::-1]
    return [(documents[i], float(sims[i])) for i in top_idx]

# ---- helper: shorten displayed snippet and provide expander for full content ----
def display_doc_snippet(idx, text, score, max_chars=320):
    snippet = text if len(text) <= max_chars else text[:max_chars].rstrip() + "..."
    st.markdown(f"- ({score:.4f}) {escape(snippet)}", unsafe_allow_html=True)
    if len(text) > max_chars:
        with st.expander("Show full document"):
            # show raw text block (monospace for clarity)
            st.code(text, language="text")

# ---- RAG function with Gemini direct fallback (clean prompts + concise answers) ----
def rag_answer(query, k=2, similarity_threshold=0.40, answer_sentences=2):
    """
    1) Retrieve from KB.
    2) If top similarity >= threshold -> build facts from KB and ask Gemini to answer using only them.
    3) If top similarity < threshold -> call Gemini directly (no KB facts) and let Gemini answer from its knowledge.
    The prompts force concise answers (answer_sentences).
    Returns: prompt_sent, answer_text, retrieved_list, mode_used
             mode_used: "kb" | "gemini_fallback"
    """
    try:
        retrieved = retrieve(query, k=k)
    except Exception as e:
        raise RuntimeError(f"Retrieval failed: {e}\n\n{traceback.format_exc()}")

    top_score = retrieved[0][1] if retrieved else 0.0
    used_mode = "kb"

    # If KB match is good enough, use KB facts
    if top_score >= similarity_threshold:
        # keep facts short and numbered; avoid sending huge docs to LLM
        facts_lines = []
        for i, (text, score) in enumerate(retrieved):
            short = text if len(text) <= 800 else text[:800] + "..."
            facts_lines.append(f"{i+1}. {short}")
        facts = "\n".join(facts_lines)

        prompt = f"""You are an assistant that must answer using ONLY the provided facts — do NOT add unsupported information.
Keep the answer extremely concise: maximum {answer_sentences} short sentence(s).

FACTS:
{facts}

QUESTION:
{query}

Answer briefly (max {answer_sentences} sentences). If the facts don't contain the answer, say "I don't know" or provide a very short best-effort answer and state uncertainty.
"""
        if not API_KEY:
            return prompt, "No API key configured - cannot call Gemini.", retrieved, used_mode

        try:
            llm = genai.GenerativeModel("gemini-2.0-flash")
            resp = llm.generate_content(prompt)
            text = getattr(resp, "text", None) or (resp.output_text if hasattr(resp, "output_text") else None)
            if not text and hasattr(resp, "candidates"):
                collected = []
                for c in resp.candidates:
                    if hasattr(c, "content") and getattr(c.content, "parts", None):
                        for part in c.content.parts:
                            t = getattr(part, "text", None)
                            if t:
                                collected.append(t)
                text = "\n".join(collected)
            if not text:
                text = str(resp)
            # post-process: collapse newlines and trim to a few sentences
            answer = " ".join([s.strip() for s in text.strip().splitlines() if s.strip()])
            # limit sentences roughly
            sentences = answer.split(". ")
            answer = ". ".join(sentences[:answer_sentences]).strip()
            if not answer.endswith("."):
                answer = answer + ("" if answer.endswith("?") else ".")
            return prompt, answer, retrieved, used_mode
        except Exception as e:
            return prompt, f"Error calling Gemini with KB facts: {e}\n\n{traceback.format_exc()}", retrieved, used_mode

    # Otherwise, fallback: let Gemini answer freely (no KB facts)
    used_mode = "gemini_fallback"
    prompt = f"""You are a helpful assistant. Answer the question below briefly and clearly using your knowledge.
Answer in maximum {answer_sentences} short sentence(s). If you're unsure, say 'I don't know' instead of inventing facts.

Question: {query}
Answer briefly:
"""
    if not API_KEY:
        return prompt, "No API key configured - cannot call Gemini.", retrieved, used_mode

    try:
        llm = genai.GenerativeModel("gemini-2.0-flash")
        resp = llm.generate_content(prompt)
        text = getattr(resp, "text", None) or (resp.output_text if hasattr(resp, "output_text") else None)
        if not text and hasattr(resp, "candidates"):
            collected = []
            for c in resp.candidates:
                if hasattr(c, "content") and getattr(c.content, "parts", None):
                    for part in c.content.parts:
                        t = getattr(part, "text", None)
                        if t:
                            collected.append(t)
            text = "\n".join(collected)
        if not text:
            text = str(resp)
        answer = " ".join([s.strip() for s in text.strip().splitlines() if s.strip()])
        sentences = answer.split(". ")
        answer = ". ".join(sentences[:answer_sentences]).strip()
        if not answer.endswith("."):
            answer = answer + ("" if answer.endswith("?") else ".")
        return prompt, answer, retrieved, used_mode
    except Exception as e:
        return prompt, f"Error calling Gemini (fallback): {e}\n\n{traceback.format_exc()}", retrieved, used_mode

# ---- UI ----
st.title("Small RAG demo — Streamlit + Gemini (KB first, Gemini fallback)")
st.markdown("This app searches a small in-memory KB first. If the top KB match is below the similarity threshold, Gemini will answer directly using its internal knowledge (no live web search).")

st.markdown(f"**Uploaded file path available in app:** `{FILE_URL}`")

query = st.text_input("Enter question", value="What is AI?")
k = st.slider("How many KB documents to retrieve (k)", min_value=1, max_value=5, value=2)
similarity_threshold = st.slider("Similarity threshold (if top KB match >= threshold → use KB)", min_value=0.0, max_value=1.0, value=0.40, step=0.01)
answer_sentences = st.slider("Max sentences in model answer", min_value=1, max_value=4, value=2)

show_debug = st.sidebar.checkbox("Show debug logs", value=False)

if st.button("Ask"):
    # Lazy-load models if not loaded
    if not st.session_state["models_loaded"]:
        with st.spinner("Loading embedder model (this may take time on first run)..."):
            try:
                lazy_load_models()
            except Exception as e:
                st.error("Model load failed. See debug info below.")
                st.session_state["models_loaded"] = False
                if show_debug:
                    st.text(st.session_state.get("last_error"))
                st.stop()

    # Now perform RAG / gemini fallback
    try:
        with st.spinner("Retrieving and calling model..."):
            t0 = time.time()
            prompt, answer, retrieved, mode_used = rag_answer(query, k=k, similarity_threshold=similarity_threshold)
            t1 = time.time()
    except Exception as e:
        st.error("Processing failed. See debug info below.")
        if show_debug:
            st.text(traceback.format_exc())
        st.stop()

    if mode_used == "kb":
        st.info("✔ Answered from Knowledge Base (KB facts were used).")
    else:
        st.success("✔ Gemini answered directly (KB had no confident match).")

    st.subheader("Timing")
    st.write(f"Total time: {t1 - t0:.2f}s")

    st.subheader("Retrieved facts (KB)")
    # display snippets cleanly with expanders for full docs
    for i, (doc, score) in enumerate(retrieved):
        display_doc_snippet(i, doc, score, max_chars=300)

    st.subheader("Prompt sent to Gemini")
    with st.expander("Show prompt (for debugging)"):
        st.code(prompt, language="text")

    st.subheader("Model answer")
    # show concise answer plainly (not as code)
    st.write(answer)

# Debug section (visible if checkbox)
if show_debug:
    st.markdown("### Debug info")
    st.write("Models loaded:", st.session_state["models_loaded"])
    st.write("API key present:", bool(API_KEY))
    try:
        import sys, pkgutil
        st.write("Python version:", sys.version)
        # Show a small list of installed top-level packages to identify install issues
        pkgs = sorted([m.name for m in pkgutil.iter_modules()][:80])
        st.write("Installed packages (sample):", pkgs[:40])
    except Exception as e:
        st.write("Could not enumerate installed packages:", e)

st.sidebar.header("Notes")
st.sidebar.write("If model loading fails with 'Killed' or OOM, your host may not have enough memory. Consider using a lighter embedder or precomputing embeddings.")
st.sidebar.write("Do NOT commit your API key to GitHub. Use Streamlit Secrets or environment variables.")
