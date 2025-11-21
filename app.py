# app.py
# pip install sentence-transformers numpy google-generativeai streamlit

import os
import streamlit as st
import google.generativeai as genai
import numpy as np
import time
import traceback
import importlib

st.set_page_config(page_title="Small RAG — Lazy Load (KB + Web fallback)", layout="wide")

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
        # import here to avoid import-time failures during app startup
        SentenceTransformer = importlib.import_module("sentence_transformers").SentenceTransformer
    except Exception as e:
        st.session_state["last_error"] = f"Failed to import sentence_transformers: {e}\n\n{traceback.format_exc()}"
        raise

    try:
        embedder = SentenceTransformer("all-MiniLM-L6-v2")
    except Exception as e:
        st.session_state["last_error"] = f"Failed to load model all-MiniLM-L6-v2: {e}\n\n{traceback.format_exc()}"
        raise

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

# ---- web search via Gemini (defensive) ----
def web_search_gemini(query, max_snippets=5):
    if not API_KEY:
        return "No API key configured - cannot perform web search."

    try:
        model = genai.GenerativeModel("gemini-2.0-flash", tools=[{"google_search": {}}])
        response = model.generate_content(
            [
                {
                    "google_search": {
                        "q": query,
                        "num_results": max_snippets
                    }
                }
            ]
        )
        snippets = []
        if hasattr(response, "candidates"):
            for c in response.candidates:
                if hasattr(c, "content") and getattr(c.content, "parts", None):
                    for part in c.content.parts:
                        text = getattr(part, "text", None)
                        if text:
                            snippets.append(text.strip())
                else:
                    snippets.append(str(c))
        else:
            snippets.append(str(response))

        unique_snips = []
        for s in snippets:
            if s not in unique_snips and len(unique_snips) < max_snippets:
                unique_snips.append(s)

        return "\n\n".join(unique_snips).strip() or ("No web snippets found for: " + query)

    except Exception as e:
        return f"Web search failed: {e}\n\n{traceback.format_exc()}"

# ---- RAG function ----
def rag_answer(query, k=2, similarity_threshold=0.40, max_web_snippets=5):
    try:
        retrieved = retrieve(query, k=k)
    except Exception as e:
        raise RuntimeError(f"Retrieval failed: {e}\n\n{traceback.format_exc()}")

    top_score = retrieved[0][1] if retrieved else 0.0
    use_web = top_score < similarity_threshold
    web_snippets = None

    if not use_web:
        facts = "\n".join([f"- {text}" for text, score in retrieved])
    else:
        web_snippets = web_search_gemini(query, max_snippets=max_web_snippets)
        if web_snippets and not web_snippets.lower().startswith("web search failed"):
            facts = "\n".join([f"- {line}" for line in web_snippets.splitlines() if line.strip()][:10])
        else:
            facts = f"- No factual snippets found for query: {query}"

    prompt = f"""
Answer the question using ONLY these facts (do NOT add external info you don't know):

{facts}

Question: {query}
Answer briefly:
"""

    if not API_KEY:
        return prompt, "No API key configured - cannot call Gemini.", retrieved, use_web, web_snippets

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

        answer_text = text.strip() if hasattr(text, "strip") else str(text)
        return prompt, answer_text, retrieved, use_web, web_snippets

    except Exception as e:
        return prompt, f"Error calling Gemini: {e}\n\n{traceback.format_exc()}", retrieved, use_web, web_snippets

# ---- UI ----
st.title("Small RAG demo — Streamlit + Gemini (lazy load)")
st.markdown("App loads heavy ML model only when you click Ask. This helps avoid startup-time failures on hosts with strict time/memory limits.")

query = st.text_input("Enter question", value="what is india?")
k = st.slider("How many KB documents to retrieve (k)", min_value=1, max_value=5, value=2)
similarity_threshold = st.slider("Similarity threshold", min_value=0.0, max_value=1.0, value=0.40, step=0.01)
max_web_snippets = st.number_input("Max web snippets to fetch", min_value=1, max_value=10, value=5)

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

    # Now perform RAG / web fallback
    try:
        with st.spinner("Retrieving and calling model..."):
            t0 = time.time()
            prompt, answer, retrieved, used_web, web_snips = rag_answer(query, k=k, similarity_threshold=similarity_threshold, max_web_snippets=max_web_snippets)
            t1 = time.time()
    except Exception as e:
        st.error("RAG processing failed. See debug info below.")
        if show_debug:
            st.text(traceback.format_exc())
        st.stop()

    if used_web:
        st.success("✔ Used Web Search (KB didn't return a confident match).")
    else:
        st.info("✔ Answered from Knowledge Base (no web search needed).")

    st.subheader("Timing")
    st.write(f"Total time: {t1 - t0:.2f}s")

    st.subheader("Retrieved facts (KB)")
    for doc, score in retrieved:
        st.write(f"- ({score:.4f}) {doc}")

    if used_web:
        st.subheader("Web snippets (first results)")
        st.write(web_snips)

    st.subheader("Prompt sent to Gemini")
    st.code(prompt, language="text")

    st.subheader("Model answer")
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
        pkgs = sorted([m.name for m in pkgutil.iter_modules()][:60])
        st.write("Installed packages (sample):", pkgs[:40])
    except Exception as e:
        st.write("Could not enumerate installed packages:", e)

st.sidebar.header("Notes")
st.sidebar.write("If model loading fails with 'Killed' or OOM, your host may not have enough memory. Consider using a lighter embedder or precomputing embeddings.")
st.sidebar.write("Do NOT commit your API key to GitHub. Use Streamlit Secrets or environment variables.")
