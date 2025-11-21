# app.py
# pip install sentence-transformers numpy google-generativeai streamlit

import os
import streamlit as st
import google.generativeai as genai
import numpy as np
import time
import traceback
import importlib

st.set_page_config(page_title="Small RAG — Lazy Load (KB + Gemini fallback)", layout="wide")

# Local uploaded file path (available in this session)
# Use this path (it will be transformed to a URL by your deployment/tooling if needed)
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

# ---- RAG function with Gemini direct fallback ----
def rag_answer(query, k=2, similarity_threshold=0.40):
    """
    1) Retrieve from KB.
    2) If top similarity >= threshold -> build facts from KB and ask Gemini to answer using only them.
    3) If top similarity < threshold -> call Gemini directly (no KB facts) and let Gemini answer from its knowledge.
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
        facts = "\n".join([f"- {text}" for text, score in retrieved])
        prompt = f"""
Answer the question using ONLY these facts (do NOT add external facts you can't verify):

{facts}

Question: {query}
Answer briefly:
"""
        # If no API key, return early
        if not API_KEY:
            return prompt, "No API key configured - cannot call Gemini.", retrieved, used_mode

        try:
            llm = genai.GenerativeModel("gemini-2.0-flash")
            resp = llm.generate_content(prompt)
            # robust extraction
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
            return prompt, text.strip(), retrieved, used_mode
        except Exception as e:
            return prompt, f"Error calling Gemini with KB facts: {e}\n\n{traceback.format_exc()}", retrieved, used_mode

    # Otherwise, fallback: let Gemini answer freely (no KB facts)
    used_mode = "gemini_fallback"
    prompt = f"""
You are a helpful assistant. Answer the question below briefly and clearly using your knowledge.
If you are unsure, say you are unsure rather than inventing facts.

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
        return prompt, text.strip(), retrieved, used_mode
    except Exception as e:
        return prompt, f"Error calling Gemini (fallback): {e}\n\n{traceback.format_exc()}", retrieved, used_mode

# ---- UI ----
st.title("Small RAG demo — Streamlit + Gemini (KB first, Gemini fallback)")
st.markdown("This app searches a small in-memory KB first. If the top KB match is below the similarity threshold, Gemini will answer directly using its internal knowledge (no live web search).")

st.markdown(f"**Uploaded file path available in app:** `{FILE_URL}`")

query = st.text_input("Enter question", value="what is india?")
k = st.slider("How many KB documents to retrieve (k)", min_value=1, max_value=5, value=2)
similarity_threshold = st.slider("Similarity threshold (if top KB match >= threshold → use KB)", min_value=0.0, max_value=1.0, value=0.40, step=0.01)

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
    for doc, score in retrieved:
        st.write
