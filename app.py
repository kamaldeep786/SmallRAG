# app.py
# Required libraries for this app:
# - sentence-transformers : to create embeddings for documents and queries
# - numpy                 : for vector math (dot products, etc.)
# - google-generativeai   : to call Gemini models
# - streamlit             : to build the web UI

import os
import streamlit as st
import google.generativeai as genai
import numpy as np
import time
import traceback
import importlib
from html import escape

# Basic Streamlit page configuration (title + wide layout)
st.set_page_config(page_title="Small RAG — Streamlit + Gemini (clean answers)", layout="wide")

# Local uploaded file path (available during this session only).
# This is just an example to show how you might expose a file path in the UI.
FILE_URL = "/mnt/data/edf991c1-66d2-49aa-9bfc-f30bdcf212c6.png"

# ---- Configuration / secrets ----
# First, try to read API key from Streamlit Secrets.
# If not found there, fall back to environment variable GENAI_API_KEY.
API_KEY = st.secrets.get("GENAI_API_KEY", os.environ.get("GENAI_API_KEY"))

# If no API key is set, show a warning in the UI.
if not API_KEY:
    st.warning("No Gemini API key found. Set GENAI_API_KEY in Streamlit Secrets or environment.")
else:
    # Configure the Gemini SDK with the key.
    try:
        genai.configure(api_key=API_KEY)
    except Exception as e:
        # If SDK configuration fails, show error in the UI.
        st.error(f"Failed to configure Gemini SDK: {e}")

# ---- session state for lazy loading ----
# Session state is used so that models and data are loaded only once,
# and reused across multiple user interactions.

if "models_loaded" not in st.session_state:
    st.session_state["models_loaded"] = False  # flag to indicate if embeddings model is loaded

if "embedder" not in st.session_state:
    st.session_state["embedder"] = None       # will hold SentenceTransformer instance

if "documents" not in st.session_state:
    st.session_state["documents"] = None      # list of KB documents (strings)

if "doc_embs" not in st.session_state:
    st.session_state["doc_embs"] = None       # numpy array of document embeddings

if "last_error" not in st.session_state:
    st.session_state["last_error"] = None     # store last error for debug display

# ---- function to lazy load heavy deps ----
def lazy_load_models():
    """
    Load SentenceTransformer and prepare the small KB.
    Called only when user clicks Ask and models aren't loaded yet.
    This avoids slow imports at app startup.
    """
    try:
        # Dynamically import SentenceTransformer (avoids importing on initial app load)
        SentenceTransformer = importlib.import_module("sentence_transformers").SentenceTransformer
    except Exception as e:
        # If import fails, store full traceback in session state and re-raise.
        st.session_state["last_error"] = f"Failed to import sentence_transformers: {e}\n\n{traceback.format_exc()}"
        raise

    try:
        # Load a small, general-purpose embedding model.
        embedder = SentenceTransformer("all-MiniLM-L6-v2")
    except Exception as e:
        # If model loading fails, record error and re-raise.
        st.session_state["last_error"] = f"Failed to load model all-MiniLM-L6-v2: {e}\n\n{traceback.format_exc()}"
        raise

    # sample in-memory KB: a simple list of strings (facts about AI).
    # In a real app, you could load from files, database, etc.
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

    # Compute embeddings for all KB documents and normalize them.
    doc_embs = embedder.encode(documents, normalize_embeddings=True)

    # Store model + docs + embeddings in session state for reuse.
    st.session_state["embedder"] = embedder
    st.session_state["documents"] = documents
    st.session_state["doc_embs"] = doc_embs
    st.session_state["models_loaded"] = True
    st.session_state["last_error"] = None  # clear any previous error

# ---- retrieval function (uses session state) ----
def retrieve(query, k=3):
    """
    Given a query string, return top-k most similar documents from KB.
    Uses cosine similarity via dot product (since vectors are normalized).
    """
    # Read embedder and data from session state.
    embedder = st.session_state.get("embedder")
    documents = st.session_state.get("documents")
    doc_embs = st.session_state.get("doc_embs")

    # If something is missing, ask caller to load models first.
    if not embedder or documents is None or doc_embs is None:
        raise RuntimeError("Models not loaded. Call lazy_load_models() first.")

    # Encode the query into an embedding (1D vector).
    qvec = embedder.encode([query], normalize_embeddings=True)[0]

    # Similarity = dot product between document embeddings and query embedding.
    sims = doc_embs @ qvec

    # Get indices of top-k highest similarity scores (descending order).
    top_idx = np.argsort(sims)[-k:][::-1]

    # Return list of (document_text, similarity_score) pairs.
    return [(documents[i], float(sims[i])) for i in top_idx]

# ---- helper: shorten displayed snippet and provide expander for full content ----
def display_doc_snippet(idx, text, score, max_chars=320):
    """
    Show a short snippet of a document plus an optional expander
    to reveal the full text.
    """
    # Truncate long text to max_chars for clean display.
    snippet = text if len(text) <= max_chars else text[:max_chars].rstrip() + "..."
    # Show similarity score along with escaped snippet (avoid HTML injection).
    st.markdown(f"- ({score:.4f}) {escape(snippet)}", unsafe_allow_html=True)

    # If document is longer than snippet, let user expand to see full text.
    if len(text) > max_chars:
        with st.expander("Show full document"):
            # Show full text in a code-style block (monospace).
            st.code(text, language="text")

# ---- RAG function with Gemini direct fallback (clean prompts + concise answers) ----
def rag_answer(query, k=2, similarity_threshold=0.40, answer_sentences=2):
    """
    Main RAG pipeline:

    1) Retrieve top-k documents from KB.
    2) If top similarity >= threshold:
         - build a 'FACTS' section from KB documents
         - call Gemini and instruct it to answer using ONLY those facts.
    3) If top similarity < threshold:
         - call Gemini directly without KB (fallback mode).

    The prompts enforce concise answers (max answer_sentences).
    Returns:
        prompt_sent, answer_text, retrieved_list, mode_used
        where mode_used is "kb" or "gemini_fallback".
    """
    try:
        # Step 1: retrieve relevant docs from KB using embeddings.
        retrieved = retrieve(query, k=k)
    except Exception as e:
        # If retrieval fails, wrap exception with full traceback.
        raise RuntimeError(f"Retrieval failed: {e}\n\n{traceback.format_exc()}")

    # Highest similarity score among retrieved docs (0.0 if list empty).
    top_score = retrieved[0][1] if retrieved else 0.0
    used_mode = "kb"  # default assumption

    # ---- Case 1: Use KB facts (top_score passes threshold) ----
    if top_score >= similarity_threshold:
        # Build concise, numbered list of facts from retrieved docs.
        facts_lines = []
        for i, (text, score) in enumerate(retrieved):
            # Avoid sending very long text; truncate to 800 chars.
            short = text if len(text) <= 800 else text[:800] + "..."
            facts_lines.append(f"{i+1}. {short}")
        facts = "\n".join(facts_lines)

        # Prompt for Gemini: strict instructions to use ONLY given facts.
        prompt = f"""You are an assistant that must answer using ONLY the provided facts — do NOT add unsupported information.
Keep the answer extremely concise: maximum {answer_sentences} short sentence(s).

FACTS:
{facts}

QUESTION:
{query}

Answer briefly (max {answer_sentences} sentences). If the facts don't contain the answer, say "I don't know" or provide a very short best-effort answer and state uncertainty.
"""

        # If no API key, return prompt and an error message instead of calling Gemini.
        if not API_KEY:
            return prompt, "No API key configured - cannot call Gemini.", retrieved, used_mode

        try:
            # Initialize Gemini model (fast, lightweight model).
            llm = genai.GenerativeModel("gemini-2.0-flash")
            # Call the model with the constructed prompt.
            resp = llm.generate_content(prompt)

            # Try to extract text from response. Different SDK versions expose it differently.
            text = getattr(resp, "text", None) or (resp.output_text if hasattr(resp, "output_text") else None)

            # If text is not directly available, try to collect it from candidates/parts.
            if not text and hasattr(resp, "candidates"):
                collected = []
                for c in resp.candidates:
                    if hasattr(c, "content") and getattr(c.content, "parts", None):
                        for part in c.content.parts:
                            t = getattr(part, "text", None)
                            if t:
                                collected.append(t)
                text = "\n".join(collected)

            # Fallback: if still nothing, just stringify the whole response object.
            if not text:
                text = str(resp)

            # Clean up whitespace and collapse multiple lines into one.
            answer = " ".join([s.strip() for s in text.strip().splitlines() if s.strip()])

            # Rough sentence limiting: split on ". " and keep only first N sentences.
            sentences = answer.split(". ")
            answer = ". ".join(sentences[:answer_sentences]).strip()

            # Ensure answer ends with punctuation (unless it ends with "?").
            if not answer.endswith("."):
                answer = answer + ("" if answer.endswith("?") else ".")

            # Return everything for display (including prompt for debugging).
            return prompt, answer, retrieved, used_mode

        except Exception as e:
            # Any error when calling Gemini is returned as part of the answer for debug.
            return prompt, f"Error calling Gemini with KB facts: {e}\n\n{traceback.format_exc()}", retrieved, used_mode

    # ---- Case 2: Fallback mode (KB not confident enough) ----
    used_mode = "gemini_fallback"

    # Prompt that lets Gemini answer using its own knowledge, still concise.
    prompt = f"""You are a helpful assistant. Answer the question below briefly and clearly using your knowledge.
Answer in maximum {answer_sentences} short sentence(s). If you're unsure, say 'I don't know' instead of inventing facts.

Question: {query}
Answer briefly:
"""

    # If no API key, again return prompt and error instead of calling the model.
    if not API_KEY:
        return prompt, "No API key configured - cannot call Gemini.", retrieved, used_mode

    try:
        # Same model as above, but without KB facts.
        llm = genai.GenerativeModel("gemini-2.0-flash")
        resp = llm.generate_content(prompt)

        # Robust extraction logic for text.
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

        # Clean and sentence-limit as above.
        answer = " ".join([s.strip() for s in text.strip().splitlines() if s.strip()])
        sentences = answer.split(". ")
        answer = ". ".join(sentences[:answer_sentences]).strip()
        if not answer.endswith("."):
            answer = answer + ("" if answer.endswith("?") else ".")

        return prompt, answer, retrieved, used_mode

    except Exception as e:
        # If Gemini fails in fallback mode, return error message.
        return prompt, f"Error calling Gemini (fallback): {e}\n\n{traceback.format_exc()}", retrieved, used_mode

# ---- UI ----
# Main title and description for the Streamlit app.
st.title("Small RAG demo — Streamlit + Gemini (KB first, Gemini fallback)")
st.markdown(
    "This app searches a small in-memory KB first. "
    "If the top KB match is below the similarity threshold, Gemini will answer directly using its internal knowledge (no live web search)."
)

# Show the example uploaded file path (just as an informational line).
st.markdown(f"**Uploaded file path available in app:** `{FILE_URL}`")

# Text input for user's question, with a default example.
query = st.text_input("Enter question", value="What is AI?")

# Slider to choose how many documents (k) to retrieve from KB.
k = st.slider("How many KB documents to retrieve (k)", min_value=1, max_value=5, value=2)

# Slider to adjust similarity threshold controlling when to use KB vs fallback.
similarity_threshold = st.slider(
    "Similarity threshold (if top KB match >= threshold → use KB)",
    min_value=0.0,
    max_value=1.0,
    value=0.40,
    step=0.01
)

# Slider for controlling maximum number of sentences in the model answer.
answer_sentences = st.slider("Max sentences in model answer", min_value=1, max_value=4, value=2)

# Sidebar checkbox: when ON, show extra debug information.
show_debug = st.sidebar.checkbox("Show debug logs", value=False)

# When user clicks the "Ask" button, run the pipeline.
if st.button("Ask"):
    # Step 1: Lazy-load models if not already loaded in this session.
    if not st.session_state["models_loaded"]:
        # Show spinner while loading model (first run can be slow).
        with st.spinner("Loading embedder model (this may take time on first run)..."):
            try:
                lazy_load_models()
            except Exception as e:
                # If model load fails, show error and (optionally) debug logs.
                st.error("Model load failed. See debug info below.")
                st.session_state["models_loaded"] = False
                if show_debug:
                    st.text(st.session_state.get("last_error"))
                # Stop further execution of this callback.
                st.stop()

    # Step 2: Run RAG + Gemini answer.
    try:
        with st.spinner("Retrieving and calling model..."):
            t0 = time.time()  # start timing
            prompt, answer, retrieved, mode_used = rag_answer(
                query,
                k=k,
                similarity_threshold=similarity_threshold
            )
            t1 = time.time()  # end timing
    except Exception as e:
        # Generic processing failure: show error and optional traceback.
        st.error("Processing failed. See debug info below.")
        if show_debug:
            st.text(traceback.format_exc())
        st.stop()

    # Show which mode was used: KB or Gemini fallback.
    if mode_used == "kb":
        st.info("✔ Answered from Knowledge Base (KB facts were used).")
    else:
        st.success("✔ Gemini answered directly (KB had no confident match).")

    # Show timing for entire RAG + model call process.
    st.subheader("Timing")
    st.write(f"Total time: {t1 - t0:.2f}s")

    # Show retrieved KB documents with similarity scores.
    st.subheader("Retrieved facts (KB)")
    for i, (doc, score) in enumerate(retrieved):
        # Use helper to display snippet with optional expander for full text.
        display_doc_snippet(i, doc, score, max_chars=300)

    # Show full prompt sent to Gemini (for debugging / transparency).
    st.subheader("Prompt sent to Gemini")
    with st.expander("Show prompt (for debugging)"):
        st.code(prompt, language="text")

    # Finally, show the model's concise answer.
    st.subheader("Model answer")
    st.write(answer)

# Debug section (only visible if checkbox is enabled).
if show_debug:
    st.markdown("### Debug info")
    st.write("Models loaded:", st.session_state["models_loaded"])
    st.write("API key present:", bool(API_KEY))
    try:
        import sys, pkgutil
        st.write("Python version:", sys.version)
        # Show a small sample of installed packages to help debug environment issues.
        pkgs = sorted([m.name for m in pkgutil.iter_modules()][:80])
        st.write("Installed packages (sample):", pkgs[:40])
    except Exception as e:
        st.write("Could not enumerate installed packages:", e)

# Sidebar notes: extra guidance / warnings for the user.
st.sidebar.header("Notes")
st.sidebar.write(
    "If model loading fails with 'Killed' or OOM, your host may not have enough memory. "
    "Consider using a lighter embedder or precomputing embeddings."
)
st.sidebar.write(
    "Do NOT commit your API key to GitHub. Use Streamlit Secrets or environment variables."
)
