# app.py
# Production-safe Small RAG — Streamlit + Gemini

import os
import streamlit as st
import google.generativeai as genai
import numpy as np
import time
import traceback
import importlib
from html import escape
from functools import lru_cache

st.set_page_config(page_title="Small RAG — Streamlit + Gemini", layout="wide")

MODEL_NAME = "gemini-2.0-flash"

# ---------------- API CONFIG ----------------
API_KEY = st.secrets.get("GENAI_API_KEY", os.environ.get("GENAI_API_KEY"))

if not API_KEY:
    st.warning("No Gemini API key found.")
else:
    genai.configure(api_key=API_KEY)

# ---------------- SESSION STATE ----------------
if "models_loaded" not in st.session_state:
    st.session_state.models_loaded = False
    st.session_state.embedder = None
    st.session_state.documents = None
    st.session_state.doc_embs = None


# ---------------- LOAD EMBEDDINGS ----------------
def lazy_load_models():
    SentenceTransformer = importlib.import_module(
        "sentence_transformers"
    ).SentenceTransformer

    embedder = SentenceTransformer("all-MiniLM-L6-v2")

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

    st.session_state.embedder = embedder
    st.session_state.documents = documents
    st.session_state.doc_embs = doc_embs
    st.session_state.models_loaded = True


# ---------------- RETRIEVAL ----------------
def retrieve(query, k=3):
    embedder = st.session_state.embedder
    documents = st.session_state.documents
    doc_embs = st.session_state.doc_embs

    qvec = embedder.encode([query], normalize_embeddings=True)[0]
    sims = doc_embs @ qvec
    top_idx = np.argsort(sims)[-k:][::-1]

    return [(documents[i], float(sims[i])) for i in top_idx]


# ---------------- SAFE GEMINI CALL ----------------
def call_gemini(prompt):
    try:
        model = genai.GenerativeModel(MODEL_NAME)
        response = model.generate_content(prompt)
        text = getattr(response, "text", None)

        if not text:
            text = str(response)

        return text.strip()

    except Exception as e:
        msg = str(e).lower()

        if "quota" in msg or "429" in msg:
            return "⚠️ Gemini API quota exceeded. Please try later."

        if "not found" in msg:
            return "⚠️ Model not available in your API version."

        return f"⚠️ Gemini error: {str(e)}"


# ---------------- RAG PIPELINE ----------------
@st.cache_data(show_spinner=False)
def rag_answer(query, k, similarity_threshold, answer_sentences):
    retrieved = retrieve(query, k)
    top_score = retrieved[0][1] if retrieved else 0.0

    if top_score >= similarity_threshold:
        facts = "\n".join(
            [f"{i+1}. {doc[:300]}" for i, (doc, _) in enumerate(retrieved)]
        )

        prompt = f"""
Use ONLY the facts below to answer.
Maximum {answer_sentences} short sentences.

FACTS:
{facts}

QUESTION:
{query}
"""
        mode = "KB"

    else:
        prompt = f"""
Answer briefly in maximum {answer_sentences} short sentences.

Question: {query}
"""
        mode = "Gemini"

    answer = call_gemini(prompt)

    # limit sentences
    sentences = answer.split(". ")
    answer = ". ".join(sentences[:answer_sentences]).strip()

    if not answer.endswith("."):
        answer += "."

    return prompt, answer, retrieved, mode


# ---------------- UI ----------------
st.title("Small RAG Demo — KB First, Gemini Fallback")

query = st.text_input("Enter question", "What is AI?")
k = st.slider("How many KB docs (k)", 1, 5, 2)
similarity_threshold = st.slider("Similarity threshold", 0.0, 1.0, 0.40, 0.01)
answer_sentences = st.slider("Max sentences", 1, 4, 2)

if st.button("Ask"):

    if not st.session_state.models_loaded:
        with st.spinner("Loading embedding model..."):
            lazy_load_models()

    with st.spinner("Processing..."):
        t0 = time.time()
        prompt, answer, retrieved, mode = rag_answer(
            query, k, similarity_threshold, answer_sentences
        )
        t1 = time.time()

    if mode == "KB":
        st.info("✔ Answered from Knowledge Base")
    else:
        st.success("✔ Answered directly by Gemini")

    st.subheader("Timing")
    st.write(f"{t1 - t0:.2f}s")

    st.subheader("Retrieved facts")
    for doc, score in retrieved:
        st.markdown(f"- ({score:.4f}) {escape(doc)}")

    with st.expander("Show prompt"):
        st.code(prompt)

    st.subheader("Model Answer")
    st.write(answer)
