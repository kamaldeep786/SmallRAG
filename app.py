# app.py
# pip install sentence-transformers numpy google-generativeai streamlit

import os
import streamlit as st
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import numpy as np

st.set_page_config(page_title="Small RAG with Gemini", layout="wide")

# ---- Configuration / secrets ----
# Streamlit Cloud: put {"GENAI_API_KEY": "your_api_key"} in Secrets
API_KEY = st.secrets.get("GENAI_API_KEY", os.environ.get("GENAI_API_KEY"))
if not API_KEY:
    st.warning("No Gemini API key found. Set GENAI_API_KEY in Streamlit Secrets or environment.")
else:
    genai.configure(api_key=API_KEY)

# ---- Model / embedder initialization (cached) ----
@st.cache_resource
def load_models():
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    # create or load your small KB here
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
    return embedder, documents, doc_embs

embedder, documents, doc_embs = load_models()

# ---- Retrieval function ----
def retrieve(query, k=3):
    qvec = embedder.encode([query], normalize_embeddings=True)[0]
    sims = doc_embs @ qvec
    top_idx = np.argsort(sims)[-k:][::-1]
    return [(documents[i], float(sims[i])) for i in top_idx]

# ---- RAG answer function ----
def rag_answer(query, k=2):
    retrieved = retrieve(query, k=k)
    facts = "\n".join([f"- {text}" for text, score in retrieved])
    prompt = f"""
Answer the question using ONLY these facts:

{facts}

Question: {query}
Answer briefly:
"""
    if not API_KEY:
        return prompt, "No API key configured - cannot call Gemini.", retrieved

    try:
        llm = genai.GenerativeModel("gemini-2.0-flash")
        resp = llm.generate_content(prompt)
        # depending on SDK version resp structure may differ:
        text = getattr(resp, "text", None) or (resp.output_text if hasattr(resp, "output_text") else str(resp))
        if hasattr(text, "strip"):
            text = text.strip()
        return prompt, text, retrieved
    except Exception as e:
        return prompt, f"Error calling Gemini: {e}", retrieved

# ---- Streamlit UI ----
st.title("Small RAG demo — Streamlit + Gemini")
st.markdown("Type a question and the app will retrieve matching facts from a tiny KB and call Gemini to produce a brief answer.")

query = st.text_input("Enter question", value="what is india?")
k = st.slider("How many documents to retrieve (k)", min_value=1, max_value=5, value=2)

if st.button("Ask"):
    with st.spinner("Retrieving and calling model..."):
        prompt, answer, retrieved = rag_answer(query, k=k)
    st.subheader("Retrieved facts")
    for doc, score in retrieved:
        st.write(f"- ({score:.4f}) {doc}")

    st.subheader("Prompt sent to Gemini")
    st.code(prompt, language="text")

    st.subheader("Model answer")
    st.write(answer)

st.sidebar.header("Debug / Settings")
st.sidebar.write("Documents in KB:", len(documents))
if st.sidebar.checkbox("Show KB documents"):
    for d in documents:
        st.sidebar.write("-", d)
