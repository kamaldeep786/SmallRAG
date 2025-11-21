# app.py
# pip install sentence-transformers numpy google-generativeai streamlit

import os
import streamlit as st
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import numpy as np
import time

st.set_page_config(page_title="Small RAG with Gemini (KB + Web fallback)", layout="wide")

# ---- Configuration / secrets ----
API_KEY = st.secrets.get("GENAI_API_KEY", os.environ.get("GENAI_API_KEY"))
if not API_KEY:
    st.warning("No Gemini API key found. Set GENAI_API_KEY in Streamlit Secrets or environment.")
else:
    genai.configure(api_key=API_KEY)

# ---- Model / embedder initialization (cached) ----
@st.cache_resource
def load_models():
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    # small in-memory KB
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

# ---- Web search via Gemini ----
def web_search_gemini(query, max_snippets=5):
    """
    Uses Gemini's google_search tool to fetch web snippets.
    Returns a string containing concatenated snippets.
    """
    if not API_KEY:
        return "No API key configured - cannot perform web search."

    try:
        # Create a model instance with access to google_search tool
        model = genai.GenerativeModel("gemini-2.0-flash", tools=[{"google_search": {}}])

        # The exact SDK call shape may vary; this uses the tool invocation style.
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
        # response.candidates may contain content parts depending on SDK version
        # We extract text parts defensively.
        if hasattr(response, "candidates"):
            for c in response.candidates:
                if hasattr(c, "content") and getattr(c.content, "parts", None):
                    for part in c.content.parts:
                        text = getattr(part, "text", None)
                        if text:
                            snippets.append(text.strip())
                else:
                    # fallback: string representation
                    snippets.append(str(c))
        else:
            # fallback to raw string
            snippets.append(str(response))

        # Deduplicate and join a few snippets
        unique_snips = []
        for s in snippets:
            if s not in unique_snips and len(unique_snips) < max_snippets:
                unique_snips.append(s)

        return "\n\n".join(unique_snips).strip() or ("No web snippets found for: " + query)

    except Exception as e:
        return f"Web search failed: {e}"

# ---- RAG answer with web fallback ----
def rag_answer(query, k=2, similarity_threshold=0.40, max_web_snippets=5):
    """
    1) Retrieve from KB.
    2) If top similarity < similarity_threshold -> perform web search.
    3) Build facts (either KB facts or web snippets) and ask Gemini to answer using ONLY those facts.
    Returns: prompt, answer_text, retrieved_list, used_web (bool), web_snippets (str or None)
    """
    retrieved = retrieve(query, k=k)
    top_score = retrieved[0][1] if retrieved else 0.0
    use_web = top_score < similarity_threshold

    web_snippets = None
    if not use_web:
        # Use local KB facts
        facts = "\n".join([f"- {text}" for text, score in retrieved])
    else:
        # Web fallback
        web_snippets = web_search_gemini(query, max_snippets=max_web_snippets)
        # If web search returned nothing sensible, still create a minimal fact instruction
        if web_snippets and not web_snippets.lower().startswith("web search failed"):
            # break into lines and prefix for prompt
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

        # Robustly extract text from response (SDKs differ)
        text = getattr(resp, "text", None) or (resp.output_text if hasattr(resp, "output_text") else None)
        if not text:
            # Try candidates content parts
            if hasattr(resp, "candidates"):
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
        return promp
