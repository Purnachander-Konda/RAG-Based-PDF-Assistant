import os, json, pickle, time
from typing import List, Tuple
import faiss
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import requests
import streamlit as st

load_dotenv()

# Default config
GENERATOR    = os.getenv("GENERATOR", "ollama").strip().lower()   # "ollama" or "gemini"
EMBED_MODEL  = (os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
                .strip().strip('"').strip("'"))
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral:3b-instruct-q4_K_M").strip()
OLLAMA_HOST  = os.getenv("OLLAMA_HOST", "http://localhost:11434").strip()

# Prompts
SYSTEM_PROMPT = (
    "You are a concise research assistant. Answer strictly using the provided context. "
    "If the answer is not in the context, say you cannot find it. "
    "Include brief citations like [source:filename, chunk]."
)

CHAR_BUDGET = 6000
_ollama_client = None   # lazy singleton


def _ollama_up() -> bool:
    try:
        import ollama
        ollama.Client(host=OLLAMA_HOST).list()
        return True
    except Exception:
        return False

def _init_ollama():
    global _ollama_client
    if _ollama_client is None:
        import ollama
        _ollama_client = ollama.Client(host=OLLAMA_HOST)
    return _ollama_client

# Models to try in order (gemini-2.5-flash is the current stable model)
_GEMINI_MODELS = ["gemini-2.5-flash", "gemini-2.0-flash"]

def _gemini_answer(prompt: str, max_retries: int = 3) -> str:
    api_key = os.getenv("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("No Gemini API key found. Add GEMINI_API_KEY in Streamlit secrets.")

    headers = {"Content-Type": "application/json"}
    payload = {"contents": [{"parts": [{"text": prompt}]}]}

    errors = []
    for model in _GEMINI_MODELS:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
        for attempt in range(max_retries):
            resp = requests.post(url, headers=headers, json=payload, timeout=60)
            if resp.status_code == 429:
                wait = 2 ** attempt * 10  # 10s, 20s, 40s
                time.sleep(wait)
                continue
            if resp.status_code in (401, 403):
                raise RuntimeError("Invalid or expired API key. Create a new one at aistudio.google.com/apikey")
            if resp.status_code >= 400:
                errors.append(f"{model}: {resp.status_code}")
                break  # try next model
            data = resp.json()
            candidates = data.get("candidates", [])
            if candidates:
                return candidates[0]["content"]["parts"][0]["text"].strip()
            errors.append(f"{model}: empty response")
            break
        else:
            errors.append(f"{model}: rate limited after {max_retries} retries")
            continue

    raise RuntimeError(
        f"Gemini API failed ({', '.join(errors)}). "
        f"Free tier allows ~15 req/min. Wait a minute and try again."
    )


@st.cache_resource(show_spinner=False)
def _load_embedding_model(model_name: str):
    """Cache the embedding model so it's not reloaded on every Streamlit rerun."""
    return SentenceTransformer(model_name)

def load_index(store_dir: str = "vector_store"):
    """Load FAISS index + metadata + embedding model."""
    index = faiss.read_index(os.path.join(store_dir, "faiss.index"))
    with open(os.path.join(store_dir, "meta.pkl"), "rb") as f:
        meta = pickle.load(f)
    with open(os.path.join(store_dir, "config.json")) as f:
        cfg = json.load(f)

    emb_model_name = cfg.get("embed_model", EMBED_MODEL)
    emb = _load_embedding_model(emb_model_name)
    return index, meta, emb

def retrieve(query: str, index, meta, emb, top_k: int = 5) -> List[dict]:
    """Vector search Top-K chunks."""
    import numpy as np
    q = emb.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(q)
    D, I = index.search(q, top_k)
    hits = []
    for score, idx in zip(D[0], I[0]):
        if idx == -1:
            continue
        d = meta[idx]
        hits.append({"score": float(score), **d})
    return hits

def format_context(hits: List[dict]) -> str:
    blocks, used = [], 0
    for h in hits:
        block = f"[source:{h['source']}, chunk:{h['chunk']}]\n{h['text']}"
        if used + len(block) > CHAR_BUDGET:
            break
        blocks.append(block)
        used += len(block)
    return "\n\n".join(blocks)

def _build_prompt(query: str, hits: List[dict]) -> str:
    context = format_context(hits)
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n"
        f"Answer:"
    )

def answer(query: str, hits: List[dict], generator: str = GENERATOR) -> str:
    prompt = _build_prompt(query, hits)

    if generator == "ollama":
        if not _ollama_up():
            raise RuntimeError(
                f"Ollama service not reachable at {OLLAMA_HOST}. "
                f"Start it with: ollama run {OLLAMA_MODEL}"
            )
        client = _init_ollama()
        resp = client.generate(
            model=OLLAMA_MODEL,
            prompt=prompt,
            options={"temperature": 0.2, "num_predict": 400, "num_ctx": 4096}
        )
        return (resp.get("response") or "").strip()

    elif generator == "gemini":
        return _gemini_answer(prompt)

    else:
        raise ValueError(f"Unsupported generator: {generator}")

def rag_query(query: str, store_dir: str = "vector_store", top_k: int = 5, generator: str = GENERATOR) -> Tuple[str, List[dict]]:
    index, meta, emb = load_index(store_dir)
    hits = retrieve(query, index, meta, emb, top_k=top_k)
    ans = answer(query, hits, generator=generator)
    return ans, hits
