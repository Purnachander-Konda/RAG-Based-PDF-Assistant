import os, json, pickle
from typing import List, Tuple
import faiss
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

# Config from .env 
GENERATOR    = os.getenv("GENERATOR", "local_ollama").strip().lower()
EMBED_MODEL  = (os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5").strip()
                .strip('"').strip("'"))
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral:3b-instruct-q4_K_M").strip()
OLLAMA_HOST  = os.getenv("OLLAMA_HOST", "http://localhost:11434").strip()

# Lazy singleton
_ollama_client = None

# Prompts
SYSTEM_PROMPT = (
    "You are a concise research assistant. Answer strictly using the provided context. "
    "If the answer is not in the context, say you cannot find it. "
    "Include brief citations like [source:filename, chunk]."
)

CHAR_BUDGET = 6000  


# Ollama
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


# Index & Retrieval
def load_index(store_dir: str = "vector_store"):
    """Load FAISS index + metadata + embedding model."""
    index = faiss.read_index(os.path.join(store_dir, "faiss.index"))
    with open(os.path.join(store_dir, "meta.pkl"), "rb") as f:
        meta = pickle.load(f)
    with open(os.path.join(store_dir, "config.json")) as f:
        cfg = json.load(f)
    emb_model_name = cfg.get("embed_model", EMBED_MODEL)
    emb = SentenceTransformer(emb_model_name)
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


# Prompt building & Answering
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

def answer(query: str, hits: List[dict]) -> str:
    if not _ollama_up():
        raise RuntimeError(
            f"Ollama service not reachable at {OLLAMA_HOST}. "
            f"Start it with: ollama run {OLLAMA_MODEL}"
        )
    client = _init_ollama()
    prompt = _build_prompt(query, hits)
    resp = client.generate(
        model=OLLAMA_MODEL,
        prompt=prompt,
        options={
            "temperature": 0.2,
            "num_predict": 400,
            "num_ctx": 4096
        }
    )
    return (resp.get("response") or "").strip()


# ==== Public API ====
def rag_query(query: str, store_dir: str = "vector_store", top_k: int = 5) -> Tuple[str, List[dict]]:
    index, meta, emb = load_index(store_dir)
    hits = retrieve(query, index, meta, emb, top_k=top_k)
    ans = answer(query, hits)
    return ans, hits
