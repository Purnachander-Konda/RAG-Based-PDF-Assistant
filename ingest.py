import os, json, pickle, uuid, glob
from pathlib import Path
from typing import List, Dict

from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
from dotenv import load_dotenv

load_dotenv()
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

def read_pdf(path: str) -> str:
    reader = PdfReader(path)
    texts = []
    for i, page in enumerate(reader.pages):
        try:
            texts.append(page.extract_text() or "")
        except Exception:
            texts.append("")
    return "\n".join(texts)

def chunk_text(text: str, source: str) -> List[Dict]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900, chunk_overlap=150, length_function=len
    )
    chunks = splitter.split_text(text)
    docs = []
    for i, c in enumerate(chunks):
        docs.append({
            "id": f"{source}-{i}-{uuid.uuid4().hex[:8]}",
            "text": c,
            "source": source,
            "chunk": i
        })
    return docs

def build_index(input_paths: List[str], out_dir: str = "vector_store"):
    # Expand any globs and directories on Windows/PowerShell
    expanded_paths = []
    for p in input_paths:
        if os.path.isdir(p):
            expanded_paths.extend(
                [str(Path(p) / f) for f in os.listdir(p) if f.lower().endswith(".pdf")]
            )
        elif "*" in p or "?" in p:
            expanded_paths.extend(glob.glob(p))
        else:
            expanded_paths.append(p)

    input_paths = expanded_paths
    os.makedirs(out_dir, exist_ok=True)
    # 1) read + chunk
    corpus = []
    for p in input_paths:
        raw = read_pdf(p)
        corpus.extend(chunk_text(raw, source=Path(p).name))
    if not corpus:
        raise ValueError("No text extracted. Check your PDFs.")

    # 2) embed
    model = SentenceTransformer(EMBED_MODEL)
    vectors = model.encode([d["text"] for d in corpus], show_progress_bar=True, convert_to_numpy=True)

    # 3) FAISS index
    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)  # cosine via normalized dot
    faiss.normalize_L2(vectors)
    index.add(vectors)

    faiss.write_index(index, os.path.join(out_dir, "faiss.index"))

    with open(os.path.join(out_dir, "meta.pkl"), "wb") as f:
        pickle.dump(corpus, f)

    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump({"embed_model": EMBED_MODEL}, f)

    print(f"Indexed {len(corpus)} chunks from {len(input_paths)} PDFs.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdfs", nargs="+", required=True, help="Paths to PDF files")
    parser.add_argument("--out", default="vector_store")
    args = parser.parse_args()
    build_index(args.pdfs, args.out)
