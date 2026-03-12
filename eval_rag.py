import os, time, json, re, argparse, statistics, csv
from pathlib import Path
from typing import List, Dict

from rag import rag_query
from ingest import read_pdf

try:
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except Exception:
    HAVE_MPL = False

DATA_DIR = Path("data")
OUT_DIR = Path("benchmarks")
OUT_DIR.mkdir(exist_ok=True)

def norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip().lower()

def load_corpus_texts(data_dir: Path) -> Dict[str, str]:
    texts = {}
    if not data_dir.exists():
        return texts
    for p in data_dir.iterdir():
        if p.suffix.lower() == ".pdf":
            try:
                texts[p.name] = read_pdf(str(p))
            except Exception as e:
                print(f"[WARN] Could not read {p.name}: {e}")
    return texts

def split_sentences(text: str) -> List[str]:
    return re.split(r"(?<=[.!?])\s+", text)

def keyword_baseline_answer(question: str, corpus: Dict[str, str]) -> str:
    q = norm(question)
    q_tokens = set([t for t in re.findall(r"[a-z0-9]+", q) if len(t) > 2])
    best_sent, best_score = "", -1
    for _, txt in corpus.items():
        for sent in split_sentences(txt):
            s_norm = norm(sent)
            s_tokens = set([t for t in re.findall(r"[a-z0-9]+", s_norm) if len(t) > 2])
            overlap = len(q_tokens & s_tokens)
            if overlap > best_score:
                best_score = overlap
                best_sent = sent
    return best_sent or ""

def contains_expected(pred: str, expected_substring: str) -> bool:
    return norm(expected_substring) in norm(pred)

def evaluate(qa_path: Path, store_dir: str, top_k: int):
    qa = json.loads(Path(qa_path).read_text(encoding="utf-8"))
    assert isinstance(qa, list) and qa, "qa.json must be a non-empty list of items"

    corpus = load_corpus_texts(DATA_DIR)
    if not corpus:
        print("[WARN] No PDFs found in ./data for baseline; baseline recall may be 0.")

    base_correct = 0
    rag_correct = 0
    base_latencies, rag_latencies = [], []
    rows = []

    for item in qa:
        q = item["question"]
        expected = item["expected_substring"]

        # Baseline timing
        t0 = time.perf_counter()
        base_ans = keyword_baseline_answer(q, corpus) if corpus else ""
        t1 = time.perf_counter()

        # RAG timing
        t2 = time.perf_counter()
        rag_ans, _hits = rag_query(q, store_dir=store_dir, top_k=top_k)
        t3 = time.perf_counter()

        base_ok = contains_expected(base_ans, expected)
        rag_ok  = contains_expected(rag_ans, expected)

        base_correct += int(base_ok)
        rag_correct  += int(rag_ok)

        base_lat = t1 - t0
        rag_lat  = t3 - t2
        base_latencies.append(base_lat)
        rag_latencies.append(rag_lat)

        rows.append({
            "question": q,
            "expected_substring": expected,
            "baseline_ok": base_ok,
            "rag_ok": rag_ok,
            "baseline_answer": base_ans,
            "rag_answer": rag_ans,
            "baseline_latency_s": round(base_lat, 4),
            "rag_latency_s": round(rag_lat, 4),
        })

    n = len(qa)
    baseline_recall = base_correct / n if n else 0.0
    rag_recall = rag_correct / n if n else 0.0
    recall_improvement_pct = ((rag_recall - baseline_recall) / max(baseline_recall, 1e-9)) * 100.0

    base_avg_lat = statistics.mean(base_latencies) if base_latencies else None
    rag_avg_lat  = statistics.mean(rag_latencies)  if rag_latencies  else None

    summary = {
        "num_questions": n,
        "baseline_correct": base_correct,
        "rag_correct": rag_correct,
        "baseline_recall": round(baseline_recall, 4),
        "rag_recall": round(rag_recall, 4),
        "recall_improvement_pct": round(recall_improvement_pct, 1),
        "baseline_avg_latency_s": round(base_avg_lat, 4) if base_avg_lat is not None else None,
        "rag_avg_latency_s": round(rag_avg_lat, 4) if rag_avg_lat is not None else None,
    }
    return summary, rows

def save_csv(rows: List[Dict], path: Path):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

def save_plots(summary: Dict, outdir: Path):
    if not HAVE_MPL:
        print("[INFO] matplotlib not installed; skipping plots.")
        return
    # Recall comparison
    fig1 = plt.figure()
    xs = ["Baseline", "RAG"]
    ys = [summary["baseline_recall"], summary["rag_recall"]]
    plt.bar(xs, ys)  # Do not set colors/styles
    plt.ylabel("Recall")
    plt.title("Recall Comparison")
    for i, v in enumerate(ys):
        plt.text(i, v + 0.01, f"{v:.2f}", ha="center")
    fig1.tight_layout()
    fig1.savefig(outdir / "recall_comparison.png", dpi=150)
    plt.close(fig1)

    # Latency comparison
    fig2 = plt.figure()
    xs = ["Baseline", "RAG"]
    ys = [
        summary.get("baseline_avg_latency_s") or 0.0,
        summary.get("rag_avg_latency_s") or 0.0
    ]
    plt.bar(xs, ys)
    plt.ylabel("Average Latency (s)")
    plt.title("Latency Comparison")
    for i, v in enumerate(ys):
        plt.text(i, v + (0.02 if v else 0.0), f"{v:.2f}s", ha="center")
    fig2.tight_layout()
    fig2.savefig(outdir / "latency_comparison.png", dpi=150)
    plt.close(fig2)

def main():
    ap = argparse.ArgumentParser(description="Evaluate RAG vs keyword baseline on a QA set.")
    ap.add_argument("--qa", default="qa.json", help="Path to QA JSON file")
    ap.add_argument("--store_dir", default="vector_store", help="FAISS store directory")
    ap.add_argument("--top_k", type=int, default=5, help="Top-K passages to retrieve")
    ap.add_argument("--csv", default=str(OUT_DIR / "results.csv"), help="Where to save detailed CSV")
    ap.add_argument("--no_plots", action="store_true", help="Disable PNG plot export")
    args = ap.parse_args()

    summary, rows = evaluate(Path(args.qa), store_dir=args.store_dir, top_k=args.top_k)

    print("\n=== RAG Evaluation Summary ===")
    print(f"Questions:                 {summary['num_questions']}")
    print(f"Baseline correct:          {summary['baseline_correct']}")
    print(f"RAG correct:               {summary['rag_correct']}")
    print(f"Baseline recall:           {summary['baseline_recall']:.2f}")
    print(f"RAG recall:                {summary['rag_recall']:.2f}")
    print(f"Recall improvement (%):    {summary['recall_improvement_pct']:.1f}%")
    if summary["baseline_avg_latency_s"] is not None:
        print(f"Baseline avg latency (s):  {summary['baseline_avg_latency_s']:.3f}")
    if summary["rag_avg_latency_s"] is not None:
        print(f"RAG avg latency (s):       {summary['rag_avg_latency_s']:.3f}")

    if rows:
        save_csv(rows, Path(args.csv))
        print(f"\nSaved per-question details to {args.csv}")

    if not args.no_plots:
        save_plots(summary, OUT_DIR)
        print(f"Saved plots to {OUT_DIR / 'recall_comparison.png'} and {OUT_DIR / 'latency_comparison.png'}")

if __name__ == "__main__":
    main()
