import os
import streamlit as st
from dotenv import load_dotenv
from ingest import build_index
from rag_deploy import rag_query

load_dotenv()

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="RAG PDF Assistant",
    page_icon="📄",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
    /* Main container spacing */
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }

    /* Hero header */
    .hero {
        text-align: center;
        padding: 1.5rem 0 0.5rem;
    }
    .hero h1 {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .hero p {
        color: #888;
        font-size: 1.05rem;
    }

    /* Cards */
    .card {
        border: 1px solid rgba(128,128,128,0.15);
        border-radius: 12px;
        padding: 1.4rem 1.6rem;
        margin-bottom: 1rem;
        background: rgba(128,128,128,0.03);
    }
    .card-title {
        font-size: 1.05rem;
        font-weight: 600;
        margin-bottom: 0.6rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* Answer box */
    .answer-box {
        border-left: 4px solid #667eea;
        padding: 1rem 1.2rem;
        border-radius: 0 10px 10px 0;
        background: rgba(102,126,234,0.06);
        margin: 0.8rem 0;
        line-height: 1.7;
    }

    /* Source chips */
    .source-chip {
        display: inline-block;
        padding: 0.25rem 0.7rem;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 500;
        background: rgba(102,126,234,0.12);
        color: #667eea;
        margin: 0.15rem 0.2rem;
    }

    /* Context block */
    .ctx-block {
        border: 1px solid rgba(128,128,128,0.1);
        border-radius: 8px;
        padding: 0.9rem 1rem;
        margin-bottom: 0.6rem;
        font-size: 0.9rem;
        background: rgba(128,128,128,0.02);
    }
    .ctx-header {
        font-weight: 600;
        font-size: 0.82rem;
        margin-bottom: 0.4rem;
        color: #888;
    }

    /* Score bar */
    .score-bar {
        height: 4px;
        border-radius: 2px;
        background: linear-gradient(90deg, #667eea, #764ba2);
        margin-top: 0.5rem;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: rgba(128,128,128,0.02);
    }
    section[data-testid="stSidebar"] .block-container {
        padding-top: 1.5rem;
    }

    /* Footer */
    .footer {
        text-align: center;
        padding: 1.5rem 0 0.5rem;
        color: #999;
        font-size: 0.82rem;
    }
    .footer a { color: #667eea; text-decoration: none; }
    .footer a:hover { text-decoration: underline; }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ── Sidebar — Settings & Upload ──────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Settings")

    gen_display = st.selectbox(
        "LLM Provider",
        ["Gemini (Cloud)", "Ollama (Local)"],
        index=0,
        help="Gemini works everywhere. Ollama requires a local server.",
    )
    generator = "ollama" if "Ollama" in gen_display else "gemini"

    top_k = st.slider("Passages to retrieve", 3, 10, 5, help="More passages = more context but slower")

    st.markdown("---")
    st.markdown("### 📂 Upload PDFs")

    uploaded = st.file_uploader(
        "Drop your PDFs here",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded:
        os.makedirs("data", exist_ok=True)
        saved_paths = []
        for f in uploaded:
            path = os.path.join("data", f.name)
            with open(path, "wb") as out:
                out.write(f.read())
            saved_paths.append(path)
        st.success(f"{len(saved_paths)} PDF(s) uploaded")

        # Auto-build the index right after upload
        with st.spinner("Building search index…"):
            build_index(saved_paths, out_dir="vector_store")
        st.success("✅ Index ready — you can now ask questions!")

    if st.button("🔨 Rebuild Index", use_container_width=True,
                 help="Re-index all PDFs in case you uploaded new ones"):
        pdf_dir = "data"
        if not os.path.isdir(pdf_dir):
            st.error("Upload PDFs first.")
        else:
            pdfs = [os.path.join(pdf_dir, p) for p in os.listdir(pdf_dir) if p.lower().endswith(".pdf")]
            if not pdfs:
                st.error("No PDFs found. Upload first.")
            else:
                with st.spinner("Rebuilding FAISS index…"):
                    build_index(pdfs, out_dir="vector_store")
                st.success("✅ Index rebuilt!")

    st.markdown("---")
    st.markdown(
        '<p style="font-size:0.78rem;color:#999;">'
        'Built with FAISS · SentenceTransformers · Gemini<br>'
        'By <b>Purnachander Konda</b></p>',
        unsafe_allow_html=True,
    )

# ── Hero ─────────────────────────────────────────────────────
st.markdown(
    '<div class="hero">'
    '<h1>📄 RAG PDF Assistant</h1>'
    '<p>Upload research papers and ask questions — answers are grounded in your documents.</p>'
    '</div>',
    unsafe_allow_html=True,
)

# ── How-it-works pills ───────────────────────────────────────
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown(
        '<div class="card"><div class="card-title">📤 Upload</div>'
        '<span style="font-size:0.88rem;color:#888;">Drop one or more PDFs in the sidebar</span></div>',
        unsafe_allow_html=True,
    )
with c2:
    st.markdown(
        '<div class="card"><div class="card-title">🔍 Index</div>'
        '<span style="font-size:0.88rem;color:#888;">Build a vector index for fast retrieval</span></div>',
        unsafe_allow_html=True,
    )
with c3:
    st.markdown(
        '<div class="card"><div class="card-title">💬 Ask</div>'
        '<span style="font-size:0.88rem;color:#888;">Get cited answers from your documents</span></div>',
        unsafe_allow_html=True,
    )

st.markdown("")

# ── Query input ──────────────────────────────────────────────
query = st.text_input(
    "Ask a question",
    placeholder="e.g. What are the main findings of the paper?",
    label_visibility="collapsed",
)

col_btn, _ = st.columns([1, 3])
with col_btn:
    ask = st.button("🚀 Get Answer", use_container_width=True, type="primary")

# ── Answer ───────────────────────────────────────────────────
_index_exists = os.path.isfile(os.path.join("vector_store", "faiss.index"))

if ask and query:
    if not _index_exists:
        st.warning("⚠️ No index found yet. Upload PDFs in the sidebar first — the index will be built automatically.")
    else:
        with st.spinner("Searching documents & generating answer…"):
            try:
                answer, hits = rag_query(
                    query,
                    store_dir="vector_store",
                    top_k=top_k,
                    generator=generator,
                )

                # Answer card
                st.markdown(f'<div class="answer-box">{answer}</div>', unsafe_allow_html=True)

                # Source chips
                sources = list({h["source"] for h in hits})
                chips = " ".join(f'<span class="source-chip">{s}</span>' for s in sources)
                st.markdown(f"**Sources:** {chips}", unsafe_allow_html=True)

                # Expandable context
                with st.expander("📚 View retrieved passages"):
                    for i, h in enumerate(hits):
                        score_pct = min(h["score"] * 100, 100)
                        st.markdown(
                            f'<div class="ctx-block">'
                            f'<div class="ctx-header">{h["source"]} · chunk {h["chunk"]} · relevance {h["score"]:.3f}</div>'
                            f'{h["text"]}'
                            f'<div class="score-bar" style="width:{score_pct}%"></div>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

            except Exception as e:
                st.error(f"Something went wrong: {e}")

elif ask and not query:
    st.warning("Please type a question first.")

# ── Footer ───────────────────────────────────────────────────
st.markdown("")
st.markdown(
    '<div class="footer">'
    'Made with ❤️ by <a href="https://github.com/Purnachander-Konda" target="_blank">Purnachander Konda</a> · '
    'Powered by FAISS + Gemini'
    '</div>',
    unsafe_allow_html=True,
)
