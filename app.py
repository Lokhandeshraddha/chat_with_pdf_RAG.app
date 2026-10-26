import os
import time

import faiss
import numpy as np
import streamlit as st
from fastembed import TextEmbedding
from groq import Groq
from pypdf import PdfReader

st.set_page_config(
    page_title="Chat with your PDF",
    page_icon="📄",
    layout="centered",
    initial_sidebar_state="expanded",
)

MODEL = "openai/gpt-oss-20b"  # use any chat model available on your Groq key
EMBED_MODEL = "BAAI/bge-small-en-v1.5"  # small ONNX model, no PyTorch needed

EXAMPLES = [
    "Give a short summary of this document",
    "What are the key points?",
    "What topics does this document cover?",
]

CSS = """
<style>
.block-container { padding-top: 2rem; max-width: 860px; }
.hero {
    background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%);
    padding: 1.6rem 1.8rem; border-radius: 18px; margin-bottom: 1.2rem;
    box-shadow: 0 8px 24px rgba(79, 70, 229, 0.25);
}
.hero h1 { margin: 0; font-size: 2rem; color: #ffffff !important; padding: 0; }
.hero p { margin: .45rem 0 0; color: #E0E7FF !important; font-size: 1rem; }
.badge {
    display: inline-block; background: #E0E7FF; color: #3730A3;
    border-radius: 999px; padding: 2px 11px; font-size: .78rem;
    font-weight: 600; margin: 6px 6px 0 0;
}
.meta { color: #64748B; font-size: .78rem; margin-top: 6px; }
.empty-card {
    background: #ffffff; border: 1px dashed #A5B4FC; border-radius: 14px;
    padding: 1.1rem 1.3rem; color: #334155; margin-top: .4rem;
}
.loader { display: flex; align-items: center; gap: 12px; color: #475569; padding: 4px 0; }
.loader .icon { font-size: 1.4rem; animation: pulse 1.4s infinite ease-in-out; }
.dots span {
    display: inline-block; width: 9px; height: 9px; border-radius: 50%;
    background: #4F46E5; margin-right: 5px;
    animation: bounce 1.2s infinite ease-in-out;
}
.dots span:nth-child(2) { animation-delay: .15s; }
.dots span:nth-child(3) { animation-delay: .30s; }
@keyframes bounce {
    0%, 80%, 100% { transform: scale(.6); opacity: .45; }
    40% { transform: scale(1); opacity: 1; }
}
@keyframes pulse {
    0%, 100% { transform: scale(1); }
    50% { transform: scale(1.18); }
}
section[data-testid="stSidebar"] { border-right: 1px solid #E0E7FF; }
.stButton > button { border-radius: 10px; border: 1px solid #C7D2FE; }
</style>
"""


def loader_html(text, icon=True):
    icon_html = '<span class="icon">📄</span>' if icon else ""
    return (
        f'<div class="loader">{icon_html}'
        '<span class="dots"><span></span><span></span><span></span></span>'
        f"<span>{text}</span></div>"
    )


def get_api_key():
    try:
        return st.secrets["GROQ_API_KEY"]
    except Exception:
        return os.environ.get("GROQ_API_KEY")


@st.cache_resource(show_spinner=False)
def load_embedder():
    return TextEmbedding(model_name=EMBED_MODEL)


def embed(texts):
    vecs = np.array(list(load_embedder().embed(texts)), dtype="float32")
    faiss.normalize_L2(vecs)
    return vecs


def load_pdf(file):
    reader = PdfReader(file)
    return [(i + 1, p.extract_text() or "") for i, p in enumerate(reader.pages)]


def chunk_text(pages, size=800, overlap=150):
    chunks = []
    for pg, text in pages:
        text = " ".join(text.split())
        start = 0
        while start < len(text):
            chunks.append({"page": pg, "text": text[start:start + size]})
            start += size - overlap
    return chunks


def build_index(file):
    pages = load_pdf(file)
    chunks = chunk_text(pages)
    if not chunks:
        return None
    vecs = embed([c["text"] for c in chunks])
    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)
    return {"index": index, "chunks": chunks, "pages": len(pages)}


def retrieve(question, store, k=4):
    qv = embed([question])
    _, ids = store["index"].search(qv, k)
    return [store["chunks"][i] for i in ids[0] if i != -1]


def answer(question, store, client):
    start = time.time()
    ctx = retrieve(question, store)
    context = "\n\n".join(f"[Page {c['page']}] {c['text']}" for c in ctx)
    messages = [
        {
            "role": "system",
            "content": "Answer ONLY using the given context. "
            "If the answer is not in the context, say you could not find it in the document.",
        },
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
    ]
    resp = client.chat.completions.create(model=MODEL, messages=messages, temperature=0.2)
    pages_used = sorted({c["page"] for c in ctx})
    return resp.choices[0].message.content, pages_used, time.time() - start, ctx


def render_assistant(msg):
    st.markdown(msg["content"])
    if msg.get("pages"):
        badges = "".join(f'<span class="badge">Page {p}</span>' for p in msg["pages"])
        st.markdown(badges, unsafe_allow_html=True)
    if msg.get("secs") is not None:
        st.markdown(f'<div class="meta">Answered in {msg["secs"]:.1f}s</div>', unsafe_allow_html=True)
    if msg.get("sources"):
        with st.expander("View source passages"):
            for s in msg["sources"]:
                st.caption(f"Page {s['page']}")
                st.text(s["text"])


def handle_question(question, store, client):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant", avatar="📄"):
        slot = st.empty()
        slot.markdown(loader_html("Searching your document...", icon=False), unsafe_allow_html=True)
        try:
            text, pages, secs, ctx = answer(question, store, client)
            msg = {
                "role": "assistant", "content": text, "pages": pages,
                "secs": secs, "sources": ctx,
            }
        except Exception as e:  # rate limit, model not available, etc.
            msg = {
                "role": "assistant",
                "content": f"Sorry, the language model request failed: {e}",
            }
        slot.empty()
        render_assistant(msg)
    st.session_state.messages.append(msg)


# ---------------------------------------------------------------- UI
st.markdown(CSS, unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### 📄 Chat with your PDF")
    st.markdown("**How it works**")
    st.markdown(
        "1. Upload a PDF\n"
        "2. It is split into small chunks and embedded\n"
        "3. Your question finds the most relevant chunks\n"
        "4. An LLM answers only from those chunks"
    )
    st.markdown("**Tips**")
    st.markdown(
        "- Works best with text PDFs (not scanned images)\n"
        "- Ask specific questions\n"
        "- Check the page badges to verify answers\n"
        "- Please avoid uploading private documents"
    )
    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    st.caption("Built with Streamlit, FastEmbed, FAISS and the Groq API.")

st.markdown(
    '<div class="hero"><h1>📄 Chat with your PDF</h1>'
    "<p>Upload a document and ask questions. Answers come only from your PDF "
    "and show the pages they came from.</p></div>",
    unsafe_allow_html=True,
)

api_key = get_api_key()
if not api_key:
    st.error("GROQ_API_KEY is not set. Add it in the app's Secrets (see README).")
    st.stop()
client = Groq(api_key=api_key)

uploaded = st.file_uploader("Upload a PDF", type=["pdf"], label_visibility="collapsed")

if uploaded is None:
    st.markdown(
        '<div class="empty-card"><b>Get started</b><br>'
        "Drop a PDF above (notes, a research paper, a report) and start asking questions. "
        "Every answer shows its source pages.</div>",
        unsafe_allow_html=True,
    )
    st.stop()

file_key = f"{uploaded.name}-{uploaded.size}"
if st.session_state.get("file_key") != file_key:
    slot = st.empty()
    slot.markdown(loader_html("Reading and indexing your PDF..."), unsafe_allow_html=True)
    try:
        store = build_index(uploaded)
    except Exception as e:
        slot.empty()
        st.error(f"Could not read this PDF: {e}")
        st.stop()
    slot.empty()
    st.session_state.file_key = file_key
    st.session_state.store = store
    st.session_state.messages = []

store = st.session_state.get("store")
if store is None:
    st.warning("No text found in this PDF (it may be a scanned image). Try another PDF.")
    st.stop()

st.success(f"Ready: {store['pages']} pages, {len(store['chunks'])} chunks indexed.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"], avatar="📄" if m["role"] == "assistant" else None):
        if m["role"] == "assistant":
            render_assistant(m)
        else:
            st.markdown(m["content"])

examples_slot = st.empty()
if not st.session_state.messages:
    with examples_slot.container():
        st.markdown("**Try asking:**")
        cols = st.columns(len(EXAMPLES))
        for i, (col, q) in enumerate(zip(cols, EXAMPLES)):
            if col.button(q, key=f"ex{i}", use_container_width=True):
                st.session_state.pending = q
                st.rerun()

question = st.chat_input("Ask a question about your PDF")
if question is None:
    question = st.session_state.pop("pending", None)
if question:
    examples_slot.empty()
    handle_question(question, store, client)
