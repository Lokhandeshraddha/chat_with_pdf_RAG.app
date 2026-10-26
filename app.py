import os
import time

import faiss
import numpy as np
import streamlit as st
from fastembed import TextEmbedding
from groq import Groq
from pypdf import PdfReader

st.set_page_config(page_title="Chat with your PDF", page_icon="📄")

MODEL = "openai/gpt-oss-20b"  # use any chat model available on your Groq key
EMBED_MODEL = "BAAI/bge-small-en-v1.5"  # small ONNX model, no PyTorch needed


def get_api_key():
    try:
        return st.secrets["GROQ_API_KEY"]
    except Exception:
        return os.environ.get("GROQ_API_KEY")


@st.cache_resource(show_spinner="Loading embedding model...")
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
    return resp.choices[0].message.content, pages_used, time.time() - start


st.title("📄 Chat with your PDF")
st.caption(
    "RAG: upload a PDF and ask questions. Answers come only from the document "
    "and show the source page numbers."
)

api_key = get_api_key()
if not api_key:
    st.error("GROQ_API_KEY is not set. Add it in the app's Secrets (see README).")
    st.stop()
client = Groq(api_key=api_key)

uploaded = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded is not None:
    if st.session_state.get("file_id") != uploaded.file_id:
        with st.spinner("Reading and indexing the PDF..."):
            store = build_index(uploaded)
        st.session_state.file_id = uploaded.file_id
        st.session_state.store = store
        st.session_state.messages = []
    store = st.session_state.get("store")
    if store is None:
        st.warning(
            "No text found in this PDF (it may be a scanned image). Try another PDF."
        )
        st.stop()
    st.success(f"Ready: {store['pages']} pages, {len(store['chunks'])} chunks.")

    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    question = st.chat_input("Ask a question about your PDF")
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            try:
                with st.spinner("Thinking..."):
                    text, pgs, secs = answer(question, store, client)
                reply = f"{text}\n\n*Sources: pages {pgs} | {secs:.1f}s*"
            except Exception as e:  # rate limit, model not available, etc.
                reply = f"The language model request failed: {e}"
            st.markdown(reply)
        st.session_state.messages.append({"role": "assistant", "content": reply})
else:
    st.info("Upload a PDF to get started.")
