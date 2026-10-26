# Chat with your PDF (RAG)

Upload a PDF and ask questions. Answers come only from the document and show source page numbers.

**Live demo:** https://fegqopnz8eaxefxmhvbzsu.streamlit.app/

Built with Streamlit, FastEmbed, FAISS and the Groq API.

## How it works
1. Extract text from the PDF page by page
2. Split into overlapping chunks and embed them
3. Retrieve the most relevant chunks with FAISS
4. Ask the LLM to answer only from those chunks and show the source pages
