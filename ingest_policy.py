"""
RAG knowledge base ingest: load airline_policy.pdf, chunk, embed, store in Chroma.
Set GEMINI_API_KEY before running.
"""

from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Paths and config
PDF_PATH = Path(__file__).resolve().parent / "airline_policy.pdf"
CHROMA_DIR = Path(__file__).resolve().parent / "policy_db"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


def main() -> None:
    # 1. Load PDF with PyPDFLoader
    loader = PyPDFLoader(str(PDF_PATH))
    documents = loader.load()
    print(
        f"[1] Loaded PDF: {len(documents)} pages, "
        f"{sum(len(d.page_content) for d in documents)} chars"
    )

    # 2. Split into chunks (size=1000, overlap=200)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )
    chunks = splitter.split_documents(documents)
    print(f"[2] Split into {len(chunks)} chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")

    # 3. Create embeddings via GoogleGenerativeAIEmbeddings (uses GEMINI_API_KEY)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

    # 4. Persist chunks into local Chroma store 'policy_db'
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(CHROMA_DIR),
        collection_name="policy",
    )
    print(f"[4] Saved to Chroma at {CHROMA_DIR}")


if __name__ == "__main__":
    main()
