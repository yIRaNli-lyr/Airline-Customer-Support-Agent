"""
RAG module for policy QA: retrieve from Chroma, answer using LLM.

Uses the same Chroma store and embeddings as ingest_policy.py.
Run `python ingest_policy.py` from project root before using.
"""

from pathlib import Path
from typing import List

from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from litellm import completion

# Project root (I3). Same resolution as agent policy path.
_AGENT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _AGENT_DIR.parent.parent.parent.parent
CHROMA_DIR = _PROJECT_ROOT / "policy_db"
COLLECTION_NAME = "policy"
EMBED_MODEL = "models/gemini-embedding-001"


def _get_embeddings():
    return GoogleGenerativeAIEmbeddings(model=EMBED_MODEL)


def _get_chroma():
    return Chroma(
        persist_directory=str(CHROMA_DIR),
        collection_name=COLLECTION_NAME,
        embedding_function=_get_embeddings(),
    )


def retrieve(query: str, k: int = 5) -> List[dict]:
    """
    Retrieve top-k chunks from the policy Chroma store.

    Args:
        query: User question or search text
        k: Number of chunks to return

    Returns:
        List of dicts with "text" and optional "metadata" (e.g. page)
    """
    chroma = _get_chroma()
    docs = chroma.similarity_search(query, k=k)
    return [
        {"text": d.page_content, "metadata": getattr(d, "metadata", {}) or {}}
        for d in docs
    ]


def answer_policy_rag_only(query: str, model: str = "gemini/gemini-2.0-flash") -> str:
    """
    Answer a policy question using only RAG retrieval + one LLM call.
    No tools, no agent loop. Used for pipeline eval.

    Args:
        query: Policy question
        model: LLM model string for LiteLLM

    Returns:
        Model answer string
    """
    chunks = retrieve(query, k=5)
    context = "\n\n---\n\n".join(c["text"] for c in chunks)

    system = (
        "Answer based ONLY on the following policy context. "
        "If the context does not contain enough information, say so. "
        "Do not invent details. When possible, quote or cite the relevant part."
    )
    user = f"Context:\n\n{context}\n\nQuestion: {query}\n\nAnswer:"

    response = completion(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.0,
    )
    return (response.choices[0].message.content or "").strip()


def format_chunks_for_tool(chunks: List[dict]) -> str:
    """Format retrieved chunks for use as tool result (agent context)."""
    if not chunks:
        return "No relevant policy sections found."
    parts = []
    for i, c in enumerate(chunks, 1):
        meta = c.get("metadata") or {}
        page = meta.get("page", meta.get("source", ""))
        label = f"[{i}]" + (f" (page {page})" if page else "")
        parts.append(f"{label}\n{c['text']}")
    return "\n\n---\n\n".join(parts)


def query_policy_tool_fn(query: str) -> str:
    """Tool implementation: retrieve and return formatted policy chunks."""
    chunks = retrieve(query, k=5)
    return format_chunks_for_tool(chunks)


def register_query_policy_tool(tool_manager) -> None:
    """Register the query_policy local tool on the given ToolManager."""
    tool_manager.add_local_tool(
        name="query_policy",
        fn=query_policy_tool_fn,
        description=(
            "Retrieve relevant airline policy sections for a question. "
            "Call this when the user asks about refunds, cancellation, baggage, "
            "compensation, booking rules, or other policy. Use the returned context to answer."
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The user's policy-related question or keywords",
                },
            },
            "required": ["query"],
        },
    )
