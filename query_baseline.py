"""
query_baseline.py — same retrieval/generation pipeline as query.py, but WITHOUT the
citation-enforcing system prompt or the citation-rejection check.

This exists purely as a comparison baseline for eval.py: it shows what a plain RAG
setup produces (or fails to produce, citation-wise) when nothing requires it to name
its sources. Status is always "OK" (or "NO_CONTEXT" if retrieval came up empty) —
there is no REJECTED_NO_CITATION path here by design.

Usage:
    from query_baseline import answer_question
    result = answer_question("What is a Sprint in Scrum?")
"""

import re
import sys

from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from config import (
    OLLAMA_BASE_URL,
    OLLAMA_EMBED_MODEL,
    OLLAMA_LLM_MODEL,
    QDRANT_COLLECTION,
    QDRANT_URL,
    TOP_K,
)

# Same pattern as query.py, used here only to measure whether an uncoached model
# spontaneously produces citation-shaped text (it essentially never does).
CITATION_PATTERN = re.compile(r"\[source_id:\s*([\w\-]+)\]")

SYSTEM_PROMPT = """You are a research assistant. Answer the question using the \
provided context excerpts as your primary source of information."""

_index = None


def _get_index() -> VectorStoreIndex:
    global _index
    if _index is not None:
        return _index

    Settings.llm = Ollama(model=OLLAMA_LLM_MODEL, base_url=OLLAMA_BASE_URL, request_timeout=120.0)
    Settings.embed_model = OllamaEmbedding(model_name=OLLAMA_EMBED_MODEL, base_url=OLLAMA_BASE_URL)

    client = QdrantClient(url=QDRANT_URL)
    vector_store = QdrantVectorStore(client=client, collection_name=QDRANT_COLLECTION)
    _index = VectorStoreIndex.from_vector_store(vector_store)
    return _index


def _build_context(nodes) -> str:
    blocks = []
    for node in nodes:
        source_id = node.metadata.get("source_id", "unknown")
        blocks.append(f"[source_id: {source_id}]\n{node.get_content()}")
    return "\n\n---\n\n".join(blocks)


def answer_question(question: str, top_k: int = TOP_K) -> dict:
    """Same return shape as query.answer_question, minus the rejection path."""
    index = _get_index()
    retriever = index.as_retriever(similarity_top_k=top_k)
    nodes = retriever.retrieve(question)

    retrieved_source_ids = sorted({n.metadata.get("source_id", "unknown") for n in nodes})

    if not nodes:
        return {
            "status": "NO_CONTEXT",
            "question": question,
            "retrieved_source_ids": [],
        }

    context = _build_context(nodes)
    messages = [
        ChatMessage(role=MessageRole.SYSTEM, content=SYSTEM_PROMPT),
        ChatMessage(role=MessageRole.USER, content=f"Context:\n\n{context}\n\nQuestion: {question}"),
    ]

    response = Settings.llm.chat(messages)
    raw_answer = str(response.message.content).strip()
    cited_source_ids = sorted(set(CITATION_PATTERN.findall(raw_answer)))

    return {
        "status": "OK",
        "question": question,
        "answer": raw_answer,
        "cited_source_ids": cited_source_ids,
        "retrieved_source_ids": retrieved_source_ids,
    }


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "What is a Sprint in Scrum?"
    result = answer_question(q)
    print(f"Status: {result['status']}")
    if "answer" in result:
        print(f"\nAnswer:\n{result['answer']}")
