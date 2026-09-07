"""
query.py — citation-enforced query layer over the indexed corpus.

Every factual claim in the answer must end in a [source_id: <id>] tag citing the
retrieved chunk it came from. The check is whole-answer, single-pass: if the raw
model output contains zero citation tags anywhere, this returns a
REJECTED_NO_CITATION status (with the raw text attached for debugging) instead of
silently returning an uncited answer. There is no per-sentence enforcement and no
retry — a 3B local model isn't reliable enough for either to be worth the added
complexity in an MVP.

Usage:
    from query import answer_question
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

CITATION_PATTERN = re.compile(r"\[source_id:\s*([\w\-]+)\]")

SYSTEM_PROMPT = """You are a research assistant answering questions using ONLY the \
provided context excerpts. Each excerpt is labeled with a [source_id: <id>] tag.

Rules:
1. Answer using only information in the context below. Do not use outside knowledge.
2. Every factual claim you make MUST end with a citation tag in the exact format \
[source_id: <id>] — square brackets, exactly as shown, using the source_id of the \
excerpt that supports the claim. Do not use parentheses or any other format.
3. If the context does not contain enough information to answer, say so plainly \
instead of guessing, and still cite the source_id(s) you drew on if any.
4. Never fabricate a source_id that is not listed in the context.

Example of the required format:
The Sprint is time-boxed to one month or less. [source_id: scrum_guide]
"""

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
    """Retrieve context, generate a cited answer, and enforce citation presence.

    Returns a dict with at least a "status" key:
      - "OK": answer text is present and contains >=1 citation tag.
      - "REJECTED_NO_CITATION": model produced no citation tag anywhere; the raw
        uncited answer is still included under "raw_answer" for debugging.
      - "NO_CONTEXT": retrieval returned nothing to answer from.
    """
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

    if not cited_source_ids:
        return {
            "status": "REJECTED_NO_CITATION",
            "question": question,
            "raw_answer": raw_answer,
            "retrieved_source_ids": retrieved_source_ids,
        }

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
    if result["status"] == "OK":
        print(f"\nAnswer:\n{result['answer']}")
    elif result["status"] == "REJECTED_NO_CITATION":
        print(f"\nRaw (uncited) answer, for debugging:\n{result['raw_answer']}")
