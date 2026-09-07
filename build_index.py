"""
build_index.py — embed corpus.json into Qdrant via LlamaIndex.

Run: python build_index.py
Requires: Ollama running locally with `nomic-embed-text` pulled, and Qdrant running
(see README for the docker run command).

Each corpus entry becomes one LlamaIndex Document carrying source_id/url/title as
metadata; LlamaIndex's SentenceSplitter then chunks it into nodes that all inherit
that same metadata, so every retrieved chunk still knows which document it came from.
"""

import json
import logging
import sys

from llama_index.core import Document, Settings, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    CORPUS_PATH,
    OLLAMA_BASE_URL,
    OLLAMA_EMBED_MODEL,
    QDRANT_COLLECTION,
    QDRANT_URL,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("build_index")


def main() -> None:
    try:
        with open(CORPUS_PATH, "r", encoding="utf-8") as f:
            corpus = json.load(f)
    except FileNotFoundError:
        logger.error("%s not found. Run ingest.py first.", CORPUS_PATH)
        sys.exit(1)

    if not corpus:
        logger.error("%s is empty. Nothing to index.", CORPUS_PATH)
        sys.exit(1)

    documents = [
        Document(
            text=item["content"],
            metadata={
                "source_id": item["source_id"],
                "url": item["url"],
                "title": item["title"],
            },
        )
        for item in corpus
    ]
    logger.info("Loaded %d documents from %s", len(documents), CORPUS_PATH)

    Settings.embed_model = OllamaEmbedding(model_name=OLLAMA_EMBED_MODEL, base_url=OLLAMA_BASE_URL)
    Settings.node_parser = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

    client = QdrantClient(url=QDRANT_URL)
    if client.collection_exists(QDRANT_COLLECTION):
        logger.info("Collection '%s' already exists, deleting to rebuild from scratch.", QDRANT_COLLECTION)
        client.delete_collection(QDRANT_COLLECTION)

    vector_store = QdrantVectorStore(client=client, collection_name=QDRANT_COLLECTION)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    VectorStoreIndex.from_documents(documents, storage_context=storage_context, show_progress=True)

    logger.info(
        "Indexed %d documents into Qdrant collection '%s' at %s",
        len(documents), QDRANT_COLLECTION, QDRANT_URL,
    )


if __name__ == "__main__":
    main()
