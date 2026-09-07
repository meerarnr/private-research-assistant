"""Shared configuration for the Private Research Assistant pipeline."""

import os

# --- Qdrant ---
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = "research_assistant"

# --- Ollama models ---
OLLAMA_LLM_MODEL = "llama3.2:3b"
OLLAMA_EMBED_MODEL = "nomic-embed-text"
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# --- Chunking ---
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50

# --- Retrieval ---
TOP_K = 4

# --- Files ---
CORPUS_PATH = "corpus.json"
EVAL_SET_PATH = "eval_set.json"
EVAL_RESULTS_PATH = "eval_results.json"

# --- Content quality gate for ingestion: skip pages that scrape too thin to be useful ---
MIN_CONTENT_CHARS = 200
