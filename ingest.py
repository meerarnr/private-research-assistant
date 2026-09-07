"""
ingest.py — scrape a fixed list of source URLs via Firecrawl and save to corpus.json.

Run: python ingest.py
Requires FIRECRAWL_API_KEY in .env (see .env.example).

A scrape that fails outright (network/API error) or comes back too thin to be
useful (JS-rendered page, paywall, block) is logged as a warning and skipped —
one bad URL does not abort the whole ingestion run.
"""

import json
import logging
import os
import sys
from datetime import datetime, timezone

import requests
from dotenv import load_dotenv

from config import CORPUS_PATH, MIN_CONTENT_CHARS

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("ingest")

FIRECRAWL_SCRAPE_URL = "https://api.firecrawl.dev/v1/scrape"

# Fixed source list for this corpus: BABOK, Postman API testing, Scrum, RAG/LLM.
# Edit this list to point at the specific study material you want indexed.
SOURCES = [
    {"source_id": "babok_business_analysis", "url": "https://en.wikipedia.org/wiki/Business_analysis"},
    {"source_id": "babok_requirements_elicitation", "url": "https://en.wikipedia.org/wiki/Requirements_elicitation"},
    {"source_id": "scrum_guide", "url": "https://www.scrumguides.org/scrum-guide.html"},
    {"source_id": "scrum_wikipedia", "url": "https://en.wikipedia.org/wiki/Scrum_(software_development)"},
    {"source_id": "postman_test_scripts", "url": "https://learning.postman.com/docs/tests-and-scripts/write-scripts/test-scripts/"},
    {"source_id": "postman_collections_overview", "url": "https://learning.postman.com/docs/collections/collections-overview/"},
    {"source_id": "business_requirements_doc", "url": "https://en.wikipedia.org/wiki/Business_requirements"},
    {"source_id": "requirements_traceability", "url": "https://en.wikipedia.org/wiki/Requirements_traceability"},
    {"source_id": "acceptance_testing", "url": "https://en.wikipedia.org/wiki/Acceptance_testing"},
    {"source_id": "postman_using_collections", "url": "https://learning.postman.com/docs/collections/using-collections/"},
    {"source_id": "rag_wikipedia", "url": "https://en.wikipedia.org/wiki/Retrieval-augmented_generation"},
    {"source_id": "llm_wikipedia", "url": "https://en.wikipedia.org/wiki/Large_language_model"},
]


def scrape_url(url: str, api_key: str) -> dict | None:
    """Call Firecrawl's /v1/scrape endpoint for a single URL.

    Returns the response's "data" payload, or None if the scrape failed outright
    (network error or non-2xx/unsuccessful response).
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {"url": url, "formats": ["markdown"]}

    try:
        resp = requests.post(FIRECRAWL_SCRAPE_URL, headers=headers, json=payload, timeout=60)
    except requests.RequestException as e:
        logger.warning("Request to Firecrawl failed for %s: %s", url, e)
        return None

    if resp.status_code != 200:
        logger.warning("Firecrawl returned HTTP %s for %s: %s", resp.status_code, url, resp.text[:300])
        return None

    body = resp.json()
    if not body.get("success"):
        logger.warning("Firecrawl reported failure for %s: %s", url, body.get("error", "unknown error"))
        return None

    return body.get("data")


def main() -> None:
    api_key = os.getenv("FIRECRAWL_API_KEY")
    if not api_key:
        logger.error("FIRECRAWL_API_KEY is not set. Copy .env.example to .env and add your key.")
        sys.exit(1)

    corpus = []
    for source in SOURCES:
        source_id, url = source["source_id"], source["url"]
        logger.info("Scraping %s (%s)...", source_id, url)

        data = scrape_url(url, api_key)
        if data is None:
            logger.warning("Skipping %s: scrape request failed.", source_id)
            continue

        page_status = (data.get("metadata") or {}).get("statusCode")
        if page_status is not None and not (200 <= page_status < 300):
            logger.warning(
                "Skipping %s: target page returned HTTP %s (likely moved/removed). "
                "Check the URL is still correct: %s",
                source_id, page_status, url,
            )
            continue

        content = (data.get("markdown") or "").strip()
        if len(content) < MIN_CONTENT_CHARS:
            logger.warning(
                "Skipping %s: scraped content too thin (%d chars, need >= %d). "
                "Page may be JS-rendered, paywalled, or blocking scrapers.",
                source_id, len(content), MIN_CONTENT_CHARS,
            )
            continue

        title = (data.get("metadata") or {}).get("title", source_id)

        corpus.append({
            "source_id": source_id,
            "url": url,
            "title": title,
            "content": content,
            "scraped_at": datetime.now(timezone.utc).isoformat(),
        })
        logger.info("OK: %s (%d chars)", source_id, len(content))

    if not corpus:
        logger.error("No sources were successfully scraped. Aborting without writing corpus.json.")
        sys.exit(1)

    with open(CORPUS_PATH, "w", encoding="utf-8") as f:
        json.dump(corpus, f, indent=2, ensure_ascii=False)

    logger.info("Wrote %d/%d sources to %s", len(corpus), len(SOURCES), CORPUS_PATH)


if __name__ == "__main__":
    main()
