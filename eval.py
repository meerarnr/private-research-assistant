"""
eval.py — runs eval_set.json against both query.py (citation-enforced) and
query_baseline.py (no enforcement), and reports three metrics for each:

  - answer_rate: fraction of questions that produced a real answer (status "OK"
    with non-empty text). For query.py, a REJECTED_NO_CITATION counts as a
    non-answer here — that's the whole point of the comparison.
  - citation_precision: among answered questions, the fraction where AT LEAST ONE
    cited source_id overlaps with that question's expected_source_ids
    (any-overlap hit rate, not strict per-citation precision).
  - content_match_rate: among answered questions, the fraction where EVERY
    keyword in expected_answer_contains appears (case-insensitive substring
    match) in the answer text.

Full per-question results (including raw text for rejected/uncited answers) are
written to eval_results.json.

Run: python eval.py
Requires build_index.py to have been run already.
"""

import json
import logging

import query
import query_baseline
from config import EVAL_RESULTS_PATH, EVAL_SET_PATH

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("eval")


def _content_match(answer: str, keywords: list) -> bool:
    if not keywords:
        return True
    lowered = answer.lower()
    return all(kw.lower() in lowered for kw in keywords)


def _run_system(system_name: str, module, eval_set: list) -> dict:
    per_question = []
    answered = 0
    citation_hits = 0
    content_hits = 0

    for item in eval_set:
        result = module.answer_question(item["question"])
        status = result["status"]
        answer_text = result.get("answer", "")
        is_answered = status == "OK" and bool(answer_text.strip())

        citation_hit = False
        content_hit = False

        if is_answered:
            answered += 1
            cited = set(result.get("cited_source_ids", []))
            expected = set(item.get("expected_source_ids", []))
            citation_hit = bool(cited & expected)
            content_hit = _content_match(answer_text, item.get("expected_answer_contains", []))
            if citation_hit:
                citation_hits += 1
            if content_hit:
                content_hits += 1

        per_question.append({
            "id": item["id"],
            "question": item["question"],
            "status": status,
            "answer": answer_text or result.get("raw_answer", ""),
            "cited_source_ids": result.get("cited_source_ids", []),
            "expected_source_ids": item.get("expected_source_ids", []),
            "citation_hit": citation_hit,
            "content_hit": content_hit,
        })

        logger.info("[%s] %s -> %s", system_name, item["id"], status)

    n = len(eval_set)
    summary = {
        "system": system_name,
        "num_questions": n,
        "answer_rate": answered / n if n else 0.0,
        "citation_precision": citation_hits / answered if answered else 0.0,
        "content_match_rate": content_hits / answered if answered else 0.0,
    }
    return {"summary": summary, "per_question": per_question}


def main() -> None:
    with open(EVAL_SET_PATH, "r", encoding="utf-8") as f:
        eval_set = json.load(f)

    if not eval_set:
        logger.error("%s is empty.", EVAL_SET_PATH)
        return

    results = {
        "citation_enforced": _run_system("citation_enforced", query, eval_set),
        "baseline": _run_system("baseline", query_baseline, eval_set),
    }

    with open(EVAL_RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    for name, r in results.items():
        s = r["summary"]
        logger.info(
            "%s: answer_rate=%.2f citation_precision=%.2f content_match_rate=%.2f",
            name, s["answer_rate"], s["citation_precision"], s["content_match_rate"],
        )
    logger.info("Full results written to %s", EVAL_RESULTS_PATH)


if __name__ == "__main__":
    main()
