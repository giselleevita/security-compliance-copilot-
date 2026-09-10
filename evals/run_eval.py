import json
import math
import re
from pathlib import Path
from typing import Any

from app.core.dependencies import get_chat_service, get_retrieval_service
from app.generation.context_builder import build_context
from app.generation.service import GroundingValidator

CITATION_PATTERN = re.compile(r"\[(S\d+)\]")


def load_questions(questions_path: Path) -> list[dict]:
    return json.loads(questions_path.read_text(encoding="utf-8"))


def enum_value(value: Any) -> str:
    return getattr(value, "value", value)


def matches_expected_source(chunk: Any, expected: dict) -> bool:
    framework = str(expected.get("framework") or "")
    title_contains = str(expected.get("title_contains") or "").lower()
    source_id = str(expected.get("source_id") or "")
    chunk_id = str(expected.get("chunk_id") or "")
    if framework and getattr(chunk, "framework", "") != framework:
        return False
    if title_contains and title_contains not in getattr(chunk, "title", "").lower():
        return False
    if source_id and getattr(chunk, "source_id", "") != source_id:
        return False
    if chunk_id and getattr(chunk, "chunk_id", "") != chunk_id:
        return False
    return bool(framework or title_contains or source_id or chunk_id)


def reciprocal_rank(chunks: list[Any], expected_sources: list[dict]) -> float:
    if not expected_sources:
        return 0.0
    for index, chunk in enumerate(chunks, start=1):
        if any(matches_expected_source(chunk, expected) for expected in expected_sources):
            return round(1 / index, 4)
    return 0.0


def ndcg_at_k(chunks: list[Any], expected_sources: list[dict], k: int = 5) -> float:
    if not expected_sources:
        return 0.0
    gains = [
        1.0 if any(matches_expected_source(chunk, expected) for expected in expected_sources) else 0.0
        for chunk in chunks[:k]
    ]
    dcg = sum(gain / math.log2(index + 2) for index, gain in enumerate(gains))
    ideal_hits = min(len(expected_sources), k)
    ideal_dcg = sum(1.0 / math.log2(index + 2) for index in range(ideal_hits))
    return round(dcg / ideal_dcg, 4) if ideal_dcg else 0.0


def retrieval_hit(chunks: list[Any], expected_sources: list[dict]) -> float:
    if not expected_sources:
        return 0.0
    hits = sum(1 for expected in expected_sources if any(matches_expected_source(chunk, expected) for chunk in chunks))
    return round(hits / len(expected_sources), 4)


def citation_precision(answer: str, source_labels: set[str]) -> float:
    labels = CITATION_PATTERN.findall(answer)
    if not labels:
        return 0.0
    valid = sum(1 for label in labels if label in source_labels)
    return round(valid / len(labels), 4)


def evaluate_questions(
    questions: list[dict],
    chat_service,
    retrieval_service,
) -> list[dict]:
    results: list[dict] = []
    grounding_validator = GroundingValidator()
    for item in questions:
        retrieved = retrieval_service.retrieve(item["question"])
        response_error = None
        try:
            response = chat_service.answer_question(item["question"])
        except Exception as exc:  # noqa: BLE001 - eval harness records any failure
            response = None
            response_error = f"{type(exc).__name__}: {exc}"
        if response is None:
            expected_sources = item.get("expected_sources", [])
            result = {
                "id": item["id"],
                "category": item["category"],
                "question": item["question"],
                "expected_guardrail_status": item["expected_guardrail_status"],
                "guardrail_status": "error",
                "retrieved_chunk_count": len(retrieved),
                "retrieval_hit_rate": retrieval_hit(retrieved, expected_sources),
                "retrieval_mrr": reciprocal_rank(retrieved, expected_sources),
                "retrieval_ndcg_at_5": ndcg_at_k(retrieved, expected_sources, k=5),
                "citation_precision": 0.0,
                "faithfulness": 0.0,
                "refusal_correct": False,
                "source_frameworks": [],
                "answer_length": 0,
                "sources": [],
                "answer": "",
                "error": response_error,
                "passed": False,
            }
            results.append(result)
            continue
        sources = getattr(response, "sources", [])
        source_labels = {source.label for source in sources}
        expected_sources = item.get("expected_sources", [])
        context_package = build_context(retrieved, max_chars=24000)
        status = enum_value(response.guardrail_status)
        expected_status = item["expected_guardrail_status"]
        result = {
            "id": item["id"],
            "category": item["category"],
            "question": item["question"],
            "expected_guardrail_status": expected_status,
            "guardrail_status": status,
            "retrieved_chunk_count": len(retrieved),
            "retrieval_hit_rate": retrieval_hit(retrieved, expected_sources),
            "retrieval_mrr": reciprocal_rank(retrieved, expected_sources),
            "retrieval_ndcg_at_5": ndcg_at_k(retrieved, expected_sources, k=5),
            "citation_precision": citation_precision(response.answer, source_labels),
            "faithfulness": grounding_validator.faithfulness_score(response.answer, context_package),
            "refusal_correct": status == expected_status if expected_status in {"refused", "insufficient_context"} else None,
            "source_frameworks": [source.framework for source in sources],
            "answer_length": len(response.answer),
            "sources": [source.model_dump() for source in sources],
            "answer": response.answer,
            "error": response_error,
        }
        result["passed"] = result_passed(item, result)
        results.append(result)
    return results


def result_passed(item: dict, result: dict) -> bool:
    expected_status = item["expected_guardrail_status"]
    if result["guardrail_status"] != expected_status:
        return False
    if expected_status == "ok":
        return (
            result["retrieval_hit_rate"] >= float(item.get("min_retrieval_hit_rate", 0.0))
            and result["retrieval_mrr"] >= float(item.get("min_mrr", 0.0))
            and result["citation_precision"] >= float(item.get("min_citation_precision", 1.0))
            and result["faithfulness"] >= float(item.get("min_faithfulness", 0.5))
        )
    return True


def summarize_results(results: list[dict]) -> dict:
    summary: dict[str, Any] = {}
    by_category: dict[str, list[dict]] = {}
    for result in results:
        by_category.setdefault(result["category"], []).append(result)
    for category, rows in sorted(by_category.items()):
        summary[category] = summarize_rows(rows)
    summary["overall"] = summarize_rows(results)
    return summary


def summarize_rows(rows: list[dict]) -> dict:
    if not rows:
        return {"total": 0, "passed": 0, "pass_rate": 0.0}
    numeric_fields = ["retrieval_hit_rate", "retrieval_mrr", "retrieval_ndcg_at_5", "citation_precision", "faithfulness"]
    payload: dict[str, Any] = {
        "total": len(rows),
        "passed": sum(1 for row in rows if row["passed"]),
    }
    payload["pass_rate"] = round(payload["passed"] / payload["total"], 4)
    for field in numeric_fields:
        payload[f"mean_{field}"] = round(sum(float(row[field]) for row in rows) / len(rows), 4)
    refusal_rows = [row for row in rows if row["refusal_correct"] is not None]
    payload["refusal_accuracy"] = (
        round(sum(1 for row in refusal_rows if row["refusal_correct"]) / len(refusal_rows), 4)
        if refusal_rows
        else None
    )
    return payload


def write_results(results: list[dict], results_path: Path) -> None:
    results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")


def print_summary(summary: dict) -> None:
    print("Category                 Passed/Total  Pass Rate  MRR    Faithful")
    print("------------------------------------------------------------------")
    for category, row in summary.items():
        if category == "overall":
            continue
        print(
            f"{category:<24} {row['passed']:>2}/{row['total']:<9} "
            f"{row['pass_rate']:>8.2f} {row['mean_retrieval_mrr']:>6.2f} "
            f"{row['mean_faithfulness']:>8.2f}"
        )
    overall = summary["overall"]
    print("------------------------------------------------------------------")
    print(f"{'overall':<24} {overall['passed']:>2}/{overall['total']:<9} {overall['pass_rate']:>8.2f}")


def run_evaluation(
    questions_path: Path | None = None,
    results_path: Path | None = None,
    chat_service=None,
    retrieval_service=None,
) -> tuple[list[dict], dict]:
    resolved_questions_path = questions_path or Path("evals/questions.json")
    resolved_results_path = results_path or Path("evals/results.json")
    questions = load_questions(resolved_questions_path)
    chat_service = chat_service or get_chat_service()
    retrieval_service = retrieval_service or get_retrieval_service()

    results = evaluate_questions(questions, chat_service=chat_service, retrieval_service=retrieval_service)
    summary = summarize_results(results)
    write_results(results, resolved_results_path)
    return results, summary


def main() -> None:
    _, summary = run_evaluation()
    print_summary(summary)
    print("Detailed results written to evals/results.json")


if __name__ == "__main__":
    main()
