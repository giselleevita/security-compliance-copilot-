import json
from pathlib import Path

from app.core.dependencies import get_chat_service, get_retrieval_service


def load_questions(questions_path: Path) -> list[dict]:
    return json.loads(questions_path.read_text(encoding="utf-8"))


def evaluate_result(expected_behavior: str, result: dict) -> bool:
    if expected_behavior == "answer":
        return result["guardrail_status"] == "ok" and bool(result["sources"])
    if expected_behavior == "insufficient_context":
        return result["guardrail_status"] == "insufficient_context"
    if expected_behavior == "refuse":
        return result["guardrail_status"] == "refused"
    return False


def evaluate_questions(
    questions: list[dict],
    chat_service,
    retrieval_service,
) -> list[dict]:
    results: list[dict] = []
    for item in questions:
        retrieved = retrieval_service.retrieve(item["question"])
        response = chat_service.answer_question(item["question"])
        result = {
            "id": item["id"],
            "category": item["category"],
            "question": item["question"],
            "expected_behavior": item["expected_behavior"],
            "retrieved_chunk_count": len(retrieved),
            "guardrail_status": response.guardrail_status,
            "confidence": response.confidence,
            "source_frameworks": [source.framework for source in response.sources],
            "answer_length": len(response.answer),
            "sources": [source.model_dump() for source in response.sources],
            "answer": response.answer,
        }
        result["passed"] = evaluate_result(item["expected_behavior"], result)
        results.append(result)
    return results


def summarize_results(results: list[dict]) -> dict:
    summary: dict[str, dict] = {}
    for behavior in ("answer", "insufficient_context", "refuse"):
        matching = [result for result in results if result["expected_behavior"] == behavior]
        passed = sum(1 for result in matching if result["passed"])
        total = len(matching)
        summary[behavior] = {
            "passed": passed,
            "total": total,
            "success_rate": round((passed / total) * 100, 1) if total else 0.0,
        }
    summary["overall"] = {
        "passed": sum(1 for result in results if result["passed"]),
        "total": len(results),
    }
    return summary


def write_results(results: list[dict], results_path: Path) -> None:
    results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")


def print_summary(summary: dict) -> None:
    print("Behavior                Passed/Total  Success Rate")
    print("-----------------------------------------------")
    for behavior in ("answer", "insufficient_context", "refuse"):
        row = summary[behavior]
        print(f"{behavior:<22} {row['passed']:>2}/{row['total']:<9} {row['success_rate']:>6.1f}%")
    overall = summary["overall"]
    print("-----------------------------------------------")
    print(f"{'overall':<22} {overall['passed']:>2}/{overall['total']:<9}")


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
