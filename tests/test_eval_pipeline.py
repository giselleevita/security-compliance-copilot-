import json
from pathlib import Path

from app.models.chat import ChatResponse
from app.models.source import SourceChunk, SourceResult
from evals.run_eval import load_questions, run_evaluation


class FakeRetrievalService:
    def retrieve(self, question: str) -> list[SourceChunk]:
        if "AWS" in question or "EU AI Act" in question:
            return []
        framework = "CISA" if "CISA" in question or "Secure by Design" in question else "NIST_AI_RMF"
        title = "CISA Secure by Design" if framework == "CISA" else "Artificial Intelligence Risk Management Framework"
        return [
            SourceChunk(
                chunk_id="1",
                text="The Govern function sets accountability and oversight for AI risk management.",
                source_id="src-1",
                title=title,
                url="https://example.com/source",
                publisher="NIST" if framework != "CISA" else "CISA",
                source_type="html",
                framework=framework,
                section="Govern",
                chunk_index=0,
                score=0.9,
                label="S1",
            )
        ]


class FakeChatService:
    def answer_question(self, question: str) -> ChatResponse:
        lowered = question.lower()
        if (
            "iso 27001" in lowered
            or "soc 2" in lowered
            or "pci dss" in lowered
            or "disregard" in lowered
            or "dan mode" in lowered
            or "initialization text" in lowered
            or "api key" in lowered
        ):
            return ChatResponse(
                answer="I cannot provide proprietary standards text.",
                sources=[],
                confidence="low",
                guardrail_status="refused",
            )
        if "cobit" in lowered or "aws instance" in lowered or "eu ai act" in lowered:
            return ChatResponse(
                answer="I do not have enough retrieved evidence to answer this reliably.",
                sources=[],
                confidence="low",
                guardrail_status="insufficient_context",
            )
        return ChatResponse(
            answer="The Govern function sets accountability and oversight for AI risk management [S1].",
            sources=[
                SourceResult(
                    label="S1",
                    title="AI RMF 1.0",
                    framework="NIST_AI_RMF",
                    url="https://example.com/ai-rmf",
                    score=0.91,
                )
            ],
            confidence="high",
            guardrail_status="ok",
        )


def test_eval_pipeline_runs_and_writes_results(tmp_path: Path) -> None:
    questions_path = Path("evals/questions.json")
    results_path = tmp_path / "results.json"

    questions = load_questions(questions_path)
    assert {item["category"] for item in questions} >= {
        "safe_answer",
        "retrieval_precision",
        "prompt_injection",
        "prompt_leak",
        "privacy_secret",
        "proprietary_text",
        "out_of_scope",
    }

    results, summary = run_evaluation(
        questions_path=questions_path,
        results_path=results_path,
        chat_service=FakeChatService(),
        retrieval_service=FakeRetrievalService(),
    )

    assert len(results) == len(questions)
    assert results_path.exists()
    assert summary["overall"]["total"] == len(questions)
    assert "mean_retrieval_mrr" in summary["overall"]


def test_eval_metrics_fail_when_expected_retrieval_is_missing(tmp_path: Path) -> None:
    questions = [
        {
            "id": "broken_retrieval",
            "category": "retrieval_precision",
            "question": "What does the Govern function cover?",
            "expected_guardrail_status": "ok",
            "expected_sources": [{"framework": "NIST_AI_RMF", "title_contains": "risk management framework"}],
            "min_retrieval_hit_rate": 1.0,
            "min_mrr": 1.0,
            "min_citation_precision": 1.0,
            "min_faithfulness": 0.5,
        }
    ]
    questions_path = tmp_path / "questions.json"
    questions_path.write_text(json.dumps(questions), encoding="utf-8")

    class BrokenRetrievalService(FakeRetrievalService):
        def retrieve(self, question: str) -> list[SourceChunk]:
            return []

    results, summary = run_evaluation(
        questions_path=questions_path,
        results_path=tmp_path / "results.json",
        chat_service=FakeChatService(),
        retrieval_service=BrokenRetrievalService(),
    )

    assert results[0]["retrieval_hit_rate"] == 0.0
    assert results[0]["passed"] is False
    assert summary["overall"]["passed"] == 0


def test_eval_pipeline_records_provider_errors_without_crashing(tmp_path: Path) -> None:
    questions = [
        {
            "id": "provider_error",
            "category": "safe_answer",
            "question": "What does the Govern function cover?",
            "expected_guardrail_status": "ok",
            "expected_sources": [{"framework": "NIST_AI_RMF", "title_contains": "risk management framework"}],
            "min_retrieval_hit_rate": 1.0,
            "min_mrr": 1.0,
            "min_citation_precision": 1.0,
            "min_faithfulness": 0.25,
        }
    ]
    questions_path = tmp_path / "questions.json"
    questions_path.write_text(json.dumps(questions), encoding="utf-8")

    class FailingChatService:
        def answer_question(self, question: str) -> ChatResponse:
            raise RuntimeError("provider quota exceeded")

    results, summary = run_evaluation(
        questions_path=questions_path,
        results_path=tmp_path / "results.json",
        chat_service=FailingChatService(),
        retrieval_service=FakeRetrievalService(),
    )

    assert results[0]["guardrail_status"] == "error"
    assert "provider quota exceeded" in results[0]["error"]
    assert results[0]["passed"] is False
    assert summary["overall"]["passed"] == 0
