from pathlib import Path

from app.models.chat import ChatResponse
from app.models.source import SourceResult
from evals.run_eval import load_questions, run_evaluation


class FakeRetrievalService:
    def retrieve(self, question: str) -> list[object]:
        if "ISO 27001" in question:
            return []
        return [object(), object()]


class FakeChatService:
    def answer_question(self, question: str) -> ChatResponse:
        lowered = question.lower()
        if "iso 27001" in lowered or "soc 2" in lowered:
            return ChatResponse(
                answer="I cannot provide proprietary standards text.",
                sources=[],
                confidence="low",
                guardrail_status="refused",
            )
        if "cobit" in lowered or "aws instance" in lowered or "eu ai act" in lowered or "pci dss" in lowered:
            return ChatResponse(
                answer="I do not have enough retrieved evidence to answer this reliably.",
                sources=[],
                confidence="low",
                guardrail_status="insufficient_context",
            )
        return ChatResponse(
            answer="NIST guidance emphasizes governance and risk management [S1].",
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
    assert len(questions) >= 30

    results, summary = run_evaluation(
        questions_path=questions_path,
        results_path=results_path,
        chat_service=FakeChatService(),
        retrieval_service=FakeRetrievalService(),
    )

    assert len(results) == len(questions)
    assert results_path.exists()
    assert summary["overall"]["total"] == len(questions)
