import json
from typing import Any

from fastapi.testclient import TestClient

from app.main import app
from app.models.chat import ChatResponse, ConfidenceLevel, GuardrailStatus
from app.models.source import SourceResult


def make_source(
    label: str,
    *,
    title: str = "Doc 1",
    framework: str = "NIST_AI_RMF",
    url: str = "https://example.com/1",
    score: float = 0.91,
) -> SourceResult:
    return SourceResult(label=label, title=title, framework=framework, url=url, score=score)


class RecordingService:
    def __init__(self, response: ChatResponse, *, rewritten_query: str | None = None) -> None:
        self.response = response
        self.rewritten_query = rewritten_query
        self.calls: list[dict[str, Any]] = []

    def answer_question_with_trace(self, question: str, filters: dict[str, str] | None = None):
        self.calls.append({"question": question, "filters": filters})
        return (
            self.response,
            {
                "rewritten_query": self.rewritten_query or question,
                "top_retrieval_count": len(self.response.sources),
                "detection_flags": [],
            },
        )


def test_empty_query_is_rejected_by_schema() -> None:
    client = TestClient(app)
    response = client.post("/chat", json={"question": ""})
    assert response.status_code == 422


def test_too_short_query_is_rejected_by_schema() -> None:
    client = TestClient(app)
    response = client.post("/chat", json={"question": "hi"})
    assert response.status_code == 422


def test_logging_failure_does_not_break_chat(monkeypatch) -> None:
    service = RecordingService(
        ChatResponse(
            answer="Governance allocates accountability [S1].",
            sources=[make_source("S1")],
            confidence=ConfidenceLevel.HIGH,
            guardrail_status=GuardrailStatus.OK,
        )
    )
    monkeypatch.setattr("app.api.chat.get_chat_service", lambda: service)
    monkeypatch.setattr(
        "app.api.chat.write_audit_event",
        lambda payload: (_ for _ in ()).throw(OSError("disk full")),
    )
    client = TestClient(app)

    response = client.post("/chat", json={"question": "What is the purpose of NIST AI RMF?"})

    assert response.status_code == 200
    payload = ChatResponse.model_validate(response.json())
    assert payload.guardrail_status is GuardrailStatus.OK


def test_audit_log_contains_required_fields_and_matches_response(monkeypatch) -> None:
    events: list[dict[str, Any]] = []
    service = RecordingService(
        ChatResponse(
            answer="The Govern function sets accountability and oversight [S1][S2].",
            sources=[
                make_source("S1", title="AI RMF 1.0", framework="NIST_AI_RMF"),
                make_source("S2", title="AI RMF Playbook", framework="NIST_AI_RMF", url="https://example.com/2"),
            ],
            confidence=ConfidenceLevel.MEDIUM,
            guardrail_status=GuardrailStatus.OK,
        ),
        rewritten_query="What is the role of the Govern function in NIST AI RMF?",
    )
    monkeypatch.setattr("app.api.chat.get_chat_service", lambda: service)
    monkeypatch.setattr("app.api.chat.write_audit_event", lambda payload: events.append(payload))
    client = TestClient(app)

    response = client.post("/chat", json={"question": "What is the role of the Govern function in AI risk management?"})

    assert response.status_code == 200
    body = ChatResponse.model_validate(response.json())
    assert len(events) == 1
    event = events[0]
    assert set(event) >= {
        "timestamp",
        "request_id",
        "original_query",
        "rewritten_query",
        "guardrail_status",
        "confidence",
        "source_labels",
        "source_titles",
        "source_frameworks",
        "top_retrieval_count",
        "final_answer_length",
        "refused_or_blocked",
        "detection_flags",
    }
    assert event["guardrail_status"] == body.guardrail_status.value
    assert event["confidence"] == body.confidence.value
    assert event["source_labels"] == [source.label for source in body.sources]
    assert event["source_titles"] == [source.title for source in body.sources]
    assert event["source_frameworks"] == [source.framework for source in body.sources]
    assert event["top_retrieval_count"] == len(body.sources)
    assert event["final_answer_length"] == len(body.answer)
    assert event["refused_or_blocked"] is False
    assert event["rewritten_query"].startswith("What is the role of the Govern")


def test_audit_log_one_json_line_per_request(tmp_path) -> None:
    from app.core.audit import write_audit_event

    path = tmp_path / "audit.jsonl"
    write_audit_event({"request_id": "req-1"}, log_path=path)
    write_audit_event({"request_id": "req-2"}, log_path=path)

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert [json.loads(line)["request_id"] for line in lines] == ["req-1", "req-2"]


def test_repeatability_same_query_produces_stable_source_selection(monkeypatch) -> None:
    service = RecordingService(
        ChatResponse(
            answer="NIST CSF 2.0 emphasizes governance, outcomes, and continuous improvement [S1][S2].",
            sources=[
                make_source("S1", title="NIST CSF 2.0 Core", framework="NIST_CSF"),
                make_source("S2", title="NIST CSF 2.0 Overview", framework="NIST_CSF", url="https://example.com/2"),
            ],
            confidence=ConfidenceLevel.HIGH,
            guardrail_status=GuardrailStatus.OK,
        )
    )
    monkeypatch.setattr("app.api.chat.get_chat_service", lambda: service)
    client = TestClient(app)

    seen = []
    for _ in range(5):
        response = client.post("/chat", json={"question": "What does NIST CSF 2.0 emphasize?"})
        assert response.status_code == 200
        payload = ChatResponse.model_validate(response.json())
        seen.append(
            (
                payload.guardrail_status.value,
                payload.confidence.value,
                tuple((source.label, source.title, source.framework) for source in payload.sources),
            )
        )

    assert len(set(seen)) == 1


def test_long_unicode_and_multiline_queries_keep_valid_response_shape(monkeypatch) -> None:
    service = RecordingService(
        ChatResponse(
            answer="CISA describes Secure by Design as shifting security responsibility toward vendors [S1].",
            sources=[make_source("S1", title="CISA Secure by Design", framework="CISA")],
            confidence=ConfidenceLevel.MEDIUM,
            guardrail_status=GuardrailStatus.OK,
        )
    )
    monkeypatch.setattr("app.api.chat.get_chat_service", lambda: service)
    client = TestClient(app)
    long_query = (
        "What is secure by design in CISA guidance?\n\n"
        "| key | value |\n| --- | --- |\n| a | b |\n"
        "```json\n{\"framework\": \"CISA\", \"topic\": \"secure by design\"}\n```\n"
        + "安全性؟ " * 500
        + "!!! ??? :::"
    )

    response = client.post("/chat", json={"question": long_query})

    assert response.status_code == 200
    validated = ChatResponse.model_validate(response.json())
    assert validated.guardrail_status is GuardrailStatus.OK
    assert validated.sources[0].framework == "CISA"
