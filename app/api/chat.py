import logging
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Request

from app.core.audit import utc_now_iso8601, write_audit_event
from app.core.dependencies import get_chat_service
from app.models.chat import ChatRequest, ChatResponse

router = APIRouter(tags=["chat"])
logger = logging.getLogger(__name__)


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest, raw_request: Request) -> ChatResponse:
    service = get_chat_service()
    request_id = raw_request.headers.get("x-request-id") or str(uuid4())
    try:
        if hasattr(service, "answer_question_with_trace"):
            response, trace = service.answer_question_with_trace(request.question, request.filters)
        else:
            response = service.answer_question(request.question, request.filters)
            trace = {"rewritten_query": request.question, "top_retrieval_count": len(response.sources), "detection_flags": []}
        audit_event = {
            "timestamp": utc_now_iso8601(),
            "request_id": request_id,
            "original_query": request.question,
            "rewritten_query": trace.get("rewritten_query", ""),
            "guardrail_status": response.guardrail_status.value,
            "confidence": response.confidence.value,
            "source_labels": [source.label for source in response.sources],
            "source_titles": [source.title for source in response.sources],
            "source_frameworks": [source.framework for source in response.sources],
            "top_retrieval_count": trace.get("top_retrieval_count", 0),
            "final_answer_length": len(response.answer),
            "refused_or_blocked": response.guardrail_status.value != "ok",
            "detection_flags": trace.get("detection_flags", []),
        }
        try:
            write_audit_event(audit_event)
        except Exception:  # pragma: no cover - audit failures are intentionally non-fatal
            logger.exception("Audit logging failed during /chat")
        return response
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - defensive fallback
        logger.exception("Unhandled /chat error")
        raise HTTPException(status_code=500, detail=str(exc) or "Internal server error") from exc
