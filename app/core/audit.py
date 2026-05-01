import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)
AUDIT_LOG_PATH = Path("logs/audit.jsonl")


def utc_now_iso8601() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def write_audit_event(event: dict[str, Any], log_path: Path = AUDIT_LOG_PATH) -> None:
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=True) + "\n")
    except Exception:  # pragma: no cover - audit failures are intentionally non-fatal
        logger.exception("Audit logging failed")
