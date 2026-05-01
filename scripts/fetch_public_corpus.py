import json
import ssl
from datetime import UTC, datetime
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
MANIFEST_PATH = RAW_DIR / "source_manifest.json"
FETCH_RESULTS_PATH = RAW_DIR / "fetch_results.json"
USER_AGENT = "SecurityComplianceCopilot/1.0 (+local ingestion fetcher)"
REQUEST_TIMEOUT_SECONDS = 20


def load_manifest() -> list[dict]:
    payload = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    return payload["accepted_documents"]


def build_output_path(document: dict) -> Path:
    return RAW_DIR / document["suggested_filename"]


def build_sidecar_path(output_path: Path) -> Path:
    return Path(f"{output_path}.metadata.json")


def fetch_bytes(url: str) -> bytes:
    request = Request(url, headers={"User-Agent": USER_AGENT})
    context = ssl.create_default_context()
    with urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS, context=context) as response:
        return response.read()


def write_sidecar(document: dict, output_path: Path) -> None:
    metadata = {
        "title": document["title"],
        "url": document["url"],
        "publisher": document["publisher"],
        "framework": document["framework"],
        "source_type": output_path.suffix.lower().lstrip("."),
        "document_type": document["document_type"],
        "license_status": document["license_status"],
        "priority": document["priority"],
        "content_format": document["content_format"],
        "fetched_at": datetime.now(UTC).isoformat(),
        "local_path": str(output_path),
    }
    build_sidecar_path(output_path).write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def fetch_document(document: dict) -> dict:
    output_path = build_output_path(document)
    record = {
        "title": document["title"],
        "url": document["url"],
        "output_path": str(output_path),
        "status": "pending",
    }
    if output_path.exists() and build_sidecar_path(output_path).exists():
        record["status"] = "already_present"
        record["bytes"] = output_path.stat().st_size
        return record
    try:
        content = fetch_bytes(document["url"])
        output_path.write_bytes(content)
        write_sidecar(document, output_path)
        record["status"] = "fetched"
        record["bytes"] = len(content)
    except (HTTPError, URLError, TimeoutError, ssl.SSLError, Exception) as exc:
        record["status"] = "failed"
        record["error"] = str(exc)
    return record


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    documents = load_manifest()
    results = []
    total = len(documents)
    for index, document in enumerate(documents, start=1):
        result = fetch_document(document)
        results.append(result)
        print(f"[{index}/{total}] {result['status']}: {document['suggested_filename']}", flush=True)
    FETCH_RESULTS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")

    fetched = sum(1 for result in results if result["status"] in {"fetched", "already_present"})
    failed = sum(1 for result in results if result["status"] == "failed")
    print(f"Fetched: {fetched}")
    print(f"Failed: {failed}")
    print(f"Results: {FETCH_RESULTS_PATH}")


if __name__ == "__main__":
    main()
