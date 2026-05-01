from fastapi import APIRouter, HTTPException

from app.core.dependencies import get_ingestion_pipeline

router = APIRouter(tags=["ingest"])


@router.post("/ingest")
def ingest() -> dict:
    pipeline = get_ingestion_pipeline()
    try:
        result = pipeline.run()
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {
        "ingested_documents": result.documents_processed,
        "stored_chunks": result.chunks_stored,
    }
