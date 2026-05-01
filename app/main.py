import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from app.api.chat import router as chat_router
from app.api.health import router as health_router
from app.api.ingest import router as ingest_router
from app.core.config import get_settings
from app.core.logging import configure_logging
from app.core.middleware import RequestLoggingMiddleware
from app.core.dependencies import (
    get_embedding_client,
    get_vector_store,
    get_ingestion_pipeline,
)

configure_logging()
logger = logging.getLogger(__name__)

app = FastAPI(title=get_settings().app_name)
app.add_middleware(RequestLoggingMiddleware)
app.include_router(health_router)
app.include_router(chat_router)
app.include_router(ingest_router)


@app.on_event("startup")
def startup_event() -> None:
    logger.info("Starting up: pre-loading embedding model and vector store")
    try:
        logger.info("Pre-loading embedding model...")
        embedding_client = get_embedding_client()
        logger.info("Embedding model loaded successfully")

        logger.info("Checking vector store...")
        vector_store = get_vector_store()
        count = vector_store.count()
        logger.info("Vector store has %s indexed chunks", count)

        if count == 0:
            logger.info("Vector store is empty, starting document ingestion...")
            pipeline = get_ingestion_pipeline()
            result = pipeline.run()
            logger.info(
                "Ingestion complete: %s documents processed, %s chunks stored",
                result.documents_processed,
                result.chunks_stored,
            )
        else:
            logger.info("Vector store already populated, skipping ingestion")

        logger.info("Startup complete, application ready")
    except Exception as e:
        logger.exception("Startup initialization failed")
        raise


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    html_path = Path(__file__).parent / "frontend" / "index.html"
    return html_path.read_text(encoding="utf-8")
