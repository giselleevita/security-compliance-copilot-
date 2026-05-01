import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from app.ingestion.chunker import chunk_text
from app.ingestion.cleaning import clean_text, infer_framework, infer_title
from app.ingestion.loaders import list_supported_files, load_sidecar_metadata, load_text_from_file
from app.retrieval.embeddings import OpenAIEmbeddingClient
from app.retrieval.vectorstore import ChromaVectorStore


@dataclass
class IngestionResult:
    documents_processed: int
    chunks_stored: int


class IngestionPipeline:
    def __init__(
        self,
        raw_dir: Path,
        processed_dir: Path,
        vector_store: ChromaVectorStore,
        embedding_client: OpenAIEmbeddingClient,
        chunk_size: int = 1200,
        chunk_overlap: int = 200,
    ) -> None:
        self.raw_dir = raw_dir
        self.processed_dir = processed_dir
        self.vector_store = vector_store
        self.embedding_client = embedding_client
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def run(self) -> IngestionResult:
        files = list_supported_files(self.raw_dir)
        if not files:
            return IngestionResult(documents_processed=0, chunks_stored=0)

        total_chunks = 0
        for path in files:
            total_chunks += self._ingest_file(path)
        return IngestionResult(documents_processed=len(files), chunks_stored=total_chunks)

    def _ingest_file(self, path: Path) -> int:
        sidecar = load_sidecar_metadata(path)
        raw_text = load_text_from_file(path)
        cleaned = clean_text(raw_text)
        title = str(sidecar.get("title") or infer_title(cleaned, path))
        framework = str(sidecar.get("framework") or infer_framework(cleaned, path))
        source_url = str(sidecar.get("url") or path)
        publisher = str(sidecar.get("publisher") or "unknown")
        source_type = str(sidecar.get("source_type") or path.suffix.lower().lstrip("."))
        source_id = hashlib.sha1(str(path).encode("utf-8")).hexdigest()

        chunks = chunk_text(cleaned, chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        if not chunks:
            return 0

        documents: list[str] = []
        metadatas: list[dict] = []
        ids: list[str] = []

        for chunk in chunks:
            chunk_id = f"{source_id}:{chunk.chunk_index}"
            documents.append(chunk.text)
            metadatas.append(
                {
                    "source_id": source_id,
                    "title": title,
                    "url": source_url,
                    "publisher": publisher,
                    "source_type": source_type,
                    "framework": framework,
                    "section": chunk.section,
                    "chunk_index": chunk.chunk_index,
                }
            )
            ids.append(chunk_id)

        embeddings = self.embedding_client.embed_texts(documents)
        self.vector_store.upsert(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)
        self._write_processed_record(path=path, title=title, framework=framework, chunks=chunks)
        return len(chunks)

    def _write_processed_record(self, path: Path, title: str, framework: str, chunks: list) -> None:
        record = {
            "path": str(path),
            "title": title,
            "framework": framework,
            "chunk_count": len(chunks),
        }
        output_path = self.processed_dir / f"{path.stem}.json"
        output_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
