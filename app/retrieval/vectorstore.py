import json
import logging
from collections.abc import Sequence
from pathlib import Path

from app.models.source import SourceChunk

logger = logging.getLogger(__name__)


class SidecarMetadataStore:
    def __init__(self, raw_dir: str | None = None) -> None:
        self.by_url: dict[str, dict] = {}
        self.by_title: dict[str, dict] = {}
        if not raw_dir:
            return

        for sidecar_path in Path(raw_dir).glob("*.metadata.json"):
            payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
            url = str(payload.get("url") or "").strip()
            title = str(payload.get("title") or "").strip()
            if url:
                self.by_url[url] = payload
            if title:
                self.by_title[title] = payload

    def lookup(self, url: str, title: str) -> dict:
        if url and url in self.by_url:
            return self.by_url[url]
        if title and title in self.by_title:
            return self.by_title[title]
        return {}


class ChromaVectorStore:
    def __init__(self, persist_directory: str, collection_name: str, raw_dir: str | None = None) -> None:
        import chromadb

        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(name=collection_name)
        self.sidecar_store = SidecarMetadataStore(raw_dir=raw_dir)

    def upsert(
        self,
        ids: Sequence[str],
        documents: Sequence[str],
        metadatas: Sequence[dict],
        embeddings: Sequence[Sequence[float]],
    ) -> None:
        self.collection.upsert(
            ids=list(ids),
            documents=list(documents),
            metadatas=list(metadatas),
            embeddings=[list(vector) for vector in embeddings],
        )

    def query(
        self,
        embedding: list[float],
        top_k: int,
        filters: dict[str, str] | None = None,
    ) -> list[SourceChunk]:
        result = self.collection.query(
            query_embeddings=[embedding],
            n_results=top_k,
            where=filters or None,
        )
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        ids = result.get("ids", [[]])[0]
        distances = result.get("distances", [[]])[0]

        chunks: list[SourceChunk] = []
        for chunk_id, document, metadata, distance in zip(ids, documents, metadatas, distances):
            merged_metadata = self._merge_metadata(metadata or {})
            score = 1 / (1 + float(distance))
            chunks.append(
                SourceChunk(
                    chunk_id=chunk_id,
                    text=document,
                    source_id=str(merged_metadata.get("source_id", "")),
                    title=str(merged_metadata.get("title", "Untitled Source")),
                    url=str(merged_metadata.get("url", "")),
                    publisher=str(merged_metadata.get("publisher", "unknown")),
                    source_type=str(merged_metadata.get("source_type", "unknown")),
                    framework=str(merged_metadata.get("framework", "general")),
                    section=str(merged_metadata.get("section", "Unknown Section")),
                    chunk_index=int(merged_metadata.get("chunk_index", 0)),
                    score=score,
                )
            )

        logger.info(
            "Retrieved %s chunks from Chroma (filters=%s, top_k=%s)",
            len(chunks),
            filters,
            top_k,
        )
        return chunks

    def count(self) -> int:
        return self.collection.count()

    def _merge_metadata(self, metadata: dict) -> dict:
        sidecar = self.sidecar_store.lookup(
            url=str(metadata.get("url") or ""),
            title=str(metadata.get("title") or ""),
        )
        merged = dict(sidecar)
        for key, value in metadata.items():
            if value not in (None, ""):
                merged[key] = value
        return merged
