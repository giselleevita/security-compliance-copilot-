import logging
from sentence_transformers import SentenceTransformer


logger = logging.getLogger(__name__)


class OpenAIEmbeddingClient:
    def __init__(self, api_key: str, model: str, base_url: str = "") -> None:
        self.model = model
        if api_key:
            from openai import OpenAI

            client_kwargs = {"api_key": api_key}
            if base_url:
                client_kwargs["base_url"] = base_url
            self.client = OpenAI(**client_kwargs)
        else:
            self.client = None

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not self.client:
            raise RuntimeError("OPENAI_API_KEY is required for ingestion and retrieval.")
        logger.info("Embedding %s text item(s) with %s", len(texts), self.model)
        response = self.client.embeddings.create(model=self.model, input=texts)
        return [item.embedding for item in response.data]

    def embed_query(self, text: str) -> list[float]:
        cleaned = " ".join(text.split())
        return self.embed_texts([cleaned])[0]


class LocalEmbeddingClient:
    def __init__(self, model: str = "all-MiniLM-L6-v2") -> None:
        self.model_name = model
        logger.info("Loading local embedding model: %s", self.model_name)
        self.model = SentenceTransformer(self.model_name)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        logger.info("Embedding %s text item(s) with %s", len(texts), self.model_name)
        embeddings = self.model.encode(texts, convert_to_tensor=False)
        return [emb.tolist() for emb in embeddings]

    def embed_query(self, text: str) -> list[float]:
        cleaned = " ".join(text.split())
        return self.embed_texts([cleaned])[0]
