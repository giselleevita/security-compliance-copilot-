from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = Field(default="Security & Compliance Copilot", alias="APP_NAME")
    gemini_api_key: str = Field(default="", alias="GEMINI_API_KEY")
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    openai_base_url: str = Field(default="", alias="OPENAI_BASE_URL")
    openai_embedding_model: str = Field(
        default="text-embedding-3-small",
        alias="OPENAI_EMBEDDING_MODEL",
    )
    openai_chat_model: str = Field(default="gpt-4.1-mini", alias="OPENAI_CHAT_MODEL")
    api_host: str = Field(default="127.0.0.1", alias="API_HOST")
    api_port: int = Field(default=8000, alias="API_PORT")
    data_raw_dir: Path = Field(default=Path("data/raw"), alias="DATA_RAW_DIR")
    data_processed_dir: Path = Field(
        default=Path("data/processed"),
        alias="DATA_PROCESSED_DIR",
    )
    chroma_dir: Path = Field(default=Path("data/chroma"), alias="CHROMA_DIR")
    chroma_collection: str = Field(
        default="security_compliance_copilot",
        alias="CHROMA_COLLECTION",
    )
    top_k: int = Field(default=8, alias="TOP_K")
    rerank_k: int = Field(default=6, alias="RERANK_K")
    max_context_chars: int = Field(default=24000, alias="MAX_CONTEXT_CHARS")
    min_retrieval_score: float = Field(default=0.15, alias="MIN_RETRIEVAL_SCORE")
    min_good_results: int = Field(default=2, alias="MIN_GOOD_RESULTS")
    chunk_size: int = 1200
    chunk_overlap: int = 200

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.data_raw_dir.mkdir(parents=True, exist_ok=True)
    settings.data_processed_dir.mkdir(parents=True, exist_ok=True)
    settings.chroma_dir.mkdir(parents=True, exist_ok=True)
    return settings
