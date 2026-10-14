from functools import lru_cache
import sys
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Lahore Traffic Intelligence AI"
    api_prefix: str = "/api/v1"
    groq_api_key: str = ""
    groq_model: str = "openai/gpt-oss-20b"
    # The RAG extraction/vision path can run on a different provider from the chat
    # path. Groq's free tier allows only 8k tokens per minute and 200k per day,
    # which is far below the cost of extracting the PDF corpus; Gemini's quota is
    # large enough to finish it.
    rag_llm_provider: str = "groq"
    gemini_api_key: str = ""
    gemini_model: str = "gemini-3.8-flash"
    gemini_vision_model: str = "gemini-3.8-flash"
    gemini_base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai/"
    rag_extraction_model: str = "openai/gpt-oss-120b"
    groq_vision_model: str = "qwen/qwen3.8-27b"
    rag_enabled: bool = True
    rag_parser: str = "mineru"
    rag_query_mode: str = "hybrid"
    # Groq reasoning models can spend a whole completion budget reasoning and return
    # empty content. Asking for low effort keeps the JSON extraction on the output side.
    rag_llm_reasoning_effort: str = "low"
    rag_llm_max_retries: int = 3
    # Groq charges an uncapped completion allowance against its per-minute token
    # limit, so these bounds keep each request inside that ceiling. Gemini has a
    # large context and quota and does not need them tightened; scripts/ingest_
    # rag_documents.py raises them when RAG_LLM_PROVIDER is not groq.
    rag_llm_max_completion_tokens: int = 1600
    rag_chunk_token_size: int = 900
    # Extraction requests are large enough that several in flight exhaust Groq's
    # per-minute budget, so bulk loading is sequential by default.
    rag_llm_max_async: int = 1
    backend_cors_origins: str = "http://localhost:3000,http://127.0.0.1:3000,http://localhost:3001,http://127.0.0.1:3001"
    max_upload_mb: int = 8
    semantic_worker_url: str = "http://127.0.0.1:8765"
    semantic_python: str = "python"
    semantic_startup_timeout: float = 600
    image_embeddings_enabled: bool = True
    image_similarity_threshold: float = 0.22
    image_photo_similarity_threshold: float = 0.18
    image_similarity_margin: float = 0.015

    model_config = SettingsConfigDict(
        env_file=("../.env", ".env"), env_file_encoding="utf-8", extra="ignore"
    )

    @property
    def project_root(self) -> Path:
        return Path(__file__).resolve().parents[3]

    @property
    def data_dir(self) -> Path:
        return self.project_root / "data"

    @property
    def rag_storage(self) -> Path:
        return self.project_root / "backend" / "rag_storage"

    @property
    def embedding_dimension(self) -> int:
        # Read only the header; never migrate or re-embed the existing corpus on startup.
        import re
        path = self.rag_storage / "vdb_chunks.json"
        if path.is_file():
            with path.open(encoding="utf-8") as handle:
                match = re.search(r'"embedding_dim"\s*:\s*(\d+)', handle.read(256))
            if match:
                return int(match[1])
        return 384

    @property
    def upload_dir(self) -> Path:
        return self.project_root / "backend" / "storage" / "uploads"

    @property
    def image_embedding_model_path(self) -> Path:
        return self.project_root / "backend" / "models" / "image_embeddings" / "RN50.pt"

    @property
    def image_index_path(self) -> Path:
        return self.project_root / "backend" / "storage" / "image_embeddings.npz"

    @property
    def database_path(self) -> Path:
        return self.project_root / "backend" / "storage" / "app.db"

    @property
    def semantic_python_executable(self) -> str:
        if self.semantic_python != "python":
            return self.semantic_python
        # Use the same managed environment as the API, including its locked dependencies.
        return sys.executable

    @property
    def cors_origins(self) -> list[str]:
        return [item.strip() for item in self.backend_cors_origins.split(",") if item.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
