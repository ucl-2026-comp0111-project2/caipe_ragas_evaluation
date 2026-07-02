from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Consolidated configuration settings using Pydantic"""
    model_config = SettingsConfigDict(
        env_prefix="",
        case_sensitive=False,
        extra="ignore"
    )

    # OpenAI / LiteLLM API Configs
    openai_api_key: str = Field(default="mock-openai-key")
    openai_endpoint: str = Field(default="http://localhost:4000/v1")
    openai_model_name: str = Field(default="qwen3.5-35b")
    embeddings_model: str = Field(default="bge-m3")

    # CAIPE KB Configs
    caipe_datasource_id: str = Field(default="enterprise_rag_bench", validation_alias="caipe_datasource_id")
    caipe_query_endpoint: str = Field(default="http://localhost:9446/v1/query", validation_alias="caipe_query_endpoint")
    caipe_supervisor_url: str = Field(default="http://localhost:8000", validation_alias="caipe_supervisor_url")
    caipe_oidc_token: Optional[str] = Field(default=None, validation_alias="caipe_oidc_token")
    ragas_datasource: str = Field(default="enterprise_rag_bench")
    questions_path: str = Field(default="", validation_alias="questions_path")
    ragas_limit: Optional[int] = Field(default=None, validation_alias="ragas_limit")

    # RAG Pipeline Configs
    rag_eval_top_k: int = Field(default=3)
    rag_eval_retrieval_only: bool = Field(default=False)
    rag_eval_generation_only: bool = Field(default=False)
    rag_eval_short_answer: bool = Field(default=False)
    limit_per_category: Optional[int] = Field(default=None)


# Singleton settings instance
settings = Settings()
