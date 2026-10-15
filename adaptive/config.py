"""Validated environment settings for Adaptive Agentic RAG V1."""

from functools import lru_cache
from typing import Literal
from urllib.parse import parse_qs, urlparse, urlunparse

from pydantic import Field, SecretStr, ValidationInfo, field_validator
from pydantic_core import PydanticCustomError
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Application
    app_env: Literal["development", "staging", "production"] = Field(
        default="development", alias="APP_ENV"
    )
    app_host: str = Field(default="0.0.0.0", alias="APP_HOST")  # noqa: S104
    app_port: int = Field(default=8000, alias="APP_PORT")
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO", alias="LOG_LEVEL"
    )
    log_format: Literal["json", "console"] = Field(default="json", alias="LOG_FORMAT")

    # Database
    database_url: str = Field(alias="DATABASE_URL")
    database_pool_size: int = Field(default=10, alias="DATABASE_POOL_SIZE", ge=1, le=100)
    database_max_overflow: int = Field(default=20, alias="DATABASE_MAX_OVERFLOW", ge=0, le=100)

    @field_validator("database_url", mode="before")
    @classmethod
    def normalize_database_url(cls, v: str, info: ValidationInfo) -> str:  # noqa: ARG003
        parsed = urlparse(v)
        query = parse_qs(parsed.query)
        query.pop("channel_binding", None)
        new_query = "&".join(f"{k}={val}" for k, vals in query.items() for val in vals)
        return urlunparse(parsed._replace(query=new_query))

    # OpenSearch
    opensearch_url: str = Field(alias="OPENSEARCH_URL")
    opensearch_username: str = Field(default="", alias="OPENSEARCH_USERNAME")
    opensearch_password: str = Field(default="", alias="OPENSEARCH_PASSWORD")
    opensearch_verify_certs: bool = Field(default=False, alias="OPENSEARCH_VERIFY_CERTS")
    opensearch_index_prefix: str = Field(default="adaptive_rag", alias="OPENSEARCH_INDEX_PREFIX")

    # Embedding Provider
    embedding_provider: Literal["openai", "azure", "local", "fake"] = Field(
        default="openai", alias="EMBEDDING_PROVIDER"
    )
    embedding_model: str = Field(default="text-embedding-3-small", alias="EMBEDDING_MODEL")
    embedding_api_key: str = Field(default="", alias="EMBEDDING_API_KEY")
    embedding_api_base: str = Field(default="https://api.openai.com/v1", alias="EMBEDDING_API_BASE")
    embedding_batch_size: int = Field(default=100, alias="EMBEDDING_BATCH_SIZE", ge=1, le=1000)
    embedding_dimensions: int = Field(default=1536, alias="EMBEDDING_DIMENSIONS", ge=1, le=8192)
    embedding_timeout_seconds: int = Field(
        default=30, alias="EMBEDDING_TIMEOUT_SECONDS", ge=1, le=300
    )
    embedding_max_retries: int = Field(default=3, alias="EMBEDDING_MAX_RETRIES", ge=0, le=10)

    # Generation Provider
    generation_provider: Literal["openai", "azure", "local", "fake"] = Field(
        default="openai", alias="GENERATION_PROVIDER"
    )
    generation_model: str = Field(default="gpt-4o-mini", alias="GENERATION_MODEL")
    generation_api_key: str = Field(default="", alias="GENERATION_API_KEY")
    generation_api_base: str = Field(
        default="https://api.openai.com/v1", alias="GENERATION_API_BASE"
    )
    generation_timeout_seconds: int = Field(
        default=60, alias="GENERATION_TIMEOUT_SECONDS", ge=1, le=600
    )
    generation_max_retries: int = Field(default=3, alias="GENERATION_MAX_RETRIES", ge=0, le=10)

    # Optional offline Ragas evaluation judge (credentials are required only when invoked).
    ragas_judge_api_key: str = Field(default="", alias="RAGAS_JUDGE_API_KEY")
    ragas_judge_api_base: str = Field(
        default="https://api.groq.com/openai/v1", alias="RAGAS_JUDGE_API_BASE"
    )
    ragas_judge_model: str = Field(
        default="openai/gpt-oss-20b", alias="RAGAS_JUDGE_MODEL"
    )

    # NVIDIA NIMS Fallback
    nims_base_url: str = Field(default="https://integrate.api.nvidia.com/v1", alias="NIMS_BASE_URL")
    nims_api_key: str = Field(default="", alias="NIMS_API_KEY")
    # Optional: NIMS-specific model overrides for fallback
    nims_embedding_model: str = Field(
        default="nvidia/llama-nemotron-embed-vl-1b-v2", alias="NIMS_EMBEDDING_MODEL"
    )
    nims_generation_model: str = Field(
        default="meta/llama-3.1-8b-instruct", alias="NIMS_GENERATION_MODEL"
    )
    nims_rerank_model: str = Field(
        default="nvidia/llama-nemotron-rerank-vl-1b-v2", alias="NIMS_RERANK_MODEL"
    )

    # Router
    router_provider: Literal["llm", "embedding", "policy"] = Field(
        default="llm", alias="ROUTER_PROVIDER"
    )
    router_model: str = Field(default="gpt-4o-mini", alias="ROUTER_MODEL")
    router_confidence_threshold: float = Field(
        default=0.7, alias="ROUTER_CONFIDENCE_THRESHOLD", ge=0.0, le=1.0
    )
    router_escalation_enabled: bool = Field(default=True, alias="ROUTER_ESCALATION_ENABLED")

    # Semantic Cache
    cache_enabled: bool = Field(default=True, alias="CACHE_ENABLED")
    cache_similarity_threshold: float = Field(
        default=0.95, alias="CACHE_SIMILARITY_THRESHOLD", ge=0.0, le=1.0
    )
    cache_ttl_seconds: int = Field(default=3600, alias="CACHE_TTL_SECONDS", ge=1, le=86400 * 30)
    cache_max_entries_per_tenant: int = Field(
        default=10000, alias="CACHE_MAX_ENTRIES_PER_TENANT", ge=100, le=1000000
    )

    # Web Tool
    web_allowlist: str = Field(
        default="api.github.com,raw.githubusercontent.com,en.wikipedia.org", alias="WEB_ALLOWLIST"
    )
    web_timeout_seconds: int = Field(default=10, alias="WEB_TIMEOUT_SECONDS", ge=1, le=120)
    web_max_retries: int = Field(default=2, alias="WEB_MAX_RETRIES", ge=0, le=10)
    web_max_content_length: int = Field(
        default=50000, alias="WEB_MAX_CONTENT_LENGTH", ge=1000, le=1000000
    )

    @property
    def web_allowlist_domains(self) -> list[str]:
        """Parse comma-separated allowlist into list of domains."""
        return [d.strip() for d in self.web_allowlist.split(",") if d.strip()]

    # SQL Tool
    sql_statement_timeout_ms: int = Field(
        default=5000, alias="SQL_STATEMENT_TIMEOUT_MS", ge=100, le=120000
    )
    sql_max_rows: int = Field(default=1000, alias="SQL_MAX_ROWS", ge=1, le=100000)
    sql_allowed_schemas: str = Field(default="public", alias="SQL_ALLOWED_SCHEMAS")

    @property
    def sql_allowed_schemas_list(self) -> list[str]:
        return [s.strip() for s in self.sql_allowed_schemas.split(",") if s.strip()]

    # Redis
    redis_host: str = Field(default="localhost", alias="REDIS_HOST")
    redis_port: int = Field(default=6379, alias="REDIS_PORT")
    redis_username: str = Field(default="", alias="REDIS_USERNAME")
    redis_password: str = Field(default="", alias="REDIS_PASSWORD")
    redis_ssl: bool = Field(default=False, alias="REDIS_SSL")

    # Guardrails
    guardrail_input_enabled: bool = Field(default=True, alias="GUARDRAIL_INPUT_ENABLED")
    guardrail_output_enabled: bool = Field(default=True, alias="GUARDRAIL_OUTPUT_ENABLED")
    guardrail_pii_enabled: bool = Field(default=True, alias="GUARDRAIL_PII_ENABLED")
    guardrail_groundedness_threshold: float = Field(
        default=0.7, alias="GUARDRAIL_GROUNDEDNESS_THRESHOLD", ge=0.0, le=1.0
    )
    guardrail_citation_coverage_threshold: float = Field(
        default=0.8, alias="GUARDRAIL_CITATION_COVERAGE_THRESHOLD", ge=0.0, le=1.0
    )

    # Agent Budgets
    agent_max_steps: int = Field(default=10, alias="AGENT_MAX_STEPS", ge=1, le=100)
    agent_max_tokens: int = Field(default=8000, alias="AGENT_MAX_TOKENS", ge=100, le=100000)
    agent_max_cost_usd: float = Field(default=0.50, alias="AGENT_MAX_COST_USD", ge=0.0, le=100.0)
    agent_max_wall_time_seconds: int = Field(
        default=120, alias="AGENT_MAX_WALL_TIME_SECONDS", ge=10, le=3600
    )

    phoenix_enabled: bool = Field(default=True, alias="PHOENIX_ENABLED")
    phoenix_capture_content: bool = Field(default=False, alias="PHOENIX_CAPTURE_CONTENT")
    phoenix_endpoint: str = Field(
        default="http://localhost:6006/v1/traces", alias="PHOENIX_ENDPOINT"
    )
    phoenix_api_base_url: str = Field(
        default="http://localhost:6006", alias="PHOENIX_API_BASE_URL"
    )
    phoenix_eval_groq_api_base: str = Field(default="", alias="PHOENIX_EVAL_GROQ_API_BASE")
    phoenix_eval_groq_model: str = Field(default="", alias="PHOENIX_EVAL_GROQ_MODEL")
    phoenix_eval_groq_api_key: SecretStr = Field(
        default=SecretStr(""), alias="PHOENIX_EVAL_GROQ_API_KEY"
    )
    phoenix_project_name: str = Field(default="adaptive-rag", alias="PHOENIX_PROJECT_NAME")
    phoenix_redact_prompts: bool = Field(default=True, alias="PHOENIX_REDACT_PROMPTS")
    phoenix_redact_documents: bool = Field(default=True, alias="PHOENIX_REDACT_DOCUMENTS")
    phoenix_redact_credentials: bool = Field(default=True, alias="PHOENIX_REDACT_CREDENTIALS")

    # Authentication
    auth_mode: Literal["development", "production"] = Field(
        default="development", alias="AUTH_MODE"
    )
    auth_dev_tenant_id: str = Field(default="acme", alias="AUTH_DEV_TENANT_ID")
    auth_dev_subject_id: str = Field(default="user-1", alias="AUTH_DEV_SUBJECT_ID")
    auth_dev_acl: str = Field(default="support,admin", alias="AUTH_DEV_ACL")
    auth_jwt_secret: str = Field(default="dev-secret", alias="AUTH_JWT_SECRET")
    auth_jwt_algorithm: str = Field(default="HS256", alias="AUTH_JWT_ALGORITHM")
    auth_jwt_audience: str = Field(default="adaptive-rag", alias="AUTH_JWT_AUDIENCE")
    auth_jwt_issuer: str = Field(default="adaptive-rag-idp", alias="AUTH_JWT_ISSUER")

    # BM25 and cross-encoder scores are not comparable until calibrated on an eval set.
    retrieval_grader_threshold: float | None = Field(
        default=None, alias="RETRIEVAL_GRADER_THRESHOLD", ge=0.0
    )

    @property
    def auth_dev_acl_set(self) -> frozenset[str]:
        return frozenset(a.strip() for a in self.auth_dev_acl.split(",") if a.strip())

    # Ingestion
    ingestion_max_file_size_mb: int = Field(
        default=50, alias="INGESTION_MAX_FILE_SIZE_MB", ge=1, le=1000
    )
    ingestion_temp_dir: str = Field(
        default="/tmp/adaptive_rag_ingestion",  # noqa: S108
        alias="INGESTION_TEMP_DIR",  # noqa: S108
    )
    ingestion_chunk_max_tokens: int = Field(
        default=512, alias="INGESTION_CHUNK_MAX_TOKENS", ge=64, le=8192
    )
    ingestion_chunk_overlap_tokens: int = Field(
        default=50, alias="INGESTION_CHUNK_OVERLAP_TOKENS", ge=0, le=1024
    )
    ingestion_max_attempts: int = Field(default=5, alias="INGESTION_MAX_ATTEMPTS", ge=1, le=100)
    ingestion_retry_base_seconds: int = Field(
        default=2, alias="INGESTION_RETRY_BASE_SECONDS", ge=1, le=3600
    )
    ingestion_retry_max_seconds: int = Field(
        default=300, alias="INGESTION_RETRY_MAX_SECONDS", ge=1, le=86400
    )
    ingestion_job_lease_seconds: int = Field(
        default=300, alias="INGESTION_JOB_LEASE_SECONDS", ge=1, le=86400
    )
    ingestion_poll_interval_seconds: float = Field(
        default=1.0, alias="INGESTION_POLL_INTERVAL_SECONDS", gt=0.0, le=300.0
    )
    ingestion_heartbeat_interval_seconds: float | None = Field(
        default=None, alias="INGESTION_HEARTBEAT_INTERVAL_SECONDS", gt=0.0, le=3600.0
    )

    @field_validator("ingestion_chunk_overlap_tokens")
    @classmethod
    def overlap_less_than_max(cls, v: int, info: ValidationInfo) -> int:
        max_tokens = info.data.get("ingestion_chunk_max_tokens", 512)
        if v >= max_tokens:
            raise PydanticCustomError("value_error", "overlap_tokens must be less than max_tokens")
        return v

    @field_validator("ingestion_retry_max_seconds")
    @classmethod
    def retry_max_at_least_base(cls, v: int, info: ValidationInfo) -> int:
        base_seconds = info.data.get("ingestion_retry_base_seconds", 2)
        if v < base_seconds:
            raise PydanticCustomError(
                "value_error",
                "ingestion_retry_max_seconds must be >= ingestion_retry_base_seconds",
            )
        return v

    # Retrieval
    retrieval_bm25_k: int = Field(default=50, alias="RETRIEVAL_BM25_K", ge=1, le=500)
    retrieval_dense_k: int = Field(default=50, alias="RETRIEVAL_DENSE_K", ge=1, le=500)
    retrieval_fusion_k: int = Field(default=20, alias="RETRIEVAL_FUSION_K", ge=1, le=200)
    retrieval_rerank_enabled: bool = Field(default=False, alias="RETRIEVAL_RERANK_ENABLED")
    retrieval_rerank_model: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2", alias="RETRIEVAL_RERANK_MODEL"
    )
    retrieval_context_token_budget: int = Field(
        default=4000, alias="RETRIEVAL_CONTEXT_TOKEN_BUDGET", ge=500, le=32000
    )

    @field_validator("retrieval_fusion_k")
    @classmethod
    def fusion_k_not_exceed_sources(cls, v: int, info: ValidationInfo) -> int:
        bm25_k = info.data.get("retrieval_bm25_k", 50)
        dense_k = info.data.get("retrieval_dense_k", 50)
        max_k = max(bm25_k, dense_k)
        if v > max_k:
            raise PydanticCustomError(
                "value_error", "fusion_k must not exceed max(bm25_k, dense_k)"
            )
        return v


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()  # type: ignore[call-arg]
