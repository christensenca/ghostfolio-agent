from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Ghostfolio API
    ghostfolio_api_url: str = "http://localhost:3333"
    ghostfolio_access_token: str = ""  # Security token from Ghostfolio Settings
    ghostfolio_jwt: str = ""  # Direct JWT (advanced, skips token exchange)

    # LLM (via OpenRouter)
    openrouter_api_key: str = ""
    llm_model: str = "anthropic/claude-haiku-4.5"
    openrouter_base_url: str = "https://openrouter.ai/api/v1"

    # LangSmith observability (optional)
    langsmith_api_key: str = ""
    langsmith_project: str = ""

    # Agent config
    max_tool_steps: int = 5
    max_response_seconds: int = 90
    conversation_ttl_seconds: int = 3600

    # Database
    db_path: str = "ghostfolio_agent.db"

    model_config = {"env_file": ".env", "env_prefix": "", "case_sensitive": False}


settings = Settings()
