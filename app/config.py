from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    DATABASE_URL: str = "sqlite+aiosqlite:///./aic_demo.db"
    SUPABASE_URL: str = ""
    SUPABASE_SERVICE_KEY: str = ""
    SUPABASE_JWT_SECRET: str = ""
    DEEPSEEK_API_KEY: str = ""
    ENVIRONMENT: str = "development"

    # LLM settings
    LLM_MODEL: str = "deepseek-chat"
    LLM_TEMPERATURE: float = 0.7
    LLM_MAX_TOKENS: int = 150

    # Intervention settings
    INTERVENTION_COOLDOWN_MINUTES: int = 120
    BOOST_COOLDOWN_MINUTES: int = 60
    MICRO_SURVEY_DELAY_MINUTES: int = 15
    MICRO_SURVEY_WINDOW_MINUTES: int = 30
    MAX_INTERVENTION_CHARS: int = 280
    SOCIAL_OUTREACH_DAILY_CAP: int = 2

    # Bayesian shrinkage settings
    THRESHOLD_FLOOR: float = 3.0
    THRESHOLD_CEILING: float = 6.0
    INDIVIDUALIZATION_TARGET_N: int = 21
    CV_CHECK_MIN_SURVEYS: int = 9
    CV_THRESHOLD: float = 0.50
    SD_MULTIPLIER: float = 1.0

    # Survey settings
    EMA_SURVEYS_PER_DAY: int = 3
    MIN_INTER_SURVEY_MINUTES: int = 120
    EMA_RESPONSE_WINDOW_MINUTES: int = 60
    EOD_RESPONSE_WINDOW_MINUTES: int = 120
    SILENCE_DURATION_MINUTES: int = 30

    # Content screening
    COSINE_SIMILARITY_THRESHOLD: float = 0.85
    MAX_LLM_RETRIES: int = 2

    # Convergence monitoring
    KL_CONVERGENCE_THRESHOLD: float = 0.10

    model_config = {"env_file": ".env", "extra": "ignore"}


settings = Settings()
