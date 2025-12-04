import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Central configuration helper."""

    SUPPORTED_LLM_PROVIDERS = ["openai", "deepseek", "github", "moonshot", "gemini"]
    PROVIDER_DEFAULT_MODELS = {
        "openai": "gpt-4o-mini",
        "deepseek": "deepseek-chat",
        "github": "gpt-4o-mini",
        "moonshot": "moonshot-v1-32k",
        "gemini": "gemini-1.5-pro",
    }
    PROVIDER_DEFAULT_BASE_URL = {
        "openai": "https://api.openai.com/v1",
        "deepseek": "https://api.deepseek.com/v1",
        "github": "https://models.inference.ai.azure.com",
        "moonshot": "https://api.moonshot.cn/v1",
        "gemini": "https://generativelanguage.googleapis.com/v1beta",
    }

    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "deepseek").lower()

    # Data Settings
    TUSHARE_TOKEN = os.getenv("TUSHARE_TOKEN")

    _LLM_SETTINGS_CACHE = {}

    @classmethod
    def get_llm_settings(cls, provider: str | None = None) -> dict:
        """Return configuration for the requested LLM provider."""
        provider = (provider or cls.LLM_PROVIDER).lower()

        if provider not in cls.SUPPORTED_LLM_PROVIDERS:
            supported = ", ".join(cls.SUPPORTED_LLM_PROVIDERS)
            raise ValueError(f"Unsupported LLM provider '{provider}'. Supported providers: {supported}.")

        if provider in cls._LLM_SETTINGS_CACHE:
            return cls._LLM_SETTINGS_CACHE[provider]

        prefix = f"LLM_{provider.upper()}"

        settings = {
            "provider": provider,
            "api_key": os.getenv(f"{prefix}_API_KEY"),
            "base_url": os.getenv(f"{prefix}_BASE_URL") or cls.PROVIDER_DEFAULT_BASE_URL.get(provider),
            "model": os.getenv(f"{prefix}_MODEL_NAME") or cls.PROVIDER_DEFAULT_MODELS.get(provider),
        }

        cls._LLM_SETTINGS_CACHE[provider] = settings
        return settings

    @classmethod
    def validate(cls):
        if not cls.TUSHARE_TOKEN:
            print("Warning: TUSHARE_TOKEN is not set. Data fetching may fail.")

        settings = cls.get_llm_settings()
        if not settings.get("api_key"):
            prefix = f"LLM_{settings['provider'].upper()}_API_KEY"
            print(f"Warning: Missing API key for provider '{settings['provider']}'. Set {prefix} in your .env file.")
