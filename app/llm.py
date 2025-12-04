from langchain_openai import ChatOpenAI
from app.config import Config

def get_llm(provider: str | None = None, model: str | None = None, temperature: float = 0):
    """Return an LLM instance for the requested provider and model."""

    settings = Config.get_llm_settings(provider)
    provider_name = settings["provider"]
    api_key = settings.get("api_key")
    base_url = settings.get("base_url")
    model_name = model or settings.get("model")

    if provider_name in ["openai", "deepseek", "github", "moonshot"]:
        if not api_key:
            raise ValueError(f"Missing API key for provider '{provider_name}'.")

        return ChatOpenAI(
            model=model_name or "gpt-3.5-turbo",
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
        )

    # Future extension for Gemini (Google) native API if needed
    # elif provider_name == "gemini":
    #     from langchain_google_genai import ChatGoogleGenerativeAI
    #     return ChatGoogleGenerativeAI(model=model_name, google_api_key=api_key)

    raise ValueError(f"Unsupported LLM provider: {provider_name}")
