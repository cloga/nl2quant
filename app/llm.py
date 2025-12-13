from langchain_openai import ChatOpenAI
from app.config import Config
import time
from typing import Any

def get_llm(provider: str | None = None, model: str | None = None, temperature: float = 0):
    """Return an LLM instance for the requested provider and model with retry configuration."""

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
            max_retries=5,  # Enable retry for rate limit errors
            request_timeout=60,  # Increase timeout
        )

    # Future extension for Gemini (Google) native API if needed
    # elif provider_name == "gemini":
    #     from langchain_google_genai import ChatGoogleGenerativeAI
    #     return ChatGoogleGenerativeAI(model=model_name, google_api_key=api_key)

    raise ValueError(f"Unsupported LLM provider: {provider_name}")


def invoke_llm_with_retry(chain: Any, input_vars: dict, max_retries: int = 5, initial_delay: float = 1.0, provider_info: str = None) -> Any:
    """
    Invoke an LLM chain with exponential backoff retry logic.
    
    Args:
        chain: The LangChain chain to invoke
        input_vars: Input variables for the chain
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay in seconds before first retry
        provider_info: Optional string describing the provider/model (e.g. "openai/gpt-4")
        
    Returns:
        The response from the LLM chain
        
    Raises:
        Exception: If all retries are exhausted
    """
    delay = initial_delay
    last_exception = None
    
    last_error_message = None
    for attempt in range(max_retries):
        try:
            response = chain.invoke(input_vars)
            response._llm_provider = provider_info or getattr(chain, 'provider', None) or getattr(chain, 'model', None) or 'unknown'
            return response
        except Exception as e:
            last_exception = e
            error_message = str(e)
            last_error_message = error_message
            # Check if it's a rate limit or API error
            if ("429" in error_message or "RateLimitReached" in error_message or "rate limit" in error_message.lower() or "quota" in error_message.lower() or "github" in error_message.lower()):
                if attempt < max_retries - 1:
                    print(f"⚠️ Rate limit/API error (attempt {attempt + 1}/{max_retries}): {error_message}. Waiting {delay:.1f}s before retry...")
                    time.sleep(delay)
                    delay *= 2  # Exponential backoff
                    continue
                else:
                    # Fallback to DeepSeek if using GitHub model
                    from app.config import Config
                    provider = input_vars.get('provider', None) or getattr(chain, 'provider', None) or 'github'
                    if provider == 'github':
                        print("🔄 Fallback: Switching to DeepSeek due to GitHub model error/rate limit.")
                        # Try DeepSeek fallback
                        try:
                            from app.llm import get_llm
                            deepseek_llm = get_llm(provider='deepseek')
                            # Rebuild chain with DeepSeek LLM
                            # chain is likely a RunnableSequence (Prompt | LLM)
                            # We try to extract the prompt (first step) and create a new chain
                            if hasattr(chain, 'first'):
                                prompt = chain.first
                                chain_with_deepseek = prompt | deepseek_llm
                                response = chain_with_deepseek.invoke(input_vars)
                                response._llm_provider = 'deepseek'
                                response._llm_fallback_error = error_message
                                return response
                            else:
                                # Fallback for other chain types if possible, or raise original error
                                raise RuntimeError(f"Cannot swap LLM in chain type: {type(chain)}")
                        except Exception as deepseek_e:
                            raise RuntimeError(f"Both GitHub and DeepSeek LLMs failed. GitHub error: {error_message}; DeepSeek error: {deepseek_e}")
            # For other errors, raise immediately
            raise
    # If we've exhausted all retries, raise the last exception
    raise RuntimeError(f"LLM invocation failed after {max_retries} attempts. Last error: {last_error_message}")
