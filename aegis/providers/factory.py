from functools import lru_cache

from aegis import config
from aegis.providers.mock_provider import MockProvider


# maxsize=4, not 1: api_key is now a real argument (the Streamlit sidebar's
# bring-your-own-key field), so a few distinct keys used across a session can each
# get their own cached provider instead of constantly evicting one another.
@lru_cache(maxsize=4)
def get_text_provider(api_key: str | None = None):
    if api_key or config.has_live_text_provider():
        from aegis.providers.anthropic_provider import AnthropicProvider

        return AnthropicProvider(api_key=api_key)
    return MockProvider()


@lru_cache(maxsize=4)
def get_vision_provider(api_key: str | None = None):
    if api_key or config.has_live_text_provider():
        from aegis.providers.anthropic_provider import AnthropicProvider

        return AnthropicProvider(api_key=api_key)
    return MockProvider()


@lru_cache(maxsize=1)
def get_image_provider():
    if config.IMAGE_PROVIDER == "stability" and config.STABILITY_API_KEY:
        from aegis.providers.stability_provider import StabilityProvider

        return StabilityProvider()
    if config.IMAGE_PROVIDER == "gemini" and config.GEMINI_API_KEY:
        from aegis.providers.gemini_provider import GeminiProvider

        return GeminiProvider()
    if config.IMAGE_PROVIDER == "pollinations":
        from aegis.providers.pollinations_provider import PollinationsProvider

        return PollinationsProvider()
    return MockProvider()


def is_live_mode(api_key: str | None = None) -> bool:
    return bool(api_key) or config.has_live_text_provider()
