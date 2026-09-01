"""A provider-agnostic LLM client with a console entry point.

The design keeps the core request logic (``llm_bot.client.LLMClient``) separate
from the user interface so a web interface can be added later by reusing the
same service layer.
"""

from llm_bot.config import LLMConfig
from llm_bot.client import (
    LLMClient,
    LLMError,
    LLMRequestError,
    LLMRetryExhaustedError,
    TokenProvider,
)
from llm_bot.gigachat import GigaChatTokenProvider, build_gigachat_client

__all__ = [
    "LLMConfig",
    "LLMClient",
    "LLMError",
    "LLMRequestError",
    "LLMRetryExhaustedError",
    "TokenProvider",
    "GigaChatTokenProvider",
    "build_gigachat_client",
]

__version__ = "0.1.0"