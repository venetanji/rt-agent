"""Typed ChatLLM failures.

The reply LLM is never on the admission path, so a failure here is not a safety
problem: it means the robot stays quiet, or a memory is not written. Every failure is
still typed, because the harness logs the reason and the memory writer turns it into a
``failed`` outcome instead of letting it escape into the event loop.
"""

from __future__ import annotations

__all__ = [
    "LLMAuthError",
    "LLMConfigurationError",
    "LLMEmptyResponseError",
    "LLMError",
    "LLMProtocolError",
    "LLMRateLimitError",
    "LLMTimeoutError",
    "LLMUnavailableError",
    "LLMUsageError",
]


class LLMError(RuntimeError):
    """Base class: the completion did not produce text we may use."""


class LLMConfigurationError(LLMError):
    """The client is not configured well enough to make a call (no model id, no base url)."""


class LLMAuthError(LLMError):
    """Missing, rejected or unconfigured credentials (HTTP 401/403)."""


class LLMUsageError(LLMError):
    """The request was malformed or the model id is unknown (HTTP 400/404/422)."""


class LLMRateLimitError(LLMError):
    """Rate limited or overloaded (HTTP 429/529). The client never retries."""

    def __init__(self, message: str, retry_after_s: float | None = None) -> None:
        super().__init__(message)
        self.retry_after_s = retry_after_s


class LLMTimeoutError(LLMError):
    """The completion did not finish inside the configured budget."""


class LLMUnavailableError(LLMError):
    """Transport failure or a 5xx from the server."""


class LLMProtocolError(LLMError):
    """The response parsed as JSON but is not a usable chat completion."""


class LLMEmptyResponseError(LLMProtocolError):
    """The model returned nothing usable once reasoning and markup were stripped."""
