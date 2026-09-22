"""Typed System One failures. Every one of them means the policy must fail closed."""

from __future__ import annotations

__all__ = [
    "SystemOneAuthError",
    "SystemOneError",
    "SystemOneProtocolError",
    "SystemOneRateLimitError",
    "SystemOneTimeoutError",
    "SystemOneUnavailableError",
    "SystemOneUsageError",
]


class SystemOneError(RuntimeError):
    """Base class: the bundle call did not produce answers we may act on."""


class SystemOneAuthError(SystemOneError):
    """Missing, rejected or unconfigured credentials (HTTP 401/403)."""


class SystemOneUsageError(SystemOneError):
    """The request was malformed or the model id is unknown (HTTP 400/422)."""


class SystemOneRateLimitError(SystemOneError):
    """Rate limited or overloaded (HTTP 429/529).

    For the admission path a retry is the wrong answer: a late SPEAK is worse than a
    WAIT, so the client never retries and the caller fails closed.
    """

    def __init__(self, message: str, retry_after_s: float | None = None) -> None:
        super().__init__(message)
        self.retry_after_s = retry_after_s


class SystemOneTimeoutError(SystemOneError):
    """The call did not finish inside the configured budget."""


class SystemOneUnavailableError(SystemOneError):
    """Transport failure or a 5xx from the server."""


class SystemOneProtocolError(SystemOneError):
    """The response parsed as JSON but is not a valid answer to the questions we asked."""
