"""The remote text generator: prompts, an OpenAI-compatible client, and a scripted stand-in.

This package is never on the admission path. System One decides whether to speak; the
ChatLLM only writes what is said and the sentence a memory is stored as.
"""

from __future__ import annotations

from rt_agent.llm.client import (
    BASE_URL_ENV,
    DEFAULT_API_KEY_ENV,
    DEFAULT_BASE_URL,
    DISABLE_THINKING_ENV,
    MODEL_ENV,
    NON_THINKING_EXTRA_BODY,
    OpenAICompatibleLLM,
    parse_completion,
    strip_reasoning,
)
from rt_agent.llm.errors import (
    LLMAuthError,
    LLMConfigurationError,
    LLMEmptyResponseError,
    LLMError,
    LLMProtocolError,
    LLMRateLimitError,
    LLMTimeoutError,
    LLMUnavailableError,
    LLMUsageError,
)
from rt_agent.llm.prompts import (
    MAX_MEMORY_CHARS,
    MAX_REPLY_SENTENCES,
    MEMORY_PROMPT_VERSION,
    MEMORY_SUMMARY_SYSTEM,
    REPLY_PROMPT_VERSION,
    memory_summary_messages,
    reply_messages,
    reply_system_prompt,
    sanitize,
)
from rt_agent.llm.reply import (
    MAX_REPLY_CHARS,
    ReplyGenerator,
    clean_reply,
    split_sentences,
)
from rt_agent.llm.scripted import (
    DEFAULT_REPLY,
    RULES_SCHEMA_VERSION,
    ScriptedLLM,
    ScriptedRule,
)

__all__ = [
    "BASE_URL_ENV",
    "DEFAULT_API_KEY_ENV",
    "DEFAULT_BASE_URL",
    "DEFAULT_REPLY",
    "DISABLE_THINKING_ENV",
    "MAX_MEMORY_CHARS",
    "MAX_REPLY_CHARS",
    "MAX_REPLY_SENTENCES",
    "MEMORY_PROMPT_VERSION",
    "MEMORY_SUMMARY_SYSTEM",
    "MODEL_ENV",
    "NON_THINKING_EXTRA_BODY",
    "REPLY_PROMPT_VERSION",
    "RULES_SCHEMA_VERSION",
    "LLMAuthError",
    "LLMConfigurationError",
    "LLMEmptyResponseError",
    "LLMError",
    "LLMProtocolError",
    "LLMRateLimitError",
    "LLMTimeoutError",
    "LLMUnavailableError",
    "LLMUsageError",
    "OpenAICompatibleLLM",
    "ReplyGenerator",
    "ScriptedLLM",
    "ScriptedRule",
    "clean_reply",
    "memory_summary_messages",
    "parse_completion",
    "reply_messages",
    "reply_system_prompt",
    "sanitize",
    "split_sentences",
    "strip_reasoning",
]
