"""``ReplyGenerator`` — prompt, call, and clean up the robot's spoken reply.

The face and the TTS take plain spoken text: no markdown, no stage directions, no
speaker labels, no leaked chat-template tags, and a length the room will sit through.
The prompt asks for all of that, and this module enforces it afterwards, because a
prompt is a request and a cleaner is a guarantee.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from rt_agent.contracts.events import DecisionContext
from rt_agent.contracts.memory import MemoryRecord
from rt_agent.contracts.protocols import ChatLLM
from rt_agent.llm.client import strip_reasoning
from rt_agent.llm.errors import LLMEmptyResponseError
from rt_agent.llm.prompts import MAX_REPLY_SENTENCES, reply_messages

__all__ = ["MAX_REPLY_CHARS", "ReplyGenerator", "clean_reply", "split_sentences"]

#: A spoken reply longer than this is cut; two short sentences fit comfortably.
MAX_REPLY_CHARS = 300

_TAG = re.compile(r"<[^<>]{0,64}>")
_LINK = re.compile(r"\[([^\]]{1,200})\]\([^)]{0,300}\)")
_EMPHASIS = re.compile(r"(\*{1,3}|_{1,3}|`{1,3})")
_LINE_PREFIX = re.compile(r"^\s{0,8}(?:[-*+•>]|#{1,6}|\d{1,2}[.)])\s+", re.MULTILINE)
# Anonymous labels (S1, ROBOT) and the obvious persona prefixes only: a lowercase word
# such as "Yes:" is part of the sentence, not a label.
_SPEAKER_PREFIX = re.compile(r"^\s*(?:[A-Z][A-Z0-9_-]{0,31}|Alice|Assistant|Robot)\s*:\s+")
_CJK_STOPS = "\u3002\uff01\uff1f"  # fullwidth . ! ? — a Cantonese reply ends on these
_SENTENCE = re.compile(f"[^.!?{_CJK_STOPS}]*(?:[.!?]+(?=\\s|$)|[{_CJK_STOPS}]+|$)")
_QUOTES = "\"'\u201c\u201d\u2018\u2019\u00ab\u00bb\u300c\u300d\u300e\u300f"


def split_sentences(text: str) -> list[str]:
    """Split on sentence punctuation, keeping the punctuation. Never returns empties."""
    sentences = [match.group(0).strip() for match in _SENTENCE.finditer(text)]
    return [sentence for sentence in sentences if sentence]


def clean_reply(
    text: str,
    *,
    max_chars: int = MAX_REPLY_CHARS,
    max_sentences: int = MAX_REPLY_SENTENCES,
    strip_speaker_prefix: bool = True,
) -> str:
    """Turn whatever the model said into speakable plain text.

    Strips reasoning blocks and chat-template tags, removes markdown, drops a leading
    speaker label and surrounding quotes, collapses whitespace, keeps at most
    ``max_sentences`` sentences and caps the result at ``max_chars`` on a word boundary.
    Raises :class:`~rt_agent.llm.errors.LLMEmptyResponseError` if nothing survives.

    The memory writer reuses this with ``strip_speaker_prefix=False``, because there the
    label *is* the subject of the sentence.
    """
    stripped = strip_reasoning(text)
    stripped = _TAG.sub(" ", stripped)
    stripped = stripped.replace("\u2039", "").replace("\u203a", "")
    stripped = _LINK.sub(r"\1", stripped)
    stripped = _LINE_PREFIX.sub("", stripped)
    stripped = _EMPHASIS.sub("", stripped)
    collapsed = " ".join(stripped.split())
    if strip_speaker_prefix:
        collapsed = _SPEAKER_PREFIX.sub("", collapsed)
    collapsed = collapsed.strip(_QUOTES + " ")

    sentences = split_sentences(collapsed)
    if sentences:
        collapsed = " ".join(sentences[:max_sentences])

    if len(collapsed) > max_chars:
        cut = collapsed[:max_chars]
        space = cut.rfind(" ")
        if space > max_chars // 2:
            cut = cut[:space]
        collapsed = cut.rstrip().rstrip(",;:-\u2013\u2014")
        if collapsed and collapsed[-1] not in f".!?{_CJK_STOPS}":
            collapsed += "."

    collapsed = collapsed.strip()
    if not collapsed:
        raise LLMEmptyResponseError("the reply was empty once markup and reasoning were removed")
    return collapsed


class ReplyGenerator:
    """Asks a :class:`~rt_agent.contracts.protocols.ChatLLM` for one spoken reply."""

    def __init__(
        self,
        llm: ChatLLM,
        *,
        max_chars: int = MAX_REPLY_CHARS,
        max_sentences: int = MAX_REPLY_SENTENCES,
    ) -> None:
        self.llm = llm
        self.max_chars = max_chars
        self.max_sentences = max_sentences

    async def reply(
        self,
        ctx: DecisionContext,
        memories: Sequence[MemoryRecord] | None = None,
    ) -> str:
        """Generate the reply for one decision context.

        ``memories`` defaults to ``ctx.retrieved_memories``. Errors from the model are
        typed (:class:`~rt_agent.llm.errors.LLMError`) and are *not* swallowed here: on
        the speaking path the harness decides whether a failure means silence.
        """
        messages = reply_messages(ctx, memories)
        raw = await self.llm.complete(messages)
        return clean_reply(raw, max_chars=self.max_chars, max_sentences=self.max_sentences)
