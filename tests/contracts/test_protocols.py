"""The protocols really are the seams: the shipped implementations satisfy them."""

from __future__ import annotations

from rt_agent.contracts import ChatMessage, SystemOneBackend
from rt_agent.systemone import MockSystemOne, SystemOneClient


def test_the_mock_is_a_system_one_backend() -> None:
    assert isinstance(MockSystemOne(), SystemOneBackend)


def test_the_live_client_is_a_system_one_backend() -> None:
    assert isinstance(SystemOneClient.jev(), SystemOneBackend)


def test_chat_messages_reject_empty_content() -> None:
    message = ChatMessage(role="user", content="hello")
    assert message.role == "user"
