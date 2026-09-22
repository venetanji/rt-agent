"""Opt-in live checks against a real System One endpoint.

    RT_AGENT_LIVE=1 TYPESAFE_API_KEY=... uv run pytest -m live

These are skipped by default: the default suite is network-free.
"""

from __future__ import annotations

import os

import pytest

from rt_agent.contracts import DecisionContext
from rt_agent.policy import Policy
from rt_agent.systemone import QUESTION_BUNDLE_V1, SystemOneClient, render_state
from tests.conftest import SYSTEMONE_FIXTURES

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        os.environ.get("RT_AGENT_LIVE") != "1",
        reason="live System One tests are opt-in; set RT_AGENT_LIVE=1",
    ),
]


@pytest.fixture(scope="module")
def client() -> SystemOneClient:
    live = SystemOneClient.jev(timeout_s=20.0)
    yield live
    live.close()


def test_the_configured_model_is_advertised(client: SystemOneClient) -> None:
    assert client.model in client.check_model()


def test_the_frozen_bundle_answers_a_direct_question(client: SystemOneClient) -> None:
    import json

    document = json.loads(
        (SYSTEMONE_FIXTURES / "direct_question_time.json").read_text(encoding="utf-8")
    )
    ctx = DecisionContext.model_validate(document["context"])
    answers = client.ask(render_state(ctx), QUESTION_BUNDLE_V1.to_wire())

    assert answers.answered == set(QUESTION_BUNDLE_V1.ids)
    assert answers.model.startswith("jev-")
    assert answers.usage is not None

    # The recorded values are stable to about +/- 0.02; keep the assertions coarse.
    assert answers.noul("intelligible_complete") > 0.8
    assert answers.noul("addressed_to_robot") > 0.8
    assert answers.noul("invites_response_now") > 0.8
    assert answers.choice("addressee").chosen == "robot"

    decision = Policy().decide(ctx, answers, answers.latency_ms)
    assert decision.speak is True


@pytest.mark.asyncio
async def test_the_async_path_works_live(client: SystemOneClient) -> None:
    import json

    document = json.loads(
        (SYSTEMONE_FIXTURES / "unintelligible_fragment.json").read_text(encoding="utf-8")
    )
    ctx = DecisionContext.model_validate(document["context"])
    answers = await client.aask_context(ctx)
    assert answers.noul("intelligible_complete") < 0.4
    await client.aclose()
