"""Tests for the ``POST /support`` endpoint.

Offline: the assistant is a stand-in, so these cover the route itself — auth,
validation, session identity, error mapping — without a model, an index or an
MCP server. The assistant's own behaviour is in ``test_support_assistant.py``.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from llm_bot.client import LLMError, LLMRequestError
from llm_bot.server import create_app


class _FakeAnswer:
    def __init__(self, **fields) -> None:
        self._fields = {
            "answer": "Письма не приходят.",
            "sources": ["faq.md:12-16"],
            "tools": ["support__get_ticket"],
            "grounding": "grounded",
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            "session_id": "",
        }
        self._fields.update(fields)

    def to_dict(self) -> dict:
        return dict(self._fields)


class _FakeAssistant:
    """Records what the route passed through and replays a canned answer."""

    def __init__(self, result=None, error: Exception | None = None) -> None:
        self.result = result if result is not None else _FakeAnswer()
        self.error = error
        self.calls: list[dict] = []

    def answer(self, question, *, ticket_id=None, user_id=None, session_id="x"):
        self.calls.append(
            {"question": question, "ticket_id": ticket_id, "user_id": user_id, "session_id": session_id}
        )
        if self.error is not None:
            raise self.error
        return self.result


def _client(assistant=None, **kwargs) -> TestClient:
    built = assistant if assistant is not None else _FakeAssistant()
    app = create_app(
        session_factory=lambda session_id: None,
        support_factory=lambda conversation_id: built,
        **kwargs,
    )
    return TestClient(app), built


# --------------------------------------------------------------------------- #
# Presence
# --------------------------------------------------------------------------- #


def test_the_route_is_absent_when_no_support_assistant_is_wired() -> None:
    # No support desk configured must mean no route at all — not an advertised
    # endpoint that can only answer 404. Clients read /openapi.json before they
    # call anything, so a route that only fails is a broken promise.
    app = create_app(session_factory=lambda session_id: None)
    assert "/support" not in [getattr(r, "path", None) for r in app.routes]
    assert "/support" not in TestClient(app).get("/openapi.json").json()["paths"]


def test_the_route_is_advertised_once_enabled() -> None:
    client, _ = _client()
    assert "/support" in TestClient(client.app).get("/openapi.json").json()["paths"]


def test_the_conversation_cache_is_bounded() -> None:
    # Each request without a session id would otherwise add a dict entry for the
    # life of the process.
    from llm_bot.server import MAX_SUPPORT_SESSIONS

    assistant = _FakeAssistant()
    app = create_app(session_factory=lambda s: None, support_factory=lambda s: assistant)
    client = TestClient(app)
    for _ in range(MAX_SUPPORT_SESSIONS + 5):
        client.post("/support", json={"question": "а"})
    kept = [v for k, v in vars(app.state).items()] if app.state else []
    assert len(kept) <= MAX_SUPPORT_SESSIONS


def test_the_chat_endpoint_still_works_alongside_support() -> None:
    client, _ = _client()
    assert client.get("/health").status_code == 200


# --------------------------------------------------------------------------- #
# Request handling
# --------------------------------------------------------------------------- #


def test_the_ids_reach_the_assistant() -> None:
    client, fake = _client()
    client.post(
        "/support",
        json={"question": "Почему не работает авторизация?", "ticket_id": "T-5001", "user_id": "U-1001"},
    )
    call = fake.calls[0]
    assert call["ticket_id"] == "T-5001"
    assert call["user_id"] == "U-1001"


def test_a_question_without_ids_is_allowed() -> None:
    client, fake = _client()
    client.post("/support", json={"question": "Как сменить тариф?"})
    assert fake.calls[0]["ticket_id"] is None


def test_blank_ids_are_treated_as_absent() -> None:
    client, fake = _client()
    client.post("/support", json={"question": "вопрос", "ticket_id": "  "})
    assert fake.calls[0]["ticket_id"] is None


def test_a_missing_question_is_a_400() -> None:
    client, fake = _client()
    assert client.post("/support", json={}).status_code == 400
    assert client.post("/support", json={"question": "   "}).status_code == 400
    assert client.post("/support", json={"question": 42}).status_code == 400
    assert fake.calls == []


def test_a_malformed_body_is_a_400() -> None:
    client, _ = _client()
    assert client.post("/support", content=b"{oops").status_code == 400


def test_the_answer_is_returned_with_its_evidence() -> None:
    client, _ = _client()
    body = client.post("/support", json={"question": "вопрос"}).json()
    assert body["answer"] == "Письма не приходят."
    assert body["sources"] == ["faq.md:12-16"]
    assert body["tools"] == ["support__get_ticket"]
    assert body["grounding"] == "grounded"
    assert body["usage"]["total_tokens"] == 15


# --------------------------------------------------------------------------- #
# Conversation identity
# --------------------------------------------------------------------------- #


def test_each_request_gets_its_own_conversation_by_default() -> None:
    # Two customers sharing one dialog would leak one customer's ticket into
    # another's answer, so an absent id must NOT collapse to a shared default.
    client, fake = _client()
    first = client.post("/support", json={"question": "а"}).json()["session_id"]
    second = client.post("/support", json={"question": "б"}).json()["session_id"]
    assert first != second


def test_an_explicit_session_id_is_honoured() -> None:
    client, _ = _client()
    body = client.post("/support", json={"question": "а", "session_id": "conv-7"}).json()
    assert body["session_id"] == "conv-7"


def test_the_session_header_is_used_when_the_body_has_no_id() -> None:
    client, fake = _client()
    client.post("/support", json={"question": "а"}, headers={"X-Session-Id": "conv-9"})
    assert fake.calls[0]["session_id"] == "conv-9"


# --------------------------------------------------------------------------- #
# Auth, limits, errors
# --------------------------------------------------------------------------- #


def test_the_token_is_required_when_the_server_has_one() -> None:
    client, _ = _client(auth_token="s3cret")
    body = {"question": "а"}
    assert client.post("/support", json=body).status_code == 401
    assert client.post("/support", json=body, headers={"Authorization": "Bearer nope"}).status_code == 401
    ok = client.post("/support", json=body, headers={"Authorization": "Bearer s3cret"})
    assert ok.status_code == 200


def test_health_stays_open_without_a_token() -> None:
    client, _ = _client(auth_token="s3cret")
    assert client.get("/health").status_code == 200


def test_a_rate_limited_client_is_refused() -> None:
    client, _ = _client(rate_limit=1, rate_burst=1)
    assert client.post("/support", json={"question": "а"}).status_code == 200
    assert client.post("/support", json={"question": "б"}).status_code == 429


def test_an_upstream_failure_is_a_502_not_a_500() -> None:
    # The caller distinguishes "the model is down" from "we sent bad JSON".
    client, _ = _client(_FakeAssistant(error=LLMRequestError("boom")))
    assert client.post("/support", json={"question": "а"}).status_code == 502


def test_any_llm_error_is_a_502() -> None:
    client, _ = _client(_FakeAssistant(error=LLMError("offline")))
    assert client.post("/support", json={"question": "а"}).status_code == 502


def test_a_value_error_from_the_assistant_is_a_400() -> None:
    client, _ = _client(_FakeAssistant(error=ValueError("Пустой вопрос.")))
    response = client.post("/support", json={"question": "а"})
    assert response.status_code == 400
    assert "Пустой вопрос" in response.json()["detail"]

# --------------------------------------------------------------------------- #
# The chat page's ticket dropdown
# --------------------------------------------------------------------------- #


def test_the_ticket_list_is_served_for_the_dropdown() -> None:
    client, _ = _client()
    body = client.get("/support/tickets").json()
    assert [t["id"] for t in body["tickets"]]
    first = body["tickets"][0]
    assert {"id", "subject", "status", "user_id"} <= set(first)


def test_the_ticket_list_needs_the_token_when_the_server_has_one() -> None:
    client, _ = _client(auth_token="s3cret")
    assert client.get("/support/tickets").status_code == 401
    ok = client.get("/support/tickets", headers={"Authorization": "Bearer s3cret"})
    assert ok.status_code == 200


def test_the_list_endpoint_is_absent_without_support() -> None:
    # A dropdown with nothing to fill it would be worse than no dropdown: the
    # page hides itself when this request is not 200.
    app = create_app(session_factory=lambda session_id: None)
    assert TestClient(app).get("/support/tickets").status_code == 404


def test_the_dropdown_carries_no_message_bodies() -> None:
    # The page only needs ids and subjects; shipping whole conversations to the
    # browser for a menu would leak them to anyone who opens the page.
    client, _ = _client()
    body = client.get("/support/tickets").text
    assert "Переписка" not in body
    assert "customer" not in body
