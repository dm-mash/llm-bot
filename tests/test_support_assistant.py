"""Tests for the support assistant: question assembly, isolation, and evidence.

Everything is offline: the model is an ``httpx.MockTransport`` and the FAQ
corpus a small index built on the ``FakeEmbedder`` the other RAG tests use. No
MCP server is started here — ``tests/test_mcp_support_server.py`` covers the
desk itself, and the assistant only decides *whether* to call it.
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from llm_bot.json_session_store import JsonSessionStore
from llm_bot.stores import AgentConfig, ModelConfig
from llm_bot.support import SupportAssistant, build_question
from tests.test_rag import make_index

FAQ_CHUNKS = [
    (
        "kb/faq.md",
        "Сброс пароля: ссылка с письма действует 30 минут и одноразовая.",
        "Аккаунт и вход",
    ),
    (
        "kb/faq.md",
        "Двухфакторная аутентификация: при утере телефона доступ восстанавливает "
        "владелец организации, поддержка самостоятельно не может.",
        "Двухфакторная аутентификация",
    ),
]


class _StubModelStore:
    def __init__(self, config: ModelConfig) -> None:
        self._config = config

    def get(self, name: str) -> ModelConfig:
        return self._config

    def list(self) -> list[str]:
        return [self._config.name]


class _StubAgentStore:
    def __init__(self, config: AgentConfig) -> None:
        self._config = config

    def get(self, name: str) -> AgentConfig:
        return self._config

    def list(self) -> list[str]:
        return [self._config.name]


def _agent() -> AgentConfig:
    return AgentConfig(
        name="support",
        model="openai",
        system_prompt="Ты ассистент поддержки.",
        temperature=0.1,
        max_tokens=64,
    )


def _model() -> ModelConfig:
    return ModelConfig(
        name="openai",
        base_url="https://example.test/v1",
        api_key="k",
        model="gpt",
    )


def _transport(reply: str = "Письма не приходят."):
    """A transport recording payloads and answering with *reply*."""
    payloads: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payloads.append(json.loads(request.read().decode()))
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"role": "assistant", "content": reply}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            },
        )

    return httpx.MockTransport(handler), payloads


def _assistant(tmp_path: Path, transport, *, rag_index=None, retriever=None) -> SupportAssistant:
    return SupportAssistant(
        agent="support",
        rag_index=rag_index,
        retriever=retriever,
        mcp_servers=[],
        model_store=_StubModelStore(_model()),
        agent_store=_StubAgentStore(_agent()),
        session_store=JsonSessionStore(str(tmp_path / "sessions")),
        transport=transport,
    )


# --------------------------------------------------------------------------- #
# Question assembly
# --------------------------------------------------------------------------- #


def test_ids_are_prepended_so_the_model_fetches_the_case() -> None:
    text = build_question("Почему не работает авторизация?", ticket_id="T-5001", user_id="U-1001")
    assert "T-5001" in text and "U-1001" in text
    assert text.strip().endswith("Почему не работает авторизация?")


def test_a_general_question_is_left_alone() -> None:
    # No ids means a product question, and a "контекст обращения" header on it
    # would only invite the model to invent a customer.
    assert build_question("Как сменить тариф?") == "Как сменить тариф?"


def test_only_the_ids_travel_not_the_ticket_body() -> None:
    # The ticket text must come from MCP at answer time; a copy made here could
    # be stale by the time the model reads it.
    text = build_question("вопрос", ticket_id="T-5001")
    assert "Переписка" not in text


# --------------------------------------------------------------------------- #
# Answering
# --------------------------------------------------------------------------- #


def test_answer_returns_the_reply_and_the_usage(tmp_path: Path) -> None:
    transport, _ = _transport("Письма со сбросом не доходят.")
    answer = _assistant(tmp_path, transport).answer("Почему не приходят письма?")

    assert "не доходят" in answer.answer
    assert answer.usage.total_tokens == 15
    assert answer.session_id == "support-default"


def test_the_context_ids_reach_the_model(tmp_path: Path) -> None:
    transport, payloads = _transport()
    _assistant(tmp_path, transport).answer(
        "Почему не работает авторизация?", ticket_id="T-5001"
    )
    sent = json.dumps(payloads[0], ensure_ascii=False)
    assert "T-5001" in sent


def test_an_empty_question_is_refused_before_any_call(tmp_path: Path) -> None:
    transport, payloads = _transport()
    with pytest.raises(ValueError):
        _assistant(tmp_path, transport).answer("   ")
    assert payloads == []


def test_sources_are_reported_from_the_faq_index(tmp_path: Path, monkeypatch) -> None:
    # The operator needs to see WHICH documentation backed the answer; a reply
    # with no sources is an answer from the model's memory.
    from tests.test_index_documents import FakeEmbedder

    monkeypatch.setattr(
        "llm_bot.rag.Embedder", lambda name, local_only=False: FakeEmbedder(name)
    )
    index = make_index(tmp_path / "emb", FAQ_CHUNKS)
    transport, _ = _transport()
    answer = _assistant(tmp_path, transport, rag_index=index).answer("сброс пароля не работает")

    assert answer.sources
    assert all(":1-" in s or ":11-" in s for s in answer.sources)


def test_grounding_is_none_when_rag_is_off(tmp_path: Path) -> None:
    # With no FAQ corpus there is nothing to ground against, and reporting a
    # verdict anyway would be a made-up number on the operator's screen.
    transport, _ = _transport("Вероятно, у вас всё в порядке.")
    answer = _assistant(tmp_path, transport).answer("вопрос")
    assert answer.grounding == "none"


def test_grounding_is_reported_when_a_corpus_is_attached(tmp_path: Path, monkeypatch) -> None:
    from tests.test_index_documents import FakeEmbedder

    monkeypatch.setattr(
        "llm_bot.rag.Embedder", lambda name, local_only=False: FakeEmbedder(name)
    )
    index = make_index(tmp_path / "emb", FAQ_CHUNKS)
    transport, _ = _transport("Вероятно, у вас всё в порядке.")
    answer = _assistant(tmp_path, transport, rag_index=index).answer("сброс пароля")
    assert answer.grounding in {"grounded", "refused", "ungrounded"}


# --------------------------------------------------------------------------- #
# Conversation isolation
# --------------------------------------------------------------------------- #


def test_two_conversations_do_not_share_history(tmp_path: Path) -> None:
    # Two customers must never see each other's dialog: the assistant answers
    # questions about one customer's ticket while serving another.
    transport, payloads = _transport()
    assistant = _assistant(tmp_path, transport)
    assistant.answer("первый вопрос", session_id="c1")
    assistant.answer("второй вопрос", session_id="c2")

    assert len(payloads) == 2
    assert "первый вопрос" not in json.dumps(payloads[1], ensure_ascii=False)


def test_the_same_conversation_keeps_its_context(tmp_path: Path) -> None:
    transport, payloads = _transport()
    assistant = _assistant(tmp_path, transport)
    assistant.answer("первый вопрос", session_id="c1")
    assistant.answer("уточнение", session_id="c1")

    assert assistant.session("c1") is assistant.session("c1")
    assert len(payloads) == 2


def test_forget_starts_a_fresh_conversation(tmp_path: Path) -> None:
    transport, payloads = _transport()
    assistant = _assistant(tmp_path, transport)
    first = assistant.session("c1")
    assistant.forget("c1")
    assert assistant.session("c1") is not first


def test_to_dict_is_the_json_body_the_endpoint_returns(tmp_path: Path) -> None:
    transport, _ = _transport("ок")
    body = _assistant(tmp_path, transport).answer("вопрос").to_dict()

    assert set(body) == {
        "answer",
        "sources",
        "tools",
        "grounding",
        "usage",
        "session_id",
    }
    assert body["usage"]["total_tokens"] == 15

# --------------------------------------------------------------------------- #
# Sharing the process-wide retriever
# --------------------------------------------------------------------------- #


def test_an_injected_retriever_is_used_as_is(tmp_path: Path, monkeypatch) -> None:
    # The HTTP service already holds a Retriever over the same corpus; a second
    # one would load the embedding model again and keep both in memory.
    from llm_bot.rag import Retriever
    from tests.test_index_documents import FakeEmbedder

    monkeypatch.setattr(
        "llm_bot.rag.Embedder", lambda name, local_only=False: FakeEmbedder(name)
    )
    index = make_index(tmp_path / "emb", FAQ_CHUNKS)
    shared = Retriever(index, embedder=FakeEmbedder("fake-mini"))
    assistant = _assistant(tmp_path, _transport()[0], retriever=shared)
    assert assistant.retriever is shared


def test_without_a_retriever_there_is_no_rag(tmp_path: Path) -> None:
    assert _assistant(tmp_path, _transport()[0]).retriever is None


# --------------------------------------------------------------------------- #
# MCP off vs default, and no silent corpus
# --------------------------------------------------------------------------- #


def test_an_empty_server_list_means_mcp_off_not_the_default() -> None:
    # None means "use the default servers"; an empty list means "no tools".
    # Conflating them turns a deployment without a support desk into a startup
    # error against a config that has no such server.
    assistant = SupportAssistant(
        agent="support", rag_index=None, mcp_servers=[],
        model_store=None, agent_store=None, session_store=None,
    )
    assert assistant.mcp_servers == []


def test_no_rag_index_means_no_rag_and_no_default_corpus() -> None:
    # An assistant answering from a corpus nobody selected is worse than one that
    # answers nothing: the operator would not know where the answer came from.
    assistant = SupportAssistant(
        agent="support", rag_index=None,
        model_store=None, agent_store=None, session_store=None,
    )
    assert assistant.rag_index is None
    assert assistant.retriever is None


def test_nothing_is_written_to_disk_by_default(tmp_path: Path) -> None:
    # The endpoint mints a conversation id per request; persisting each of them
    # would grow data/sessions without bound.
    assistant = SupportAssistant(
        agent="support", rag_index=None, mcp_servers=[],
        model_store=None, agent_store=None, session_store=None,
    )
    assistant.session_store.save("c1", [{"role": "user", "content": "x"}])
    assert assistant.session_store.load("c1") == []
    assert not list(tmp_path.iterdir())


def test_the_session_cache_is_bounded(tmp_path: Path) -> None:
    from llm_bot.support import MAX_SESSIONS

    assistant = _assistant(tmp_path, _transport()[0])
    for index in range(MAX_SESSIONS + 10):
        assistant.session(f"c{index}")
    assert len(assistant._sessions) == MAX_SESSIONS


def test_evidence_is_scoped_to_the_turn_that_produced_it(tmp_path: Path) -> None:
    """The sources and tools reported must belong to THIS turn.

    ``mcp_events`` accumulates over the conversation and a skipped retrieval
    leaves ``last_rag_event`` pointing at the previous question, so reading
    either without a before/after snapshot attributes an earlier turn's evidence
    to an answer it did not produce.
    """
    class _Hit:
        location = "faq.md:1-5"
        source = "faq.md"
        start_line, end_line = 1, 5

    class _RagEvent:
        def __init__(self, sources):
            self.sources = tuple(sources)

    class _Event:
        def __init__(self, tool):
            self.tool = tool
            self.ok = True

    class _Result:
        def __init__(self, reply):
            self.reply = reply
            self.usage = None

    class _FakeSession:
        def __init__(self):
            self.mcp_events = []
            self.rag_events = []
            self.last_grounding = None
            self.turn = 0

        def chat_with_details(self, message, **kwargs):
            self.turn += 1
            self.mcp_events.append(_Event(f"get_ticket_T{self.turn}"))
            if self.turn == 2:
                # A turn that retrieved nothing: the previous turn's evidence
                # must not be re-reported for it.
                return _Result("без источников")
            self.rag_events.append(_RagEvent([_Hit()]))
            return _Result("с источником")

    assistant = _assistant(tmp_path, _transport()[0])
    assistant._sessions["c1"] = _FakeSession()

    first = assistant.answer("первый", session_id="c1")
    second = assistant.answer("второй", session_id="c1")

    assert first.sources == ("faq.md:1-5",)
    assert first.tools == ("get_ticket_T1",)

    assert second.sources == (), "источники первого хода не должны достаться второму"
    assert second.grounding == "none"
    assert second.tools == ("get_ticket_T2",), "инструменты накапливаться не должны"


def test_the_ticket_directive_does_not_call_the_reader_a_client():
    # Calling the person a «клиент» is what pushed every answer into third
    # person: the reply to «что делать-то?» came back as «Для Maria Sokolova
    # есть два пути». The case belongs to whoever is asking.
    from llm_bot.support import ticket_directive

    text = ticket_directive("T-5004", "U-1003")
    assert "ваше обращение" in text
    assert "клиент" not in text.lower()
