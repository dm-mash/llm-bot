"""Tests for llm_bot.invariants and its integration into Session/factory."""

from __future__ import annotations

import json

import httpx
import pytest

from llm_bot.agent import Session
from llm_bot.factory import make_session
from llm_bot.invariants import (
    Invariant,
    InvariantAuditEvent,
    InvariantRegistry,
    InvariantViolationError,
    audit_reply,
)
from llm_bot.json_session_store import JsonSessionStore
from llm_bot.stores import AgentConfig, ModelConfig
from llm_bot.yaml_stores import YamlInvariantStore


# --------------------------------------------------------------------------- #
# Helpers (mirror tests/test_agent.py)
# --------------------------------------------------------------------------- #


def _ok_response() -> httpx.Response:
    return httpx.Response(
        200,
        json={"choices": [{"message": {"role": "assistant", "content": "reply-text"}}]},
    )


def _agent_config() -> AgentConfig:
    return AgentConfig(
        name="assistant",
        model="openai",
        system_prompt="Ты помощник.",
    )


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


def _make_session(
    session_id: str,
    transport: httpx.BaseTransport,
    directory: str,
    *,
    invariants=None,
    audit_invariants_warn: bool = False,
) -> Session:
    model_cfg = ModelConfig(
        name="openai",
        base_url="https://example.test/v1",
        api_key="k",
        model="gpt",
    )
    session_store = JsonSessionStore(directory)
    return make_session(
        session_id,
        _agent_config().name,
        model_store=_StubModelStore(model_cfg),
        agent_store=_StubAgentStore(_agent_config()),
        session_store=session_store,
        transport=transport,
        invariants=invariants,
        audit_invariants_warn=audit_invariants_warn,
    )


def _inv(**overrides) -> Invariant:
    data = {
        "id": "STACK-1",
        "kind": "stack",
        "statement": "Никаких тяжёлых фреймворков.",
        "rationale": "Минимум зависимостей.",
        "forbidden_patterns": [r"\bflask\b"],
    }
    # ``source`` is not part of the serialized form — it is a from_dict kwarg.
    source = overrides.pop("source", "global")
    data.update(overrides)
    return Invariant.from_dict(data, source=source)


# --------------------------------------------------------------------------- #
# Invariant model
# --------------------------------------------------------------------------- #


class TestInvariant:
    def test_from_dict_normalizes(self):
        item = Invariant.from_dict({
            "id": " S1 ",
            "kind": "STACK",
            "statement": "  правило  ",
            "rationale": " потому что ",
            "forbidden_patterns": "flask",
        })
        assert item.id == "S1"
        assert item.kind == "stack"
        assert item.statement == "правило"
        assert item.rationale == "потому что"
        assert item.forbidden_patterns == ["flask"]
        assert item.source == "global"

    def test_from_dict_uses_default_id(self):
        item = Invariant.from_dict(
            {"kind": "stack", "statement": "правило"}, default_id="KEY-1"
        )
        assert item.id == "KEY-1"

    def test_validation_errors(self):
        with pytest.raises(ValueError):  # empty id
            Invariant.from_dict({"kind": "stack", "statement": "x"})
        with pytest.raises(ValueError):  # empty kind
            Invariant.from_dict({"id": "A", "statement": "x"})
        with pytest.raises(ValueError):  # empty statement
            Invariant.from_dict({"id": "A", "kind": "stack", "statement": "  "})
        with pytest.raises(ValueError):  # broken regex fails fast
            Invariant.from_dict({
                "id": "A", "kind": "stack", "statement": "x",
                "forbidden_patterns": ["["],
            })

    def test_to_dict_roundtrip_without_source(self):
        item = _inv()
        data = item.to_dict()
        assert "source" not in data
        restored = Invariant.from_dict({**data, "source": "session"})
        assert restored.id == item.id
        assert restored.kind == item.kind
        assert restored.statement == item.statement

    def test_refusal_text_names_id_kind_rationale(self):
        text = _inv().refusal_text("flask")
        assert "STACK-1" in text
        assert "stack" in text
        assert "Минимум зависимостей." in text
        assert "flask" in text

    def test_refusal_without_rationale_and_match(self):
        text = Invariant.from_dict({
            "id": "B", "kind": "rule", "statement": "правило"
        }).refusal_text()
        assert "B" in text and "правило" in text
        assert "Причина" not in text
        assert "Сработал запрет" not in text

    def test_render_block_line(self):
        line = _inv().render_block_line("ограничение по стеку")
        assert line.startswith("- [STACK-1] (ограничение по стеку)")
        assert "(причина: Минимум зависимостей.)" in line
        # without label falls back to the raw kind
        assert "(stack)" in _inv().render_block_line()


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #


class TestInvariantRegistry:
    def test_add_drop_and_duplicate_protection(self):
        registry = InvariantRegistry()
        registry.add(_inv())
        assert registry.has("STACK-1") and len(registry) == 1
        with pytest.raises(ValueError):
            registry.add(_inv())
        dropped = registry.drop("STACK-1")
        assert dropped.id == "STACK-1" and len(registry) == 0
        with pytest.raises(KeyError):
            registry.drop("STACK-1")

    def test_items_sorted_by_id(self):
        registry = InvariantRegistry(
            [_inv(id="B-2"), _inv(id="A-1")]
        )
        assert [i.id for i in registry.items()] == ["A-1", "B-2"]

    def test_is_protected(self):
        registry = InvariantRegistry(
            [_inv(source="global"), _inv(id="S-1", source="session")]
        )
        assert registry.is_protected("STACK-1")   # global
        assert not registry.is_protected("S-1")   # session-scoped
        assert registry.is_protected("missing")   # unknown -> treat protected

    def test_check_request_matches_first_violation(self):
        registry = InvariantRegistry(
            [_inv(), _inv(id="RULE-1", kind="rule",
                          forbidden_patterns=[r"джанго|django"])]
        )
        assert registry.check_request("сделай на flask") is registry.get("STACK-1")
        assert registry.check_request("перепиши на Django") is registry.get("RULE-1")
        assert registry.check_request("обычный вопрос") is None

    def test_render_prompt_block_contains_protocol(self):
        block = InvariantRegistry([_inv()]).render_prompt_block()
        assert "[STACK-1]" in block
        assert "ВЫСШИЙ ПРИОРИТЕТ" in block
        assert "ОТКАЖИ" in block
        assert "альтернативу" in block
        assert "ОБЯЗАТЕЛЬНО" in block

    def test_render_prompt_block_empty(self):
        assert InvariantRegistry().render_prompt_block() == ""

    def test_render_footer_returns_reminder(self):
        footer = InvariantRegistry([_inv()]).render_footer()
        assert "НАПОМИНАНИЕ" in footer
        assert "ОТКАЖИ" in footer

    def test_render_footer_empty_when_no_invariants(self):
        assert InvariantRegistry().render_footer() == ""

    def test_kind_labels(self):
        registry = InvariantRegistry([_inv()], kind_labels={"stack": "стек"})
        assert registry.kind_label("stack") == "стек"
        assert registry.kind_label("unknown") == "unknown"
        assert "(стек)" in registry.render_prompt_block()

    def test_on_change_hook_fires(self):
        calls = []
        registry = InvariantRegistry(on_change=lambda: calls.append(1))
        registry.add(_inv())
        registry.drop("STACK-1")
        assert len(calls) == 2


# --------------------------------------------------------------------------- #
# YAML store
# --------------------------------------------------------------------------- #


class TestYamlInvariantStore:
    def test_reads_invariants_and_labels(self, tmp_path):
        path = tmp_path / "invariants.yaml"
        path.write_text(
            "kind_labels:\n"
            "  stack: ограничение по стеку\n"
            "invariants:\n"
            "  STACK-1:\n"
            "    kind: stack\n"
            "    statement: правило один\n"
            "    rationale: потому что\n"
            "    forbidden_patterns: ['flask']\n",
            encoding="utf-8",
        )
        store = YamlInvariantStore(str(path))
        assert store.list() == ["STACK-1"]
        entry = store.get("STACK-1")
        assert entry["id"] == "STACK-1"
        assert entry["statement"] == "правило один"
        assert store.kind_labels() == {"stack": "ограничение по стеку"}
        item = Invariant.from_dict(entry)
        assert item.kind == "stack"

    def test_missing_labels_section_is_empty(self, tmp_path):
        path = tmp_path / "invariants.yaml"
        path.write_text(
            "invariants:\n  A:\n    kind: x\n    statement: y\n",
            encoding="utf-8",
        )
        store = YamlInvariantStore(str(path))
        assert store.kind_labels() == {}

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            YamlInvariantStore(str(tmp_path / "nope.yaml")).list()

    def test_missing_top_level_key_raises(self, tmp_path):
        path = tmp_path / "invariants.yaml"
        path.write_text("other: {}\n", encoding="utf-8")
        with pytest.raises(ValueError):
            YamlInvariantStore(str(path)).list()


# --------------------------------------------------------------------------- #
# Session integration
# --------------------------------------------------------------------------- #


class TestSessionInvariants:
    def _capturing_transport(self, captured: dict):
        def handler(request: httpx.Request) -> httpx.Response:
            captured["requests"] = captured.get("requests", 0) + 1
            # Capture only the FIRST request payload (the chat call).
            # Subsequent calls (audit, memory, task) overwrite — tests that
            # need the chat payload should read it before those calls fire,
            # or we store only the first one.
            if "payload" not in captured:
                captured["payload"] = request.read().decode()
            else:
                request.read()  # consume the body
            return _ok_response()

        return httpx.MockTransport(handler)

    def test_block_injected_before_system_prompt(self, tmp_path):
        captured: dict = {}
        session = _make_session(
            "inv1",
            self._capturing_transport(captured),
            str(tmp_path / "s"),
            invariants=InvariantRegistry([_inv()]),
        )
        session.chat("обычный вопрос")
        payload = json.loads(captured["payload"])
        system_contents = [
            m["content"] for m in payload["messages"]
            if m["role"] == "system"
        ]
        # invariants block is the FIRST system message, before the role prompt
        assert any("[STACK-1]" in c for c in system_contents[:1])
        assert "Ты помощник." in system_contents[1]

    def test_footer_injected_before_user_message(self, tmp_path):
        """The recency-bias footer appears as a system message right before
        the current user message."""
        captured: dict = {}
        session = _make_session(
            "inv-footer",
            self._capturing_transport(captured),
            str(tmp_path / "s"),
            invariants=InvariantRegistry([_inv()]),
        )
        session.chat("привет")
        payload = json.loads(captured["payload"])
        messages = payload["messages"]
        # Find the user message position
        user_idx = next(
            i for i, m in enumerate(messages) if m["role"] == "user"
        )
        # The message right before it! it should be the footer system message
        assert user_idx > 0
        assert messages[user_idx - 1]["role"] == "system"
        assert "НАПОМИНАНИЕ" in messages[user_idx - 1]["content"]

    def test_gate_refuses_without_llm_call(self, tmp_path):
        captured: dict = {}
        session = _make_session(
            "inv2",
            self._capturing_transport(captured),
            str(tmp_path / "s"),
            invariants=InvariantRegistry([_inv()]),
        )
        with pytest.raises(InvariantViolationError) as exc_info:
            session.chat("а давай сделаем на flask")
        # nothing was sent and nothing was recorded
        assert captured.get("requests", 0) == 0
        assert session.history == []
        refusal = str(exc_info.value)
        assert "STACK-1" in refusal
        assert exc_info.value.matched_pattern == r"\bflask\b"

    def test_add_and_drop_session_invariant_persists(self, tmp_path):
        store = JsonSessionStore(str(tmp_path / "s"))
        session = _make_session(
            "inv3",
            self._capturing_transport({}),
            str(tmp_path / "s"),
            invariants=InvariantRegistry(),
        )
        item = session.add_invariant(
            invariant_id="SESSION-1",
            kind="business_rule",
            statement="Не предлагаем платные услуги.",
            rationale="Бесплатный продукт.",
            forbidden_patterns=["платн"],
        )
        assert item.source == "session"
        # persisted as session-scoped only
        saved = store.load_invariants("inv3")
        assert [e["id"] for e in saved] == ["SESSION-1"]

        # a fresh session restores it (global merge happens in factory)
        restored = Session(
            "inv3",
            session.agent,
            store=store,
            invariants=InvariantRegistry(),
        )
        assert restored.invariants.has("SESSION-1")
        assert restored.invariants.get("SESSION-1").source == "session"

        # drop works for session-scoped
        session.drop_invariant("SESSION-1")
        assert store.load_invariants("inv3") == []

    def test_drop_global_invariant_is_rejected(self, tmp_path):
        session = _make_session(
            "inv4",
            self._capturing_transport({}),
            str(tmp_path / "s"),
            invariants=InvariantRegistry([_inv()]),
        )
        with pytest.raises(ValueError):
            session.drop_invariant("STACK-1")
        # the invariant is still there
        assert session.invariants.has("STACK-1")

    def test_merge_global_and_session_without_duplicates(self, tmp_path):
        session = _make_session(
            "inv5",
            self._capturing_transport({}),
            str(tmp_path / "s"),
            invariants=InvariantRegistry([_inv()]),
        )
        with pytest.raises(ValueError):
            # duplicate id (even from a session source) is refused
            session.add_invariant("STACK-1", "stack", "дубль")
        session.add_invariant("SESSION-1", "rule", "правило")
        ids = [i.id for i in session.invariants.items()]
        assert ids == ["SESSION-1", "STACK-1"]

    def test_audit_hard_gate_refuses_violating_reply(self, tmp_path):
        """When the audit detects a violation, the reply is refused (hard gate)."""
        replies = iter([
            json.dumps({"violated_id": "STACK-1", "rationale": "предложил flask"}),
        ])

        def handler(request: httpx.Request) -> httpx.Response:
            content = json.loads(request.read().decode())
            return httpx.Response(
                200,
                json={"choices": [{"message": {
                    "role": "assistant",
                    "content": next(replies),
                }}]},
            ) if content["messages"][0]["content"].startswith("You are an invariant") \
                else _ok_response()

        session = _make_session(
            "inv6",
            httpx.MockTransport(handler),
            str(tmp_path / "s"),
            invariants=InvariantRegistry([_inv()]),
        )
        with pytest.raises(InvariantViolationError) as exc_info:
            session.chat("обычный вопрос")
        assert "STACK-1" in str(exc_info.value)
        # History was rolled back — the turn never happened.
        assert session.history == []

    def test_audit_warn_mode_shows_reply(self, tmp_path):
        """With audit_invariants_warn=True, violation is logged but reply is shown."""
        replies = iter([
            json.dumps({"violated_id": "STACK-1", "rationale": "предложил flask"}),
        ])

        def handler(request: httpx.Request) -> httpx.Response:
            content = json.loads(request.read().decode())
            return httpx.Response(
                200,
                json={"choices": [{"message": {
                    "role": "assistant",
                    "content": next(replies),
                }}]},
            ) if content["messages"][0]["content"].startswith("You are an invariant") \
                else _ok_response()

        session = _make_session(
            "inv6w",
            httpx.MockTransport(handler),
            str(tmp_path / "s"),
            invariants=InvariantRegistry([_inv()]),
            audit_invariants_warn=True,
        )
        reply = session.chat("обычный вопрос")
        assert reply == "reply-text"  # reply is shown
        event = session.last_invariant_event
        assert event is not None and event.violated

    def test_audit_on_by_default_when_invariants_wired(self, tmp_path):
        """When invariants are wired, the audit runs automatically (2 requests)."""
        captured: dict = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["requests"] = captured.get("requests", 0) + 1
            return _ok_response()

        session = _make_session(
            "inv7",
            httpx.MockTransport(handler),
            str(tmp_path / "s"),
            invariants=InvariantRegistry([_inv()]),
        )
        session.chat("обычный вопрос")
        assert captured.get("requests", 0) == 2  # chat + audit

    def test_no_invariants_registry_no_block_no_gate(self, tmp_path):
        captured: dict = {}
        session = _make_session(
            "inv8",
            self._capturing_transport(captured),
            str(tmp_path / "s"),
            invariants=False,
        )
        session.chat("делай что угодно на flask")
        payload = json.loads(captured["payload"])
        assert all("Инварианты" not in m["content"]
                   for m in payload["messages"] if m["role"] == "system")
        assert session.history[-1]["content"] == "reply-text"


# --------------------------------------------------------------------------- #
# audit_reply (unit)
# --------------------------------------------------------------------------- #


class TestAuditReply:
    def test_clean_reply(self):
        def chat(messages):
            return json.dumps({"violated_id": "", "rationale": ""})

        event = audit_reply(InvariantRegistry([_inv()]), "чистый ответ", chat)
        assert not event.violated
        assert event.recognized

    def test_unknown_violated_id_is_dropped(self):
        def chat(messages):
            return json.dumps({"violated_id": "NOPE-9", "rationale": "?"})

        event = audit_reply(InvariantRegistry([_inv()]), "ответ", chat)
        assert event.violated_id == ""

    def test_broken_json_yields_unrecognized(self):
        def chat(messages):
            return "не JSON вовсе"

        event = audit_reply(InvariantRegistry([_inv()]), "ответ", chat)
        assert not event.recognized
        assert not event.violated

    def test_empty_registry_short_circuits(self):
        calls = []

        def chat(messages):
            calls.append(messages)
            return "{}"

        event = audit_reply(InvariantRegistry(), "ответ", chat)
        assert not event.violated and not calls

    def test_user_message_included_in_audit_prompt(self):
        """The audit prompt must contain the user message for context."""
        captured = []

        def chat(messages):
            captured.append(messages)
            return json.dumps({"violated_id": "", "rationale": ""})

        audit_reply(
            InvariantRegistry([_inv()]),
            "вот код на Kotlin",
            chat,
            user_message="перепиши проект на Kotlin",
        )
        assert len(captured) == 1
        content = captured[0][0]["content"]
        assert "перепиши проект на Kotlin" in content
        assert "вот код на Kotlin" in content

    def test_user_message_empty_defaults_to_blank(self):
        """When user_message is not passed, the prompt still works (blank)."""
        captured = []

        def chat(messages):
            captured.append(messages)
            return json.dumps({"violated_id": "", "rationale": ""})

        audit_reply(InvariantRegistry([_inv()]), "ответ", chat)
        assert len(captured) == 1
        # The User message section should be present but empty.
        content = captured[0][0]["content"]
        assert "User message:\n\n" in content


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


class TestCli:
    def test_one_shot_conflict_exits_3(self, tmp_path, monkeypatch, capsys):
        from llm_bot import cli

        monkeypatch.chdir(tmp_path)
        # stub the stores so no real YAML files are needed
        session = _make_session(
            "cli1",
            self_gate_transport(),
            str(tmp_path / "s"),
            invariants=InvariantRegistry([_inv()]),
        )
        monkeypatch.setattr(cli, "make_session", lambda *a, **k: session)
        code = cli.main([
            "--agent", "assistant",
            "--no-invariants",  # session already carries the registry
            "давай сделаем на flask",
        ])
        out = capsys.readouterr()
        assert code == 3
        assert "STACK-1" in out.out

    def test_list_invariants_flag(self, tmp_path, monkeypatch, capsys):
        from llm_bot import cli

        path = tmp_path / "inv.yaml"
        path.write_text(
            "kind_labels:\n  stack: стек\n"
            "invariants:\n  S1:\n    kind: stack\n    statement: правило\n",
            encoding="utf-8",
        )
        monkeypatch.chdir(tmp_path)
        code = cli.main(["--list-invariants", "--invariants-file", str(path)])
        out = capsys.readouterr()
        assert code == 0
        assert "[S1]" in out.out
        assert "стек" in out.out


def self_gate_transport() -> httpx.BaseTransport:
    """A transport that should never be reached when the gate fires."""
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("request must not reach the transport")

    return httpx.MockTransport(handler)
