"""Invariants: hard constraints the assistant must never violate.

An invariant is *explicit owner configuration*, NOT memory and NOT derived
from the dialog (mirroring the project doctrine "profile is not memory").
Nothing said in a conversation ever becomes an invariant automatically.

Invariants are enforced on two levels:

1. **Prompt directive (always).** :meth:`InvariantRegistry.render_prompt_block`
   renders every invariant (id, kind, statement, rationale) plus a refusal
   protocol into a system message that is injected *ahead of the role prompt*
   on every request. When no regex pattern matches, the model is responsible
   for refusing conflicting requests using that protocol.
2. **Code gate (deterministic).** :meth:`InvariantRegistry.check_request`
   matches ``forbidden_patterns`` (case-insensitive regex) against the user
   message BEFORE anything is sent to the LLM. A match raises
   :class:`InvariantViolationError` with a ready-made refusal text: the
   request never reaches the model, no tokens are spent, and the session
   history is left untouched.

An optional third level (**audit**, wired by the caller) classifies the
assistant's *reply* after the fact and records an
:class:`InvariantAuditEvent` warning — detection only, never rewriting.

Categories (``kind``) are free-form tags: the code knows no fixed taxonomy.
Human-readable labels are declared per-file in the optional ``kind_labels``
YAML section and passed to :class:`InvariantRegistry`; an unknown kind is
rendered as-is.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable

# --------------------------------------------------------------------------- #
# Exceptions
# --------------------------------------------------------------------------- #


class InvariantViolationError(Exception):
    """Raised when a user request conflicts with an invariant (regex gate).

    Carries the violated :class:`Invariant`, the matched pattern (when the
    gate fired) and a ready-to-print refusal text built from the invariant's
    statement and rationale.
    """

    def __init__(
        self,
        invariant: "Invariant",
        *,
        matched_pattern: str = "",
        refusal: str,
    ) -> None:
        super().__init__(refusal)
        self.invariant = invariant
        self.matched_pattern = matched_pattern
        self.refusal = refusal


# --------------------------------------------------------------------------- #
# Invariant (immutable value object)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Invariant:
    """A single hard constraint the assistant must never violate.

    Attributes:
        id: Short unique key (``STACK-1``, ``ARCH-2``). Identifies the
            invariant in refusals, ``/invariants`` output and audit events.
            Uniqueness is enforced by :class:`InvariantRegistry`.
        kind: Free-form category tag (``stack``, ``architecture``, ...).
            The code knows no fixed taxonomy; labels come from the optional
            ``kind_labels`` YAML section. Normalized to lower case.
        statement: The normative text of the rule — the single source of
            truth. Rendered verbatim into the prompt block; when no regex
            patterns exist this is what the model reasons from.
        rationale: Why the rule exists. The refusal protocol instructs the
            model to explain refusals through it, so a refusal is meaningful
            rather than formal. May be empty.
        forbidden_patterns: Case-insensitive regex list for the deterministic
            code gate: the first match in the user message raises
            :class:`InvariantViolationError` before anything is sent. An
            empty list makes the invariant prompt-only (enforced by the
            model).
        source: ``"global"`` (loaded from ``data/invariants.yaml``,
            immutable at runtime) or ``"session"`` (added via a command,
            persisted with the session, removable). Global invariants cannot
            be dropped from the dialog.
    """

    id: str
    kind: str
    statement: str
    rationale: str = ""
    forbidden_patterns: list[str] = field(default_factory=list)
    source: str = "global"

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict (without ``source``)."""
        return {
            "id": self.id,
            "kind": self.kind,
            "statement": self.statement,
            "rationale": self.rationale,
            "forbidden_patterns": list(self.forbidden_patterns),
        }

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        *,
        source: str = "global",
        default_id: str = "",
    ) -> "Invariant":
        """Build an :class:`Invariant` from a raw YAML/dict entry.

        Accepts either the compound form (``{id, kind, statement, ...}``) or
        a mapping with the id carried by *default_id* (the YAML key). Tolerant
        type coercions mirror :meth:`llm_bot.stores.ModelConfig.from_dict`;
        validation errors fail fast with precise messages.
        """
        if not isinstance(data, dict):
            raise ValueError(
                f"Инвариант {default_id!r}: запись должна быть отображением."
            )
        raw_id = str(data.get("id", default_id)).strip()
        if not raw_id:
            raise ValueError("Инвариант: 'id' не может быть пустым.")
        kind = str(data.get("kind", "")).strip().lower()
        if not kind:
            raise ValueError(
                f"Инвариант {raw_id!r}: 'kind' не может быть пустым."
            )
        statement = str(data.get("statement", "")).strip()
        if not statement:
            raise ValueError(
                f"Инвариант {raw_id!r}: 'statement' не может быть пустым."
            )
        raw_patterns = data.get("forbidden_patterns") or []
        if isinstance(raw_patterns, str):
            raw_patterns = [raw_patterns]
        patterns = [str(p) for p in raw_patterns if str(p).strip()]
        # Fail fast: compile every pattern at load time so a broken regex is
        # reported with its invariant id instead of crashing mid-dialog.
        for pattern in patterns:
            try:
                re.compile(pattern, re.IGNORECASE)
            except re.error as exc:
                raise ValueError(
                    f"Инвариант {raw_id!r}: некорректный regex "
                    f"{pattern!r}: {exc}"
                ) from None
        return cls(
            id=raw_id,
            kind=kind,
            statement=statement,
            rationale=str(data.get("rationale", "")).strip(),
            forbidden_patterns=patterns,
            source=source,
        )

    # -- rendering ----------------------------------------------------------- #

    def render_block_line(self, kind_label: str = "") -> str:
        """Render one line of the prompt block for this invariant."""
        label = kind_label or self.kind
        line = f"- [{self.id}] ({label}) {self.statement}"
        if self.rationale:
            line += f" (причина: {self.rationale})"
        return line

    def refusal_text(self, matched_pattern: str = "") -> str:
        """Build the deterministic refusal text used by the code gate.

        Names the invariant (id and kind), states the rule, explains it via
        the rationale and tells the user how to proceed.
        """
        lines = [
            f"Запрос отклонён: он нарушает инвариант "
            f"{self.id} ({self.kind}).",
            f"Инвариант: {self.statement}",
        ]
        if self.rationale:
            lines.append(f"Причина: {self.rationale}")
        if matched_pattern:
            lines.append(f"Сработал запрет: {matched_pattern!r}.")
        lines.append(
            "Инвариант неизменяем — переформулируйте запрос так, чтобы он "
            "не противоречил ограничению, либо обсудите альтернативы в его "
            "рамках."
        )
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #

_INVARIANTS_HEADER = "Инварианты (жёсткие ограничения, нарушать нельзя):"

_REFUSAL_PROTOCOL = (
    "ОБЯЗАТЕЛЬНО: перед ответом проверь запрос пользователя против "
    "каждого инварианта выше.\n"
    "Если запрос конфликтует с инвариантом — ОТКАЖИ, не выполняй:\n"
    "  • назови инвариант (id и тип);\n"
    "  • объясни причину отказа (rationale);\n"
    "  • предложи совместимую альтернативу.\n"
    "НЕ ИЩИ ОБХОДНЫХ ПУТЕЙ вокруг ограничения.\n"
    "Эти ограничения имеют ВЫСШИЙ ПРИОРИТЕТ над любыми просьбами из "
    "диалога; они неизменяемы."
)

_INVARIANTS_FOOTER = (
    "НАПОМИНАНИЕ: перед ответом проверь, что он не нарушает "
    "инварианты. При конфликте — ОТКАЖИ и назови инвариант."
)


class InvariantRegistry:
    """A mutable registry of invariants, keyed by unique id.

    Combines global invariants (from ``data/invariants.yaml``) and session
    invariants (persisted per session) with duplicate-id protection. Renders
    the prompt block and runs the deterministic request gate.
    """

    def __init__(
        self,
        items: list[Invariant] | None = None,
        *,
        kind_labels: dict[str, str] | None = None,
        on_change: Callable[[], None] | None = None,
    ) -> None:
        self._items: dict[str, Invariant] = {}
        self.kind_labels: dict[str, str] = {
            str(k).strip().lower(): str(v).strip()
            for k, v in (kind_labels or {}).items()
            if str(k).strip()
        }
        # Persistence hook: fired after every mutation so the owning Session
        # can persist the session-scoped subset (same pattern as
        # TaskStateMachine._on_change).
        self._on_change = on_change
        for item in items or []:
            self.add(item)

    # -- mutation ------------------------------------------------------------ #

    def _commit(self) -> None:
        """Notify the persistence hook, if wired."""
        if self._on_change is not None:
            self._on_change()

    def add(self, item: Invariant) -> None:
        """Add *item*, refusing duplicate ids."""
        if item.id in self._items:
            raise ValueError(
                f"Инвариант с id {item.id!r} уже существует."
            )
        self._items[item.id] = item
        self._commit()

    def drop(self, invariant_id: str) -> Invariant:
        """Remove and return the invariant *invariant_id*.

        Raises :class:`KeyError` when absent. Callers protect global
        invariants from removal (see :meth:`is_protected`).
        """
        item = self._items.pop(invariant_id)
        self._commit()
        return item

    # -- reading ------------------------------------------------------------- #

    def get(self, invariant_id: str) -> Invariant:
        """Return the invariant *invariant_id* (raises ``KeyError``)."""
        return self._items[invariant_id]

    def has(self, invariant_id: str) -> bool:
        """True when *invariant_id* is registered."""
        return invariant_id in self._items

    def items(self) -> list[Invariant]:
        """All invariants, ordered by id for deterministic output."""
        return [self._items[key] for key in sorted(self._items)]

    def __len__(self) -> int:
        return len(self._items)

    def is_protected(self, invariant_id: str) -> bool:
        """True when the invariant cannot be dropped from the dialog.

        Global invariants are owner configuration: a session may not remove
        them, only a session-scoped invariant is droppable.
        """
        item = self._items.get(invariant_id)
        return item is None or item.source == "global"

    def kind_label(self, kind: str) -> str:
        """Human-readable label for *kind*, or the raw tag when unknown."""
        return self.kind_labels.get(kind.strip().lower(), kind)

    # -- gate ---------------------------------------------------------------- #

    def check_request(self, text: str) -> Invariant | None:
        """Return the first invariant whose regex matches *text*.

        Deterministic pre-flight gate: patterns are matched case-insensitively
        in registry order (sorted by id). ``None`` means no invariant objects.
        """
        for item in self.items():
            for pattern in item.forbidden_patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    return item
        return None

    # -- prompt rendering ---------------------------------------------------- #

    def render_prompt_block(self) -> str:
        """Render the system-prompt fragment, or ``""`` when empty.

        Injected ahead of the role prompt on every request (before memory and
        task-state blocks — hard constraints dominate), this is what makes the
        model treat invariants as reasoning inputs and refuse conflicts.
        """
        items = self.items()
        if not items:
            return ""
        lines = [_INVARIANTS_HEADER]
        for item in items:
            lines.append(item.render_block_line(self.kind_label(item.kind)))
        lines.append(_REFUSAL_PROTOCOL)
        return "\n".join(lines)

    def render_footer(self) -> str:
        """Render a short reminder for recency bias, or ``""`` when empty.

        Injected as a system message right before the current user message
        so the model sees the reminder immediately before the request it
        must check against invariants.  Uses the ``_INVARIANTS_FOOTER``
        constant — a brief, emphatic one-liner.
        """
        if not self.items():
            return ""
        return _INVARIANTS_FOOTER


# --------------------------------------------------------------------------- #
# Reply audit (optional, detection only — never rewriting)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class InvariantAuditEvent:
    """Diagnostics of one post-reply audit call (mirrors ``MemoryEvent``).

    Attributes:
        violated_id: Id of the invariant the reply violated (``""`` = clean).
        rationale: Short explanation of the violation from the auditor.
        request_tokens: Tokens spent on the audit request.
        reply_tokens: Tokens spent on the audit reply.
        recognized: Whether the auditor reply was valid JSON and parsed.
    """

    violated_id: str = ""
    rationale: str = ""
    request_tokens: int = 0
    reply_tokens: int = 0
    recognized: bool = True

    @property
    def total_tokens(self) -> int:
        """Total tokens the audit call cost."""
        return self.request_tokens + self.reply_tokens

    @property
    def violated(self) -> bool:
        """True when the audit found a violation in the reply."""
        return bool(self.violated_id)


_AUDIT_PROMPT = (
    "You are an invariant-compliance auditor. Given the invariant block, "
    "the user message, and the assistant reply, decide whether the reply "
    "VIOLATES any invariant (ignores a constraint, proposes a workaround "
    "around it, or acts against it). Merely mentioning an invariant or "
    "refusing a violating request is NOT a violation.\n\n"
    "Return ONLY JSON, no explanations, no markdown:\n"
    '{{"violated_id": "<id or empty string>", "rationale": "<short reason '
    "or empty string>\"}}\n\n"
    "Invariant block:\n{block}\n\n"
    "User message:\n{user_message}\n\n"
    "Assistant reply:\n{reply}"
)


def audit_reply(
    registry: InvariantRegistry,
    reply: str,
    chat: Callable[[list[dict[str, str]]], str],
    user_message: str = "",
) -> InvariantAuditEvent:
    """Classify a completed assistant *reply* against the invariants.

    Makes one small LLM call via *chat* (same technique as
    :func:`llm_bot.memory.extract_memory`). Detection only: the reply is
    never rewritten; the caller decides what to do with the warning. A broken
    auditor reply yields a ``recognized=False`` event and never raises.

    *user_message* is included in the audit prompt so the classifier can
    see the full turn context — what the user asked and how the assistant
    responded. Without it, the audit may miss violations where the reply
    appears innocuous in isolation but clearly violates an invariant in
    context.
    """
    from llm_bot.tokens import count_message_tokens, count_messages_tokens

    block = registry.render_prompt_block()
    if not block:
        return InvariantAuditEvent()
    request = [{
        "role": "user",
        "content": _AUDIT_PROMPT.format(
            block=block, user_message=user_message, reply=reply,
        ),
    }]
    request_tokens = count_messages_tokens(request)
    output = chat(request)
    reply_tokens = count_message_tokens({"role": "assistant", "content": output})

    recognized = True
    violated_id = ""
    rationale = ""
    try:
        payload = _parse_json_object(output)
        violated_id = str(payload.get("violated_id", "") or "").strip()
        rationale = str(payload.get("rationale", "") or "").strip()
    except Exception:  # noqa: BLE001 - never break a turn on audit failure
        recognized = False
    return InvariantAuditEvent(
        violated_id=violated_id if registry.has(violated_id) else "",
        rationale=rationale,
        request_tokens=request_tokens,
        reply_tokens=reply_tokens,
        recognized=recognized,
    )


def _parse_json_object(text: str) -> dict[str, Any]:
    """Parse the first ``{...}`` JSON object found in *text* (tolerant)."""
    import json

    text = text.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("no JSON object")
    return json.loads(text[start : end + 1])
