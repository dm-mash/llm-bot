"""Support assistant: answers product questions from the FAQ and a customer's ticket.

The assistant is the whole day-33 service in one object. A question comes in,
and the answer comes out with the evidence that produced it:

* **RAG** over the product knowledge base (``faq/``) — how the product works,
  what the documented procedure is;
* **MCP** to the support desk (:mod:`llm_bot.mcp_servers.support`) — what is
  actually true about *this* customer's account and their open tickets.

Both halves are needed and neither is enough. The FAQ knows that a password
reset link lives 30 minutes; the ticket knows that this particular customer has
already asked twice and the mail does not arrive. An answer from one half alone
either talks past the customer or invents a policy.

The retriever (and therefore the embedding model) is built once and shared by
every conversation: it is only ever read, and a fresh one per request would
repay tens of CPU seconds on every call.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from llm_bot.factory import build_retriever, make_session
from llm_bot.rag import Retriever
from llm_bot.tokens import TokenUsage

DEFAULT_AGENT = "support"
DEFAULT_RAG_INDEX = "data/faq_emb/index_structure.json"
DEFAULT_MCP_SERVERS = ("support",)

# The prompt fragment naming the case under discussion. The model still has to
# fetch the ticket itself — naming the id is a pointer, not the data, so the
# answer cannot be built from a stale summary baked in here.
def ticket_directive(ticket_id: str, user_id: str | None = None) -> str:
    """Return the per-turn instruction that puts *ticket_id* in front of the model.

    Used both by the support assistant and by the chat endpoint's ``X-Ticket-Id``
    header, so the wording of "there is a ticket behind this question" lives in
    one place. Only the ids travel: the ticket itself is read through MCP.

    The tool is named on purpose. A vaguer "прочитай обращение" was measured to
    be answered straight from the FAQ without a single tool call — the model has
    no reason to prefer a tool call over a plausible answer from the prompt, so
    it is told what to open. This directive is attached only when a ticket was
    actually selected, so the earlier finding (naming tools made the model reach
    for them on documentation questions) does not apply: there is no case here
    where a ticket is named and a tool call would be wrong.

    The last clause exists because mixing the two sources under one numbering
    scheme produces citations that mean nothing: the model tagged a fact read
    out of the ticket with ``[1]``, which points at a chunk of the FAQ that
    never mentioned it — and a turn answered through a tool skips the citation
    audit entirely, so the wrong marker reached the operator unchallenged.

    Nothing here calls the person a «клиент». That word is what turned every
    answer into third person: told the ticket belongs to a customer, the model
    wrote «Для Maria Sokolova есть два пути» to someone who had just asked
    «что делать-то?» — the very reader. The case is the reader's own.
    """
    return (
        f"Контекст обращения: это ваше обращение, тикет {ticket_id}"
        f"{f', запись {user_id}' if user_id else ''}.\n"
        f"Сначала вызови get_ticket с id={ticket_id} и учти переписку и данные "
        f"из неё, потом отвечай. Общие правила из документации не отменяют того, "
        f"что происходит в этом обращении.\n"
        f"Номера фрагментов [N] и цитаты в кавычках относятся ТОЛЬКО к "
        f"документации. То, что вернул get_ticket, номера не имеет: не ставь "
        f"после таких фактов [N] и не цитируй их оттуда."
    )


@dataclass(frozen=True)
class SupportAnswer:
    """One answered question, with the evidence behind it.

    Attributes:
        answer: The reply text, addressed to the user who asked.
        sources: Retrieved knowledge-base locations, best first — empty when
            retrieval found nothing, which is itself a signal: the answer was
            not read off the documentation.
        tools: Support-desk tools the assistant actually called, best first.
        grounding: ``"grounded"`` / ``"ungrounded"`` / ``"none"``.
        usage: Token accounting for the turn.
        session_id: Conversation this answer belongs to.
    """

    answer: str
    sources: tuple[str, ...] = ()
    tools: tuple[str, ...] = ()
    grounding: str = "none"
    usage: TokenUsage = field(default_factory=TokenUsage)
    session_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Render as the JSON body the HTTP endpoint returns."""
        return {
            "answer": self.answer,
            "sources": list(self.sources),
            "tools": list(self.tools),
            "grounding": self.grounding,
            "usage": {
                # OpenAI-shaped, same mapping as /v1/chat/completions: the whole
                # context the request carried counts as prompt tokens.
                "prompt_tokens": self.usage.context_tokens,
                "completion_tokens": self.usage.reply_tokens,
                "total_tokens": self.usage.total_tokens,
            },
            "session_id": self.session_id,
        }


def build_question(question: str, *, ticket_id: str | None = None, user_id: str | None = None) -> str:
    """Return the message sent to the model, with the case context prepended.

    Only ids are added — never the ticket body. The assistant reads the ticket
    through MCP, so the context it reasons about is the one the tools returned,
    not a copy this module could have made stale.
    """
    text = (question or "").strip()
    if not ticket_id and not user_id:
        return text
    return f"{ticket_directive(ticket_id or '—', user_id)}\n\n{text}"


class _EphemeralSessionStore:
    """Session store that keeps nothing.

    Support dialogs are per-incident and have no value after the shift ends, and
    the endpoint mints a fresh conversation id per request when the caller does
    not supply one. Writing each of those to ``data/sessions/`` would grow the
    disk without bound, so a support assistant holds its history in memory (the
    live :class:`~llm_bot.agent.Session`) and persists nothing.
    """

    def load(self, session_id: str) -> list[dict[str, str]]:
        return []

    def save(self, session_id: str, history: list[dict[str, str]]) -> None:
        return None

    def list(self) -> list[str]:
        return []


#: How many conversations one assistant keeps live. The endpoint hands out a new
#: id per request when the caller omits one, so an unbounded cache would grow
#: with every unanswered call; the oldest is dropped instead.
MAX_SESSIONS = 64


class SupportAssistant:
    """Answers support questions using the FAQ corpus and the support desk.

    One instance serves every conversation: the retriever and the model config
    are shared, while a separate :class:`~llm_bot.agent.Session` per
    *session_id* keeps the dialog history of one customer out of another's.

    Args:
        agent: Agent name from ``data/agents.yaml``.
        rag_index: Index with the product FAQ. Ignored when *retriever* is given.
            ``None`` means RAG off — it does NOT fall back to a default path,
            because an assistant answering from a corpus nobody selected is
            worse than one that answers nothing.
        retriever: A retriever to reuse instead of building one. The HTTP
            service passes its process-wide retriever: a second one over the
            same index would load the embedding model a second time.
        mcp_servers: Servers the assistant may call. ``None`` means the default
            (``support``); an **empty list means MCP off** — the two are
            different, and conflating them turns a deployment without a support
            desk into a startup error.
        mcp_config: Path to the MCP config. Only used when servers are selected.
    """

    def __init__(
        self,
        *,
        agent: str = DEFAULT_AGENT,
        rag_index: str | Path | None = DEFAULT_RAG_INDEX,
        mcp_servers: list[str] | tuple[str, ...] | None = None,
        mcp_config: str | Path | None = None,
        model_store: Any,
        agent_store: Any,
        session_store: Any = None,
        memory_store: Any = None,
        top_k: int | None = 6,
        rerank: bool = False,
        context_window: int | None = None,
        transport: Any = None,
        retriever: Retriever | None = None,
    ) -> None:
        self.agent = agent
        self.rag_index = rag_index
        self.mcp_servers = list(mcp_servers if mcp_servers is not None else DEFAULT_MCP_SERVERS)
        self.mcp_config = mcp_config
        self.model_store = model_store
        self.agent_store = agent_store
        self.session_store = session_store if session_store is not None else _EphemeralSessionStore()
        self.memory_store = memory_store
        self.transport = transport
        self._sessions: dict[str, Any] = {}
        if retriever is not None:
            # A caller that already holds a retriever for the same corpus (the
            # HTTP service builds one for the chat endpoint) passes it in: a
            # second Retriever over the same index would load the embedding
            # model a second time and keep both copies in memory.
            self._retriever: Retriever | None = retriever
        elif rag_index:
            # Built eagerly so a missing or broken index fails at startup rather
            # than on the first customer question.
            self._retriever = build_retriever(
                rag_index,
                context_window=context_window,
                top_k=top_k,
                rerank=rerank,
            )
        else:
            self._retriever = None

    @property
    def retriever(self) -> Retriever | None:
        """The shared FAQ retriever, or ``None`` when RAG is off."""
        return self._retriever

    def session(self, session_id: str):
        """Return (and cache) the conversation session for *session_id*.

        An empty server list means MCP is off, and is passed on as ``None``
        rather than as ``[]``: an explicit but empty selection against a missing
        config is a startup error by design, and that error must not reach a
        deployment that never asked for tools.
        """
        existing = self._sessions.get(session_id)
        if existing is not None:
            return existing
        use_mcp = self.mcp_servers or None
        created = make_session(
            session_id,
            self.agent,
            model_store=self.model_store,
            agent_store=self.agent_store,
            session_store=self.session_store,
            memory_store=self.memory_store,
            retriever=self._retriever,
            mcp_servers=use_mcp,
            mcp_config_file=str(self.mcp_config) if (self.mcp_config and use_mcp) else None,
            transport=self.transport,
        )
        self._sessions[session_id] = created
        # Bounded: the endpoint mints a conversation id per request when the
        # caller omits one, so an unbounded cache would keep a Session (and its
        # history) alive for every question ever asked.
        while len(self._sessions) > MAX_SESSIONS:
            self._sessions.pop(next(iter(self._sessions)))
        return created

    def forget(self, session_id: str) -> None:
        """Drop a conversation so the next question starts clean."""
        self._sessions.pop(session_id, None)

    def answer(
        self,
        question: str,
        *,
        ticket_id: str | None = None,
        user_id: str | None = None,
        session_id: str = "support-default",
    ) -> SupportAnswer:
        """Answer one question and report the evidence behind it.

        The evidence is scoped to THIS turn: ``mcp_events`` accumulates over the
        whole conversation, and a retrieval that failed or was skipped leaves
        ``last_rag_event`` / ``last_grounding`` pointing at the previous
        question. Reporting those would attribute an earlier turn's sources to
        an answer they did not produce, which is exactly what the operator reads
        the field to find out.

        Raises:
            ValueError: The question is empty.
        """
        text = (question or "").strip()
        if not text:
            raise ValueError("Пустой вопрос.")
        session = self.session(session_id)
        events_before = len(session.mcp_events)
        retrievals_before = len(session.rag_events)
        result = session.chat_with_details(build_question(text, ticket_id=ticket_id, user_id=user_id))

        tools = tuple(dict.fromkeys(e.tool for e in session.mcp_events[events_before:] if e.ok))
        retrieved_now = session.rag_events[retrievals_before:]
        sources = tuple(
            hit.location for hit in retrieved_now[-1].sources
        ) if retrieved_now else ()
        if not retrieved_now:
            grounding = "none"
        else:
            verdict = session.last_grounding
            grounding = "none" if verdict is None else str(getattr(verdict.status, "value", verdict.status))
        return SupportAnswer(
            answer=result.reply,
            sources=sources,
            tools=tools,
            grounding=grounding,
            usage=result.usage,
            session_id=session_id,
        )