"""
LangGraph pipeline: a small per-ticket state machine with `classify` and
`score` nodes, processed for many tickets at once via `.abatch()` with
bounded concurrency (LLM_MAX_CONCURRENCY).

  - classify: only for tickets the cheap keyword/embedding pre-pass
    (categorizer.py) couldn't resolve - `state["category"]` is already
    filled in for everything else, so this node is a no-op for most
    tickets in a typical batch.
  - score: the heuristic rubric (scorer.py) always runs and is free/instant.
    Blending in an LLM judgment on top only happens if
    ENABLE_LLM_WORKLOG_SCORING is set - it's optional nuance, not required
    for a usable score, and was previously the dominant cause of slowness:
    it ran for EVERY ticket with a worklog regardless of whether
    categorization was already resolved for free, so even a 10-row batch
    where every ticket matched a keyword rule still triggered 10 extra LLM
    round-trips just for scoring.

Both LLM calls use LangChain's expression-language chain pattern
(`prompt | chat_model`) rather than passing raw message dicts to
`.ainvoke()` directly - matches the documented gen_ai_hub.proxy.langchain
usage pattern. Still invoked via `.ainvoke()` (the async form of the same
chain), not blocking `.invoke()` in a loop - for a batch of N tickets,
awaiting N calls concurrently (bounded by LLM_MAX_CONCURRENCY) is what
keeps wall-clock time close to a single call's latency instead of N times
that; switching to sync `.invoke()` per ticket would make batches slower,
not faster.

Why the pre-pass lives outside this graph: LangGraph's per-item StateGraph
model is a natural fit for "run this same small flow over N independent
things concurrently" - which is exactly what classify/score need. But
batching many tickets into ONE embedding call is fundamentally different
from "run N independent things" (it's "run ONE thing over N inputs"), so
that stays as a flat batch pass in categorizer.py rather than forcing it
through a per-item graph node.

429 handling: the chat model returned by llm_client.get_chat_model() has
LangChain's `.with_retry()` already applied (exponential backoff + jitter
on openai.RateLimitError). `abatch`'s `max_concurrency` bounds how many
tickets are in flight at once, so a 3k-row batch doesn't fire thousands of
simultaneous requests in the first place - retry handles the occasional
429 that still slips through, bounded concurrency prevents causing a storm
of them.
"""
import json
import logging

from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from app.config import get_settings
from app.models.schemas import CATEGORIES
from app.security import sanitize_for_llm
from app.services.json_utils import extract_json
from app.services.llm_client import SYSTEM_GUARDRAIL
from app.services.scorer import heuristic_score

logger = logging.getLogger(__name__)
settings = get_settings()

_CLASSIFY_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_GUARDRAIL),
        (
            "user",
            "Classify the ticket below into exactly one of these categories: "
            "{categories}.\n<<<DATA>>>\n{text}\n<<<END_DATA>>>\n"
            'Respond as JSON only: {{"category": "...", "confidence": 0.0}}',
        ),
    ]
)

_SCORE_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_GUARDRAIL),
        (
            "user",
            "Score this ITSM worklog's quality from 0-100 based on: clarity, "
            "root-cause documented, resolution steps documented, timestamps/"
            "actions present, professionalism.\n"
            "<<<DATA>>>\nContext: {context}\nWorklog: {worklog}\n<<<END_DATA>>>\n"
            'Respond as JSON only: {{"score": 0, "breakdown": {{}}, "flags": ["..."]}}',
        ),
    ]
)


class TicketState(TypedDict):
    ticket_id: str
    text: str  # combined short_description + description, used for classification/scoring context
    worklog: str
    category: str | None  # pre-filled by the keyword/embedding pre-pass; None means "needs LLM"
    category_confidence: float
    category_method: str
    worklog_score: int
    worklog_flags: list[str]
    error: str | None


def _log_node_error(node_name: str, ticket_id: str, exc: Exception) -> str:
    msg = f"{node_name} failed for ticket {ticket_id}: {exc}"
    logger.info(msg)
    return msg


async def classify_node(state: TicketState, chat_model) -> TicketState:
    if state.get("category") is not None:
        return state  # already resolved by the keyword/embedding pre-pass

    if chat_model is None or not state["text"].strip():
        return {**state, "category": "Uncategorized", "category_confidence": 0.0, "category_method": "fallback"}

    try:
        safe_text = sanitize_for_llm(state["text"])
        categories = [c for c in CATEGORIES if c != "Uncategorized"]
        chain = _CLASSIFY_PROMPT | chat_model
        response = await chain.ainvoke({"categories": categories, "text": safe_text})
        parsed = json.loads(extract_json(response.content))
        category = parsed.get("category")
        if category not in CATEGORIES:
            raise ValueError(f"LLM returned an unrecognized category: {category!r}")
        return {
            **state,
            "category": category,
            "category_confidence": float(parsed.get("confidence", 0.5)),
            "category_method": "llm",
        }
    except Exception as exc:
        error = _log_node_error("classify_node", state["ticket_id"], exc)
        return {**state, "category": "Uncategorized", "category_confidence": 0.0, "category_method": "fallback", "error": error}


async def score_node(state: TicketState, chat_model) -> TicketState:
    base = heuristic_score(state["worklog"])
    score, flags = base.score, list(base.flags)

    if settings.ENABLE_LLM_WORKLOG_SCORING and chat_model is not None and state["worklog"].strip():
        try:
            chain = _SCORE_PROMPT | chat_model
            response = await chain.ainvoke(
                {"context": sanitize_for_llm(state["text"]), "worklog": sanitize_for_llm(state["worklog"])}
            )
            parsed = json.loads(extract_json(response.content))
            llm_score = parsed.get("score")
            if isinstance(llm_score, (int, float)):
                score = max(0, min(100, round((score + llm_score) / 2)))
                flags = list(dict.fromkeys(flags + parsed.get("flags", [])))
        except Exception as exc:
            # Heuristic score already computed - an LLM scoring failure
            # degrades gracefully rather than losing the ticket's score.
            _log_node_error("score_node", state["ticket_id"], exc)

    return {**state, "worklog_score": score, "worklog_flags": flags}


def build_graph(chat_model):
    """Builds the compiled per-ticket graph: START -> classify -> score -> END."""

    async def _classify(state: TicketState) -> TicketState:
        return await classify_node(state, chat_model)

    async def _score(state: TicketState) -> TicketState:
        return await score_node(state, chat_model)

    graph = StateGraph(TicketState)
    graph.add_node("classify", _classify)
    graph.add_node("score", _score)
    graph.add_edge(START, "classify")
    graph.add_edge("classify", "score")
    graph.add_edge("score", END)
    return graph.compile()
