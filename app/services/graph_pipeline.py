"""
LangGraph pipeline: a small per-ticket state machine with `classify` and
`score` nodes, processed for many tickets at once via `.abatch()` with
bounded concurrency (LLM_MAX_CONCURRENCY). This is where the two
genuinely-per-ticket, LLM-involving steps happen:

  - classify: only for tickets the cheap keyword/embedding pre-pass
    (categorizer.py) couldn't resolve - `state["category"]` is already
    filled in for everything else, so this node is a no-op for most
    tickets in a typical batch.
  - score: always runs (the heuristic rubric is free), optionally blends
    in an LLM judgment when a chat model is configured.

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

from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from app.models.schemas import CATEGORIES
from app.security import sanitize_for_llm
from app.services.json_utils import extract_json
from app.services.llm_client import SYSTEM_GUARDRAIL
from app.services.scorer import heuristic_score

logger = logging.getLogger(__name__)


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
        prompt = (
            f"Classify the ticket below into exactly one of these categories: "
            f"{categories}.\n<<<DATA>>>\n{safe_text}\n<<<END_DATA>>>\n"
            'Respond as JSON only: {"category": "...", "confidence": 0.0}'
        )
        response = await chat_model.ainvoke(
            [{"role": "system", "content": SYSTEM_GUARDRAIL}, {"role": "user", "content": prompt}]
        )
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

    if chat_model is not None and state["worklog"].strip():
        try:
            prompt = (
                "Score this ITSM worklog's quality from 0-100 based on: clarity, "
                "root-cause documented, resolution steps documented, timestamps/"
                "actions present, professionalism.\n"
                f"<<<DATA>>>\nContext: {sanitize_for_llm(state['text'])}\nWorklog: {sanitize_for_llm(state['worklog'])}\n<<<END_DATA>>>\n"
                'Respond as JSON only: {"score": 0, "breakdown": {}, "flags": ["..."]}'
            )
            response = await chat_model.ainvoke(
                [{"role": "system", "content": SYSTEM_GUARDRAIL}, {"role": "user", "content": prompt}]
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
