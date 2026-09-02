"""
End-to-end async analysis pipeline, used by both the API and the Gradio UI.

Flow for N tickets:
  1. normalize_and_validate       - sync, cheap, unchanged.
  2. Categorizer.pre_resolve      - keyword rules (free) + one batched
                                     embedding call for whatever's left
                                     (not one call per ticket).
  3. LangGraph classify+score     - graph_pipeline.build_graph(...).abatch()
                                     over one TicketState per ticket, bounded
                                     by LLM_MAX_CONCURRENCY. classify_node is
                                     a no-op for anything step 2 already
                                     resolved; score_node always runs the
                                     heuristic rubric and optionally blends
                                     an LLM judgment.
  4. Merge back into AnalyzedTicket / AnalysisResponse.

This whole thing is async now and genuinely non-blocking - the FastAPI
routes and Gradio handlers `await` it directly rather than calling a
synchronous function that would block the event loop for the entire
duration of a large batch.
"""
import logging

from app.config import get_settings
from app.models.schemas import AnalysisResponse, AnalyzedTicket
from app.services.categorizer import Categorizer
from app.services.data_ingestion import ingest, load_unstructured_text
from app.services.graph_pipeline import TicketState, build_graph
from app.services.llm_client import get_chat_model, get_embeddings_model
from app.services.validator import normalize_and_validate

logger = logging.getLogger(__name__)
settings = get_settings()


async def run_pipeline_from_bytes(filename: str, file_bytes: bytes) -> AnalysisResponse:
    df = ingest(filename, file_bytes)
    return await _run_pipeline(df)


async def run_pipeline_from_text(raw_text: str) -> AnalysisResponse:
    df = load_unstructured_text(raw_text)
    return await _run_pipeline(df)


async def _run_pipeline(df) -> AnalysisResponse:
    valid_records, rejected = normalize_and_validate(df)

    if not valid_records:
        return AnalysisResponse(
            total_records=len(valid_records) + len(rejected),
            valid_records=0,
            rejected_records=len(rejected),
            category_counts={},
            average_worklog_score=0.0,
            results=[],
        )

    chat_model = get_chat_model()
    embeddings_model = get_embeddings_model()

    combined_texts: dict[str, str] = {
        rec.ticket_id: " ".join(filter(None, [rec.short_description, rec.description])).strip() or rec.worklog
        for rec in valid_records
    }

    # Step 1: cheap keyword + batched-embedding pre-pass (categorizer.py).
    categorizer = Categorizer(embeddings_model, similarity_threshold=settings.SIMILARITY_THRESHOLD)
    pre_resolved, _still_unresolved = await categorizer.pre_resolve(list(combined_texts.items()))

    # Step 2: LangGraph handles classify (only for anything step 1 left
    # unresolved) + score (always) for every ticket, concurrently but
    # bounded by LLM_MAX_CONCURRENCY.
    initial_states: list[TicketState] = []
    for rec in valid_records:
        pre = pre_resolved.get(rec.ticket_id)
        initial_states.append(
            TicketState(
                ticket_id=rec.ticket_id,
                text=combined_texts[rec.ticket_id],
                worklog=rec.worklog,
                category=pre.category if pre else None,
                category_confidence=pre.confidence if pre else 0.0,
                category_method=pre.method if pre else "",
                worklog_score=0,
                worklog_flags=[],
                error=None,
            )
        )

    graph = build_graph(chat_model)
    final_states = await graph.abatch(
        initial_states, config={"max_concurrency": settings.LLM_MAX_CONCURRENCY}, return_exceptions=True
    )

    records_by_id = {rec.ticket_id: rec for rec in valid_records}
    results: list[AnalyzedTicket] = []
    category_counts: dict[str, int] = {}
    total_score = 0

    for state in final_states:
        if isinstance(state, BaseException):
            # A whole graph invocation blew up for this ticket (should be
            # rare - both nodes already catch their own errors internally
            # and degrade gracefully) - don't let one bad ticket kill the
            # batch, fall back to Uncategorized/zero-score for it.
            logger.error("Graph invocation failed for a ticket: %s", state)
            continue

        rec = records_by_id[state["ticket_id"]]
        category = state["category"] or "Uncategorized"
        category_counts[category] = category_counts.get(category, 0) + 1
        total_score += state["worklog_score"]

        validation_flags = []
        if not rec.short_description and not rec.description:
            validation_flags.append("Missing description - categorized from worklog only.")
        if not rec.worklog:
            validation_flags.append("No worklog present.")
        if state.get("error"):
            validation_flags.append(f"Processing note: {state['error']}")

        results.append(
            AnalyzedTicket(
                ticket_id=rec.ticket_id,
                short_description=rec.short_description,
                description=rec.description,
                worklog=rec.worklog,
                priority=rec.priority,
                status=rec.status,
                assignment_group=rec.assignment_group,
                category=category,
                category_confidence=round(state["category_confidence"], 2),
                category_method=state["category_method"] or "fallback",
                worklog_score=state["worklog_score"],
                worklog_flags=state["worklog_flags"],
                validation_flags=validation_flags,
            )
        )

    avg_score = round(total_score / len(results), 1) if results else 0.0

    return AnalysisResponse(
        total_records=len(valid_records) + len(rejected),
        valid_records=len(valid_records),
        rejected_records=len(rejected),
        category_counts=category_counts,
        average_worklog_score=avg_score,
        results=results,
    )
