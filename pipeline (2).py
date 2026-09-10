"""
End-to-end async analysis pipeline, used by both the API and the Gradio UI.

Flow for N tickets:
  0. Ticket-cache lookup         - real duplicate detection, keyed on
                                     Incident ID (not the uploaded file).
                                     A ticket seen before with unchanged
                                     content is served from ticket_cache
                                     and never reaches the LLM, even if the
                                     rest of the upload is new.
  1. normalize_and_validate       - sync, cheap, unchanged.
  2. Categorizer.pre_resolve      - keyword rules (free) + one batched
                                     embedding call for whatever's left
                                     (not one call per ticket), only for
                                     tickets not already served from cache.
  3. LangGraph classify+score     - graph_pipeline.build_graph(...).abatch()
                                     over one TicketState per remaining
                                     ticket, bounded by LLM_MAX_CONCURRENCY.
                                     classify_node is a no-op for anything
                                     step 2 already resolved; score_node
                                     always runs the heuristic rubric and
                                     optionally blends an LLM judgment.
  4. Merge cache hits + freshly processed tickets, preserving original
     upload order, into AnalyzedTicket / AnalysisResponse. Newly processed
     tickets are written back into ticket_cache for next time.

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
from app.services.host_extraction import extract_host
from app.services.llm_client import get_chat_model, get_embeddings_model
from app.services.persistence import (
    compute_ticket_content_hash,
    get_ticket_cache_bulk,
    save_ticket_cache_bulk,
)
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
            host_counts={},
            average_worklog_score=0.0,
            results=[],
        )

    llm_worklog_scoring_enabled = bool(getattr(settings, "ENABLE_LLM_WORKLOG_SCORING", False))

    # Real duplicate detection: keyed on Incident ID, not the whole file.
    # A ticket seen before (same ID, unchanged content, same config) is
    # served straight from ticket_cache - no LLM call for it at all, even
    # if the rest of this upload is brand new.
    content_hashes = {
        rec.ticket_id: compute_ticket_content_hash(rec.short_description, rec.description, rec.worklog)
        for rec in valid_records
    }
    ticket_cache = get_ticket_cache_bulk(list(content_hashes.keys()))

    cache_hits: list = []
    records_to_process: list = []
    for rec in valid_records:
        cached = ticket_cache.get(rec.ticket_id)
        if (
            cached
            and cached["content_hash"] == content_hashes[rec.ticket_id]
            and cached["llm_worklog_scoring_enabled"] == llm_worklog_scoring_enabled
        ):
            cache_hits.append((rec, cached))
        else:
            records_to_process.append(rec)

    chat_model = get_chat_model()
    embeddings_model = get_embeddings_model()

    combined_texts: dict[str, str] = {
        rec.ticket_id: " ".join(filter(None, [rec.short_description, rec.description])).strip() or rec.worklog
        for rec in records_to_process
    }

    # Step 1: cheap keyword + batched-embedding pre-pass (categorizer.py) -
    # only for tickets that weren't already in the cache.
    categorizer = Categorizer(embeddings_model, similarity_threshold=settings.SIMILARITY_THRESHOLD)
    pre_resolved: dict = {}
    if combined_texts:
        pre_resolved, _still_unresolved = await categorizer.pre_resolve(list(combined_texts.items()))

    # Step 2: LangGraph handles classify (only for anything step 1 left
    # unresolved) + score (always) for every ticket not served from cache,
    # concurrently but bounded by LLM_MAX_CONCURRENCY.
    initial_states: list[TicketState] = []
    for rec in records_to_process:
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

    final_states = []
    if initial_states:
        graph = build_graph(chat_model)
        final_states = await graph.abatch(
            initial_states, config={"max_concurrency": settings.LLM_MAX_CONCURRENCY}, return_exceptions=True
        )

    records_by_id = {rec.ticket_id: rec for rec in valid_records}
    results_by_id: dict[str, AnalyzedTicket] = {}
    category_counts: dict[str, int] = {}
    host_counts: dict[str, int] = {}
    total_score = 0
    new_cache_entries: list[dict] = []

    # Materialize cache hits directly - no categorization/scoring logic
    # runs for these at all.
    for rec, cached in cache_hits:
        category = cached["category"] or "Uncategorized"
        category_counts[category] = category_counts.get(category, 0) + 1
        total_score += cached["worklog_score"]
        host = cached["host"]
        if host:
            host_counts[host] = host_counts.get(host, 0) + 1

        validation_flags = []
        if not rec.short_description and not rec.description:
            validation_flags.append("Missing description - categorized from worklog only.")
        if not rec.worklog:
            validation_flags.append("No worklog present.")
        if not host:
            validation_flags.append("No host/configuration item identified.")
        validation_flags.append("Duplicate Incident ID - served from cache, no LLM call made.")

        results_by_id[rec.ticket_id] = AnalyzedTicket(
            ticket_id=rec.ticket_id,
            short_description=rec.short_description,
            description=rec.description,
            worklog=rec.worklog,
            priority=rec.priority,
            status=rec.status,
            assignment_group=rec.assignment_group,
            configuration_item=rec.configuration_item,
            host=host,
            category=category,
            category_confidence=round(cached["category_confidence"] or 0.0, 2),
            category_method=cached["category_method"] or "fallback",
            worklog_score=cached["worklog_score"],
            worklog_flags=cached["worklog_flags"],
            validation_flags=validation_flags,
        )

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

        host = extract_host(rec.configuration_item, rec.description, rec.worklog)
        if host:
            host_counts[host] = host_counts.get(host, 0) + 1

        validation_flags = []
        if not rec.short_description and not rec.description:
            validation_flags.append("Missing description - categorized from worklog only.")
        if not rec.worklog:
            validation_flags.append("No worklog present.")
        if not host:
            validation_flags.append("No host/configuration item identified.")
        if state.get("error"):
            validation_flags.append(f"Processing note: {state['error']}")

        results_by_id[rec.ticket_id] = AnalyzedTicket(
            ticket_id=rec.ticket_id,
            short_description=rec.short_description,
            description=rec.description,
            worklog=rec.worklog,
            priority=rec.priority,
            status=rec.status,
            assignment_group=rec.assignment_group,
            configuration_item=rec.configuration_item,
            host=host,
            category=category,
            category_confidence=round(state["category_confidence"], 2),
            category_method=state["category_method"] or "fallback",
            worklog_score=state["worklog_score"],
            worklog_flags=state["worklog_flags"],
            validation_flags=validation_flags,
        )

        new_cache_entries.append(
            {
                "ticket_id": rec.ticket_id,
                "content_hash": content_hashes[rec.ticket_id],
                "category": category,
                "category_confidence": round(state["category_confidence"], 2),
                "category_method": state["category_method"] or "fallback",
                "worklog_score": state["worklog_score"],
                "worklog_flags": state["worklog_flags"],
                "host": host,
                "configuration_item": rec.configuration_item,
                "llm_worklog_scoring_enabled": llm_worklog_scoring_enabled,
            }
        )

    if new_cache_entries:
        save_ticket_cache_bulk(new_cache_entries)

    # Preserve original upload order rather than cache-hits-then-fresh order.
    results: list[AnalyzedTicket] = [
        results_by_id[rec.ticket_id] for rec in valid_records if rec.ticket_id in results_by_id
    ]

    avg_score = round(total_score / len(results), 1) if results else 0.0

    return AnalysisResponse(
        total_records=len(valid_records) + len(rejected),
        valid_records=len(valid_records),
        rejected_records=len(rejected),
        category_counts=category_counts,
        host_counts=host_counts,
        average_worklog_score=avg_score,
        results=results,
    )
