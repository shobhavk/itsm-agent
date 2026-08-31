"""End-to-end analysis pipeline used by both the API and the Gradio UI."""
from app.config import get_settings
from app.models.schemas import AnalysisResponse, AnalyzedTicket
from app.services.categorizer import Categorizer
from app.services.data_ingestion import ingest, load_unstructured_text
from app.services.llm_client import get_llm_client
from app.services.scorer import score_worklog
from app.services.validator import normalize_and_validate

settings = get_settings()


def run_pipeline_from_bytes(filename: str, file_bytes: bytes) -> AnalysisResponse:
    df = ingest(filename, file_bytes)
    return _run_pipeline(df)


def run_pipeline_from_text(raw_text: str) -> AnalysisResponse:
    df = load_unstructured_text(raw_text)
    return _run_pipeline(df)


def _run_pipeline(df) -> AnalysisResponse:
    valid_records, rejected = normalize_and_validate(df)

    llm_client = get_llm_client()
    categorizer = Categorizer(llm_client, similarity_threshold=settings.SIMILARITY_THRESHOLD)

    results: list[AnalyzedTicket] = []
    category_counts: dict[str, int] = {}
    total_score = 0

    for rec in valid_records:
        combined_text = " ".join(filter(None, [rec.short_description, rec.description])).strip()
        cat_result = categorizer.categorize(rec.ticket_id, combined_text or rec.worklog)
        wl_result = score_worklog(rec.ticket_id, rec.worklog, combined_text, llm_client)

        category_counts[cat_result.category] = category_counts.get(cat_result.category, 0) + 1
        total_score += wl_result.score

        validation_flags = []
        if not rec.short_description and not rec.description:
            validation_flags.append("Missing description - categorized from worklog only.")
        if not rec.worklog:
            validation_flags.append("No worklog present.")

        results.append(
            AnalyzedTicket(
                ticket_id=rec.ticket_id,
                short_description=rec.short_description,
                description=rec.description,
                worklog=rec.worklog,
                priority=rec.priority,
                status=rec.status,
                assignment_group=rec.assignment_group,
                category=cat_result.category,
                category_confidence=round(cat_result.confidence, 2),
                category_method=cat_result.method,
                worklog_score=wl_result.score,
                worklog_flags=wl_result.flags,
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
