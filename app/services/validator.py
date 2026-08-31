"""
Validates and normalizes the raw ingested DataFrame into a list of
TicketRecord objects. Records that fail hard requirements (no ticket_id AND
no description at all) are rejected and reported back to the user rather
than silently dropped - quality analysis on invisible data-loss is worse
than no analysis at all.
"""
import pandas as pd

from app.models.schemas import TicketRecord

REQUIRED_ANY = ["short_description", "description", "worklog"]


def _coerce_datetime(value) -> pd.Timestamp | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    try:
        ts = pd.to_datetime(value, errors="coerce")
        return None if pd.isna(ts) else ts.to_pydatetime()
    except Exception:
        return None


def _clean_str(value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip()


def normalize_and_validate(df: pd.DataFrame) -> tuple[list[TicketRecord], list[dict]]:
    """
    Returns (valid_records, rejected_rows).
    rejected_rows: [{"source_row": int, "reason": str}]
    """
    valid: list[TicketRecord] = []
    rejected: list[dict] = []

    if df.empty:
        return valid, rejected

    # Ensure expected columns exist even if the source lacked them
    for col in ["ticket_id", "short_description", "description", "worklog",
                "priority", "status", "assignment_group", "opened_at", "closed_at"]:
        if col not in df.columns:
            df[col] = None

    # Drop fully-empty rows first (common in exported Excel sheets)
    df = df.dropna(how="all")

    seen_ids: set[str] = set()
    for idx, row in df.reset_index(drop=True).iterrows():
        ticket_id = _clean_str(row.get("ticket_id")) or f"AUTO-{idx+1}"
        short_desc = _clean_str(row.get("short_description"))
        desc = _clean_str(row.get("description"))
        worklog = _clean_str(row.get("worklog"))

        if not any([short_desc, desc, worklog]):
            rejected.append({"source_row": idx, "reason": "No description or worklog content found."})
            continue

        # De-duplicate on ticket_id, keep first occurrence, flag the rest
        if ticket_id in seen_ids:
            rejected.append({"source_row": idx, "reason": f"Duplicate ticket_id '{ticket_id}'."})
            continue
        seen_ids.add(ticket_id)

        record = TicketRecord(
            ticket_id=ticket_id,
            short_description=short_desc,
            description=desc,
            worklog=worklog,
            priority=_clean_str(row.get("priority")) or None,
            status=_clean_str(row.get("status")) or None,
            assignment_group=_clean_str(row.get("assignment_group")) or None,
            opened_at=_coerce_datetime(row.get("opened_at")),
            closed_at=_coerce_datetime(row.get("closed_at")),
            source_row=idx,
        )
        valid.append(record)

    return valid, rejected
