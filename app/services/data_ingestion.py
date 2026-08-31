"""
Ingests Excel / CSV / free-form text into a common raw pandas DataFrame.
This layer does NOT validate business rules (see validator.py) - it only
handles "get bytes into a table" concerns: encoding detection, delimiter
sniffing, header discovery, and turning unstructured text into pseudo-rows.
"""
import io
import re

import chardet
import pandas as pd

# Common ITSM column name variants -> canonical name
COLUMN_ALIASES: dict[str, str] = {
    "incident id": "ticket_id",
    "incident number": "ticket_id",
    "ticket id": "ticket_id",
    "ticket number": "ticket_id",
    "id": "ticket_id",
    "number": "ticket_id",
    "short description": "short_description",
    "summary": "short_description",
    "title": "short_description",
    "description": "description",
    "details": "description",
    "worklog": "worklog",
    "work log": "worklog",
    "work notes": "worklog",
    "resolution notes": "worklog",
    "resolution": "worklog",
    "priority": "priority",
    "state": "status",
    "status": "status",
    "assignment group": "assignment_group",
    "assigned group": "assignment_group",
    "group": "assignment_group",
    "opened": "opened_at",
    "opened at": "opened_at",
    "created": "opened_at",
    "closed": "closed_at",
    "resolved": "closed_at",
    "closed at": "closed_at",
}


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {}
    for col in df.columns:
        key = str(col).strip().lower()
        if key in COLUMN_ALIASES:
            rename_map[col] = COLUMN_ALIASES[key]
    return df.rename(columns=rename_map)


def load_excel(file_bytes: bytes) -> pd.DataFrame:
    df = pd.read_excel(io.BytesIO(file_bytes), engine="openpyxl")
    return _normalize_columns(df)


def load_csv(file_bytes: bytes) -> pd.DataFrame:
    encoding = chardet.detect(file_bytes[:20000]).get("encoding") or "utf-8"
    try:
        df = pd.read_csv(io.BytesIO(file_bytes), encoding=encoding, sep=None, engine="python")
    except Exception:
        # Fall back to strict utf-8 with comma delimiter
        df = pd.read_csv(io.BytesIO(file_bytes), encoding="utf-8", errors="replace")
    return _normalize_columns(df)


_TICKET_BLOCK_SPLIT = re.compile(r"\n(?=(?:INC|TICKET|CASE)[-_ ]?\d+)", re.IGNORECASE)
_FIELD_LINE = re.compile(r"^\s*([A-Za-z /]+)\s*[:\-]\s*(.+)$")


def load_unstructured_text(text: str) -> pd.DataFrame:
    """
    Best-effort parse of free-form incident text/logs into pseudo-rows.
    Strategy:
      1. Split on lines that look like a new ticket header (INC123, TICKET-45, ...).
      2. Within each block, pull "Field: value" style lines into columns.
      3. Whatever isn't matched as a field goes into `description`.
    This is intentionally forgiving - unstructured input is messy by
    definition - and always yields at least one row so nothing is silently
    dropped.
    """
    text = text.strip()
    if not text:
        return pd.DataFrame(columns=["ticket_id", "description"])

    blocks = _TICKET_BLOCK_SPLIT.split(text)
    if len(blocks) == 1:
        blocks = [b for b in re.split(r"\n{2,}", text) if b.strip()] or [text]

    rows = []
    for i, block in enumerate(blocks):
        record: dict[str, str] = {}
        leftover_lines = []
        header_match = re.search(r"\b((?:INC|TICKET|CASE)[-_ ]?\d+)\b", block, re.IGNORECASE)

        for line in block.splitlines():
            m = _FIELD_LINE.match(line)
            if m:
                key = m.group(1).strip().lower()
                canonical = COLUMN_ALIASES.get(key, key.replace(" ", "_"))
                record[canonical] = m.group(2).strip()
            elif line.strip() and not (header_match and line.strip() == header_match.group(1).strip()):
                leftover_lines.append(line.strip())

        record["ticket_id"] = header_match.group(1) if header_match else f"TXT-{i+1}"

        if leftover_lines:
            record["description"] = (record.get("description", "") + " " + " ".join(leftover_lines)).strip()

        rows.append(record)

    return pd.DataFrame(rows)


def ingest(filename: str, file_bytes: bytes) -> pd.DataFrame:
    """Dispatches to the right loader based on file extension."""
    lower = filename.lower()
    if lower.endswith((".xlsx", ".xls")):
        return load_excel(file_bytes)
    if lower.endswith(".csv"):
        return load_csv(file_bytes)
    if lower.endswith(".txt"):
        text = file_bytes.decode(chardet.detect(file_bytes).get("encoding") or "utf-8", errors="replace")
        return load_unstructured_text(text)
    raise ValueError(f"Unsupported file extension for: {filename}")
