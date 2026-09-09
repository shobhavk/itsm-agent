"""
persistence.py
---------------
Persistent storage for the ITSM Quality Analysis app.

Purpose:
  When a user uploads a file, we hash its raw bytes. If we've already
  processed a file with that exact hash, we return the stored results
  directly from the DB instead of re-running classification/scoring
  through the LLM pipeline (LangGraph classify + score nodes).

Storage: SQLite via SQLAlchemy (file-based, no extra infra).
  - Mount the DB file on a Docker volume so it survives container
    restarts, e.g.:
      volumes:
        - itsm_db_data:/app/data
    and point DB_PATH at /app/data/itsm_analysis.db in that environment.

Design notes:
  - Dedup key is SHA-256 of the raw uploaded file bytes -> exact-file
    duplicate detection (not near-duplicate / fuzzy matching).
  - Full processed DataFrame is stored as JSON (records orient) so it
    round-trips exactly, including any truncated/full text columns.
  - We also store which config flags were active (e.g.
    ENABLE_LLM_WORKLOG_SCORING) when the file was processed. If the
    current config no longer matches, the cache is treated as stale
    so results aren't silently wrong after a config change.
  - A per-ticket table is also populated for future features (e.g.
    cross-file history/search) but the DataFrame JSON blob is the
    source of truth for exact reconstruction.
"""

import hashlib
import json
import os
from datetime import datetime, timezone
from typing import Optional

import pandas as pd
from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    Boolean,
    DateTime,
    Text,
    ForeignKey,
    UniqueConstraint,
)
from sqlalchemy.orm import declarative_base, sessionmaker, relationship

DB_PATH = os.environ.get("ITSM_DB_PATH", "./data/itsm_analysis.db")
os.makedirs(os.path.dirname(DB_PATH) or ".", exist_ok=True)

engine = create_engine(f"sqlite:///{DB_PATH}", connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()


class UploadedFile(Base):
    __tablename__ = "uploaded_files"

    id = Column(Integer, primary_key=True)
    file_hash = Column(String(64), unique=True, nullable=False, index=True)
    filename = Column(String(512), nullable=False)
    uploaded_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    ticket_count = Column(Integer, default=0)

    # Config snapshot at time of processing — used to invalidate cache
    # if the pipeline behavior has since changed.
    llm_worklog_scoring_enabled = Column(Boolean, default=False)
    pipeline_version = Column(String(64), default="v1")

    # Full result set, exact round-trip (pandas records-orient JSON).
    results_json = Column(Text, nullable=False)

    tickets = relationship("TicketResult", back_populates="source_file", cascade="all, delete-orphan")

    __table_args__ = (UniqueConstraint("file_hash", name="uq_file_hash"),)


class TicketResult(Base):
    """Optional per-ticket breakout for future querying/history features."""
    __tablename__ = "ticket_results"

    id = Column(Integer, primary_key=True)
    file_id = Column(Integer, ForeignKey("uploaded_files.id"), nullable=False)
    ticket_id = Column(String(128), index=True)
    category = Column(String(256))
    subcategory = Column(String(256))
    quality_score = Column(Integer)

    source_file = relationship("UploadedFile", back_populates="tickets")


Base.metadata.create_all(engine)


def compute_file_hash(file_bytes: bytes) -> str:
    """SHA-256 hash of raw file bytes. Same file contents => same hash,
    regardless of filename or upload time."""
    return hashlib.sha256(file_bytes).hexdigest()


def get_cached_result(
    file_hash: str,
    current_llm_worklog_scoring_enabled: Optional[bool] = None,
) -> Optional[dict]:
    """
    Returns a dict {"full_df": DataFrame, "summary_stats": dict,
    "category_counts": dict, "uploaded_at": str} if this exact file has
    been processed before AND the relevant config hasn't changed since.
    Returns None on cache miss or config mismatch (caller should then
    run the normal LLM pipeline).
    """
    session = SessionLocal()
    try:
        record = session.query(UploadedFile).filter_by(file_hash=file_hash).first()
        if record is None:
            return None

        if (
            current_llm_worklog_scoring_enabled is not None
            and record.llm_worklog_scoring_enabled != current_llm_worklog_scoring_enabled
        ):
            # Config changed since this file was last processed —
            # treat as a miss so results reflect current settings.
            return None

        payload = json.loads(record.results_json)
        return {
            "full_df": pd.DataFrame(payload["full_df"]),
            "summary_stats": payload["summary_stats"],
            "category_counts": payload["category_counts"],
            "uploaded_at": record.uploaded_at.isoformat(),
        }
    finally:
        session.close()


def save_result(
    file_hash: str,
    filename: str,
    full_df: pd.DataFrame,
    summary_stats: dict,
    category_counts: dict,
    llm_worklog_scoring_enabled: bool = False,
    pipeline_version: str = "v1",
) -> None:
    """Persist processed results for a newly-analyzed file. full_df should
    be the untruncated result set (same shape used for CSV export) so the
    cached load can rebuild the on-screen truncated view from it."""
    payload = json.dumps(
        {
            "full_df": json.loads(full_df.to_json(orient="records")),
            "summary_stats": summary_stats,
            "category_counts": category_counts,
        }
    )

    session = SessionLocal()
    try:
        existing = session.query(UploadedFile).filter_by(file_hash=file_hash).first()
        if existing:
            # Same hash re-saved (e.g. forced re-run) — overwrite in place.
            existing.results_json = payload
            existing.ticket_count = len(full_df)
            existing.llm_worklog_scoring_enabled = llm_worklog_scoring_enabled
            existing.pipeline_version = pipeline_version
            existing.uploaded_at = datetime.now(timezone.utc)
            session.query(TicketResult).filter_by(file_id=existing.id).delete()
            file_id = existing.id
        else:
            record = UploadedFile(
                file_hash=file_hash,
                filename=filename,
                ticket_count=len(full_df),
                llm_worklog_scoring_enabled=llm_worklog_scoring_enabled,
                pipeline_version=pipeline_version,
                results_json=payload,
            )
            session.add(record)
            session.flush()  # get record.id before commit
            file_id = record.id

        # Populate lightweight per-ticket rows if the columns exist.
        df = full_df
        cols = df.columns
        ticket_id_col = next((c for c in ["Ticket ID", "ticket_id", "Incident ID"] if c in cols), None)
        category_col = next((c for c in ["Category", "category"] if c in cols), None)
        subcategory_col = next((c for c in ["Subcategory", "subcategory"] if c in cols), None)
        score_col = next((c for c in ["Quality Score", "quality_score"] if c in cols), None)

        if ticket_id_col:
            for _, row in df.iterrows():
                session.add(
                    TicketResult(
                        file_id=file_id,
                        ticket_id=str(row.get(ticket_id_col, "")),
                        category=str(row.get(category_col, "")) if category_col else None,
                        subcategory=str(row.get(subcategory_col, "")) if subcategory_col else None,
                        quality_score=int(row[score_col]) if score_col and pd.notna(row.get(score_col)) else None,
                    )
                )

        session.commit()
    finally:
        session.close()


def file_metadata(file_hash: str) -> Optional[dict]:
    """Small helper for surfacing 'this file was already processed on X' in the UI."""
    session = SessionLocal()
    try:
        record = session.query(UploadedFile).filter_by(file_hash=file_hash).first()
        if record is None:
            return None
        return {
            "filename": record.filename,
            "uploaded_at": record.uploaded_at.isoformat(),
            "ticket_count": record.ticket_count,
        }
    finally:
        session.close()
