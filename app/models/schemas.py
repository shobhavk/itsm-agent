from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field, field_validator

CATEGORIES: list[str] = [
    "Hardware issues",
    "Software issues",
    "Network issues",
    "Database issues",
    "Security incidents",
    "Server deployment",
    "Server configuration change",
    "Performance issue",
    "Disk/file system extension",
    "Backup related",
    "CCIR related",
    "File system cleanup",
    "Virtualisation/cloud platform issues",
    "OS upgrade/service pack upgrade",
    "Server migration",
    "Uncategorized",  # safety-net bucket, never force a bad match
]


class TicketRecord(BaseModel):
    """Normalized representation of one incident, regardless of source format."""

    ticket_id: str
    short_description: str = ""
    description: str = ""
    worklog: str = ""
    priority: Optional[str] = None
    status: Optional[str] = None
    assignment_group: Optional[str] = None
    opened_at: Optional[datetime] = None
    closed_at: Optional[datetime] = None
    source_row: int = Field(description="Original row index for traceability")

    @field_validator("ticket_id", mode="before")
    @classmethod
    def _stringify_id(cls, v):
        return str(v).strip() if v is not None else ""


class CategoryResult(BaseModel):
    ticket_id: str
    category: str
    confidence: float = Field(ge=0.0, le=1.0)
    method: str  # "keyword_rule" | "embedding" | "llm" | "fallback"


class WorklogScore(BaseModel):
    ticket_id: str
    score: int = Field(ge=0, le=100)
    breakdown: dict[str, float]
    flags: list[str] = []


class AnalyzedTicket(BaseModel):
    ticket_id: str
    short_description: str
    description: str
    worklog: str
    priority: Optional[str]
    status: Optional[str]
    assignment_group: Optional[str]
    category: str
    category_confidence: float
    category_method: str
    worklog_score: int
    worklog_flags: list[str]
    validation_flags: list[str] = []


class AnalysisResponse(BaseModel):
    total_records: int
    valid_records: int
    rejected_records: int
    category_counts: dict[str, int]
    average_worklog_score: float
    results: list[AnalyzedTicket]
