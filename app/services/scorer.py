"""
Heuristic worklog/resolution-note quality rubric, 0-100. Always runs (free,
instant, explainable) regardless of whether an LLM is configured.

LLM-blended scoring (layering a model's judgment on top of this heuristic
for nuance) happens in graph_pipeline.py's score_node, not here - keeping
the async/LLM/retry concerns in one place (the LangGraph pipeline) rather
than split across this module and the graph.

Rubric, 25 points each:
  - completeness   : has enough substance (length, not a placeholder)
  - root_cause     : mentions cause/diagnosis language
  - resolution     : mentions concrete resolution/action language
  - professionalism: no ALL CAPS shouting, no placeholder text, has
                     sentence structure
"""
import re

from app.models.schemas import WorklogScore

_ROOT_CAUSE_HINTS = re.compile(
    r"\b(root cause|caused by|due to|because|diagnos(is|ed)|found that|identified)\b", re.IGNORECASE
)
_RESOLUTION_HINTS = re.compile(
    r"\b(resolved|fixed|restarted|reconfigur|patched|replaced|rebooted|restored|implemented|applied|escalated)\b",
    re.IGNORECASE,
)
_PLACEHOLDER_HINTS = re.compile(r"^\s*(n/?a|none|test|tbd|todo|\.|-)\s*$", re.IGNORECASE)
_TIMESTAMP_HINTS = re.compile(r"\b\d{1,2}[:/]\d{2}\b|\b\d{4}-\d{2}-\d{2}\b")


def _completeness_score(worklog: str) -> tuple[float, list[str]]:
    flags = []
    length = len(worklog.split())
    if _PLACEHOLDER_HINTS.match(worklog):
        flags.append("Placeholder/empty worklog content.")
        return 0.0, flags
    if length < 5:
        flags.append("Worklog is very short - lacks detail.")
        return 5.0, flags
    if length < 15:
        return 15.0, flags
    return 25.0, flags


def _root_cause_score(worklog: str) -> tuple[float, list[str]]:
    if _ROOT_CAUSE_HINTS.search(worklog):
        return 25.0, []
    return 5.0, ["Root cause not clearly documented."]


def _resolution_score(worklog: str) -> tuple[float, list[str]]:
    if _RESOLUTION_HINTS.search(worklog):
        return 25.0, []
    return 5.0, ["Resolution/action steps not clearly documented."]


def _professionalism_score(worklog: str) -> tuple[float, list[str]]:
    flags = []
    score = 25.0
    letters = [c for c in worklog if c.isalpha()]
    if letters and sum(1 for c in letters if c.isupper()) / len(letters) > 0.6:
        score -= 10
        flags.append("Excessive capitalization detected.")
    if len(worklog) > 0 and not _TIMESTAMP_HINTS.search(worklog) and len(worklog.split()) > 20:
        score -= 5
        flags.append("No timestamps/action trail found in a long worklog.")
    return max(score, 0.0), flags


def heuristic_score(worklog: str) -> WorklogScore:
    worklog = (worklog or "").strip()
    if not worklog:
        return WorklogScore(
            ticket_id="", score=0, breakdown={"completeness": 0, "root_cause": 0, "resolution": 0, "professionalism": 0},
            flags=["No worklog provided."],
        )

    completeness, f1 = _completeness_score(worklog)
    root_cause, f2 = _root_cause_score(worklog)
    resolution, f3 = _resolution_score(worklog)
    professionalism, f4 = _professionalism_score(worklog)

    total = round(completeness + root_cause + resolution + professionalism)
    return WorklogScore(
        ticket_id="",
        score=total,
        breakdown={
            "completeness": completeness,
            "root_cause": root_cause,
            "resolution": resolution,
            "professionalism": professionalism,
        },
        flags=f1 + f2 + f3 + f4,
    )
