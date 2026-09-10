"""
Resolves a best-effort "host / server / CI" label per ticket, used for the
"servers/hosts generating the most alerts" dashboard aggregate.

Priority order:
  1. A structured Configuration Item value from the source data (most
     reliable - this is what CMDB-linked ITSM tools populate).
  2. A regex-based scan of the description + worklog free text for
     hostname/server-like tokens, when no structured CI was given.

This is a heuristic for grouping tickets on a "top offenders" chart, not
a CMDB lookup - it will occasionally miss or misfire on unusual naming
conventions, and that's an acceptable tradeoff for an aggregate view.
"""
import re

# Checked in order, first match in the text wins. Ordered from most to
# least specific to reduce false positives (e.g. try FQDN before a bare
# "WORD-1234" pattern that could match all sorts of non-host tokens).
_HOST_PATTERNS: list[re.Pattern] = [
    # FQDN, e.g. db-prod01.corp.example.com
    re.compile(
        r"\b((?:[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.){2,}"
        r"[a-zA-Z]{2,})\b"
    ),
    # IPv4 address
    re.compile(r"\b(\d{1,3}(?:\.\d{1,3}){3})\b"),
    # Common server-naming conventions: SRV-DB01, WEBPRD02, APP01-PROD,
    # db-prod-1, HOSTNAME01
    re.compile(
        r"\b([A-Za-z]{2,12}[-_]?"
        r"(?:PRD|PROD|DEV|UAT|QA|STG|STAGE|TST|TEST)?[-_]?\d{1,4})\b",
        re.IGNORECASE,
    ),
]

# Tokens the patterns above can accidentally pick up that are never hosts.
_STOPWORDS = {"http", "https", "www", "e.g", "i.e"}

# Require at least one digit for the loose server-naming-convention
# pattern, otherwise ordinary words ("STAGE", "TEST") match as false
# positives far too often.
_HAS_DIGIT = re.compile(r"\d")


def extract_host(configuration_item: str | None, description: str, worklog: str) -> str | None:
    """Returns the best available host/server label for a ticket, or None
    if nothing could be identified."""
    if configuration_item and configuration_item.strip():
        return configuration_item.strip()

    text = f"{description or ''} {worklog or ''}"

    for i, pattern in enumerate(_HOST_PATTERNS):
        for match in pattern.finditer(text):
            candidate = match.group(1)
            if candidate.lower() in _STOPWORDS:
                continue
            if len(candidate) < 4:
                continue
            # The loose naming-convention pattern (last in the list) is
            # prone to false positives on plain words - require a digit.
            if i == len(_HOST_PATTERNS) - 1 and not _HAS_DIGIT.search(candidate):
                continue
            return candidate

    return None
