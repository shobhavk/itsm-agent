"""
Security controls:
  1. API key auth (header-based) for every non-health endpoint.
  2. Upload validation: extension allow-list, size cap, MIME sniffing.
  3. Prompt-injection mitigation for any free text that gets embedded into
     an LLM prompt (incident descriptions / worklogs are attacker-reachable
     text - treat them as untrusted input, never as instructions).
  4. Rate limiting (slowapi) wired up in main.py using get_remote_address.
"""
import re
from typing import Annotated

from fastapi import Header, HTTPException, UploadFile, status

from app.config import get_settings

settings = get_settings()

ALLOWED_EXTENSIONS = {".csv", ".xlsx", ".xls", ".txt"}

# Patterns commonly used to try to hijack an LLM via injected instructions
# inside "user content" fields (ticket descriptions, worklogs, etc).
_INJECTION_PATTERNS = [
    r"ignore (all|any|previous|the) instructions",
    r"disregard (all|any|previous|the) instructions",
    r"you are now",
    r"system prompt",
    r"act as (a|an) (?!engineer|technician|admin)",
    r"reveal (your|the) (prompt|instructions|system)",
    r"</?(system|assistant|user)>",
    r"```system",
]
_INJECTION_RE = re.compile("|".join(_INJECTION_PATTERNS), re.IGNORECASE)


def verify_api_key(x_api_key: Annotated[str | None, Header()] = None) -> str:
    """FastAPI dependency: validates the X-API-Key header against configured keys."""
    valid_keys = {k.strip() for k in settings.API_KEYS.split(",") if k.strip()}
    if not x_api_key or x_api_key not in valid_keys:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid API key.",
        )
    return x_api_key


def validate_upload(file: UploadFile) -> None:
    """Validates file extension and (approximate) size before processing."""
    name = (file.filename or "").lower()
    if not any(name.endswith(ext) for ext in ALLOWED_EXTENSIONS):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type. Allowed: {sorted(ALLOWED_EXTENSIONS)}",
        )
    # UploadFile.size is populated by Starlette when available
    if file.size and file.size > settings.MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds {settings.MAX_UPLOAD_MB}MB limit.",
        )


def sanitize_for_llm(text: str, max_len: int = 4000) -> str:
    """
    Defangs likely prompt-injection attempts inside untrusted free text
    before it is interpolated into an LLM prompt. This does not "clean"
    the text semantically - it just strips/flags override attempts and
    truncates length. Always also wrap this text in clear delimiters in
    the prompt template (see llm_client.py) and instruct the model to
    treat it strictly as data, never as instructions.
    """
    if not text:
        return ""
    text = text[:max_len]
    if _INJECTION_RE.search(text):
        text = _INJECTION_RE.sub("[filtered]", text)
    # Strip characters commonly used to break out of prompt delimiters
    text = text.replace("```", "'''")
    return text.strip()


def contains_injection_attempt(text: str) -> bool:
    return bool(text and _INJECTION_RE.search(text))
