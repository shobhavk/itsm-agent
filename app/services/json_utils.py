"""Extracts a JSON object from an LLM response that may be wrapped in
markdown code fences or have stray prose around it. Not every model honors
"respond with JSON only" as strictly as GPT does."""
import re

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)
_BRACES_RE = re.compile(r"\{.*\}", re.DOTALL)


def extract_json(text: str) -> str:
    """Returns the best-guess JSON substring from a raw LLM response."""
    if not text:
        return "{}"
    text = text.strip()

    fence_match = _FENCE_RE.search(text)
    if fence_match:
        return fence_match.group(1).strip()

    brace_match = _BRACES_RE.search(text)
    if brace_match:
        return brace_match.group(0).strip()

    return text
