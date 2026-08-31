"""
Hybrid categorization pipeline, cheapest/most-reliable method first:

  1. Keyword rules      - fast, free, deterministic, high precision for
                           obvious cases ("password reset" -> Security).
  2. Embedding similarity - semantic match against category description
                           vectors using EMBEDDING_MODEL_NAME. Catches
                           paraphrased/unstructured text keyword rules miss.
  3. LLM classification  - only invoked if the above are inconclusive
                           (keeps cost/latency down, and confines LLM
                           input to sanitized text with a guardrail prompt).
  4. "Uncategorized"     - explicit safety net; never force a wrong label.
"""
import logging

import numpy as np

from app.models.schemas import CATEGORIES, CategoryResult
from app.security import sanitize_for_llm
from app.services.llm_client import LLMClient

logger = logging.getLogger(__name__)

# Keyword rules: category -> indicative keywords/phrases (lowercase)
KEYWORD_RULES: dict[str, list[str]] = {
    "Hardware issues": ["hardware fail", "disk failure", "physical disk", "nic fail", "power supply", "motherboard", "faulty hardware"],
    "Software issues": ["software bug", "application error", "app crash", "patch fail", "install fail", "software issue"],
    "Network issues": ["network down", "connectivity issue", "packet loss", "vpn", "firewall", "dns issue", "network latency", "network connectivity"],
    "Database issues": ["database", "db down", "sql error", "table lock", "tablespace", "deadlock", "oracle", "mongodb"],
    "Security incidents": ["security incident", "unauthorized access", "malware", "phishing", "breach", "vulnerability",
                            "ransomware", "suspicious login", "failed login attempt", "compromised account"],
    "Server deployment": ["server deployment", "server provisioning", "provisioning", "deploy new server", "server build",
                           "new vm", "new server", "spin up"],
    "Server configuration change": ["config change", "configuration change", "change request", "cr#", "reconfigure server"],
    "Performance issue": ["performance degradation", "slow response", "high cpu", "high memory", "performance issue",
                           "latency issue", "throughput", "slow performance"],
    "Disk/file system extension": ["disk extension", "extend disk", "increase disk space", "extend filesystem",
                                     "lun extension", "disk space critical", "partition full", "disk space", "lvm"],
    "Backup related": ["backup fail", "backup job", "restore fail", "veeam", "netbackup", "backup schedule", "backup failure"],
    "CCIR related": ["ccir"],
    "File system cleanup": ["file system cleanup", "disk cleanup", "purge logs", "free up space", "clear temp", "cleanup disk"],
    "Virtualisation/cloud platform issues": ["vmware", "hypervisor", "vcenter", "aws", "azure", "gcp", "cloud platform", "virtual machine"],
    "OS upgrade/service pack upgrade": ["os upgrade", "service pack", "patch upgrade", "kernel upgrade", "os patching", "windows patch"],
    "Server migration": ["server migration", "migrate server", "p2v", "v2v", "lift and shift", "migration to"],
}

_CATEGORY_DESCRIPTIONS = {cat: f"ITSM incident category: {cat}. Keywords: {', '.join(KEYWORD_RULES.get(cat, [cat]))}"
                           for cat in CATEGORIES if cat != "Uncategorized"}


def _keyword_match(text: str) -> tuple[str, float] | None:
    lower = text.lower()
    best_cat, best_hits = None, 0
    for cat, kws in KEYWORD_RULES.items():
        hits = sum(1 for kw in kws if kw in lower)
        if hits > best_hits:
            best_cat, best_hits = cat, hits
    if best_cat and best_hits > 0:
        confidence = min(0.6 + 0.15 * best_hits, 0.95)
        return best_cat, confidence
    return None


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1e-9
    return float(np.dot(a, b) / denom)


class Categorizer:
    def __init__(self, llm_client: LLMClient, similarity_threshold: float = 0.35):
        self._llm = llm_client
        self._threshold = similarity_threshold
        self._category_vectors: dict[str, np.ndarray] | None = None

    def _ensure_category_vectors(self):
        if self._category_vectors is not None:
            return
        try:
            names = list(_CATEGORY_DESCRIPTIONS.keys())
            vecs = self._llm.embed(list(_CATEGORY_DESCRIPTIONS.values()))
            self._category_vectors = {n: np.array(v) for n, v in zip(names, vecs)}
        except Exception as exc:
            logger.info("Embedding backend unavailable, skipping semantic matching: %s", exc)
            self._category_vectors = {}

    def _embedding_match(self, text: str) -> tuple[str, float] | None:
        self._ensure_category_vectors()
        if not self._category_vectors:
            return None
        try:
            [vec] = self._llm.embed([text])
        except Exception as exc:
            logger.info("Embedding call failed for ticket text: %s", exc)
            return None
        vec = np.array(vec)
        scored = {cat: _cosine_sim(vec, cvec) for cat, cvec in self._category_vectors.items()}
        best_cat = max(scored, key=scored.get)
        best_score = scored[best_cat]
        if best_score >= self._threshold:
            return best_cat, min(best_score, 0.99)
        return None

    def _llm_match(self, text: str) -> tuple[str, float] | None:
        try:
            safe_text = sanitize_for_llm(text)
            result = self._llm.classify(safe_text, [c for c in CATEGORIES if c != "Uncategorized"])
            category = result.get("category")
            if category in CATEGORIES:
                return category, float(result.get("confidence", 0.5))
        except NotImplementedError:
            pass
        except Exception as exc:
            logger.info("LLM classification failed, falling back: %s", exc)
        return None

    def categorize(self, ticket_id: str, text: str) -> CategoryResult:
        text = text.strip()
        if not text:
            return CategoryResult(ticket_id=ticket_id, category="Uncategorized", confidence=0.0, method="fallback")

        if match := _keyword_match(text):
            return CategoryResult(ticket_id=ticket_id, category=match[0], confidence=match[1], method="keyword_rule")

        if match := self._embedding_match(text):
            return CategoryResult(ticket_id=ticket_id, category=match[0], confidence=match[1], method="embedding")

        if match := self._llm_match(text):
            return CategoryResult(ticket_id=ticket_id, category=match[0], confidence=match[1], method="llm")

        return CategoryResult(ticket_id=ticket_id, category="Uncategorized", confidence=0.0, method="fallback")
