"""
Cheap categorization pre-pass, cheapest/most-reliable method first:

  1. Keyword rules      - fast, free, deterministic, high precision for
                           obvious cases ("password reset" -> Security).
  2. Embedding similarity - semantic match against category description
                           vectors. Batched via LangChain's
                           aembed_documents (chunk_size=EMBEDDING_BATCH_SIZE)
                           rather than one call per ticket - the fix for
                           429s on large uploads.

Whatever's left unresolved after this pre-pass goes to the LangGraph
pipeline (see graph_pipeline.py) for LLM classification - kept separate
because per-ticket LLM calls benefit from LangGraph's per-item state-machine
model and bounded-concurrency `.abatch()`, whereas keyword/embedding
resolution is cheaper done as a flat batch pass upfront.
"""
import logging

import numpy as np

from app.config import get_settings
from app.models.schemas import CATEGORIES, CategoryResult
from app.services.llm_client import embeddings_retry

logger = logging.getLogger(__name__)
settings = get_settings()

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

_CATEGORY_DESCRIPTIONS = {
    cat: f"ITSM incident category: {cat}. Keywords: {', '.join(KEYWORD_RULES.get(cat, [cat]))}"
    for cat in CATEGORIES if cat != "Uncategorized"
}


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
    """Handles the keyword + embedding pre-pass only. LLM classification for
    anything left unresolved happens in graph_pipeline.py."""

    def __init__(self, embeddings_model, similarity_threshold: float = 0.35):
        self._embeddings = embeddings_model
        self._threshold = similarity_threshold
        self._category_vectors: dict[str, np.ndarray] | None = None

    async def _ensure_category_vectors(self):
        if self._category_vectors is not None:
            return
        if self._embeddings is None:
            self._category_vectors = {}
            return
        try:
            names = list(_CATEGORY_DESCRIPTIONS.keys())
            embed_fn = embeddings_retry(self._embeddings.aembed_documents)
            vecs = await embed_fn(list(_CATEGORY_DESCRIPTIONS.values()))
            self._category_vectors = {n: np.array(v) for n, v in zip(names, vecs)}
        except Exception as exc:
            logger.info("Embedding backend unavailable, skipping semantic matching: %s", exc)
            self._category_vectors = {}

    async def _batch_embedding_match(
        self, items: list[tuple[str, str]]
    ) -> tuple[dict[str, CategoryResult], list[tuple[str, str]]]:
        """Embeds many (ticket_id, text) pairs via one (chunked) call rather
        than one call per ticket - the key fix for 429s on large uploads.
        LangChain's aembed_documents handles the chunking internally at
        chunk_size=EMBEDDING_BATCH_SIZE. Returns (matched, still_unmatched)."""
        await self._ensure_category_vectors()
        matched: dict[str, CategoryResult] = {}
        if not self._category_vectors or not items:
            return matched, items

        texts = [text for _, text in items]
        try:
            embed_fn = embeddings_retry(self._embeddings.aembed_documents)
            vectors = await embed_fn(texts)
        except Exception as exc:
            logger.info("Batch embedding call failed for %d tickets, leaving for LLM fallback: %s", len(items), exc)
            return matched, items

        still_unmatched: list[tuple[str, str]] = []
        for (ticket_id, text), vec in zip(items, vectors):
            vec = np.array(vec)
            scored = {cat: _cosine_sim(vec, cvec) for cat, cvec in self._category_vectors.items()}
            best_cat = max(scored, key=scored.get)
            best_score = scored[best_cat]
            if best_score >= self._threshold:
                matched[ticket_id] = CategoryResult(
                    ticket_id=ticket_id, category=best_cat, confidence=min(best_score, 0.99), method="embedding"
                )
            else:
                still_unmatched.append((ticket_id, text))

        return matched, still_unmatched

    async def pre_resolve(self, items: list[tuple[str, str]]) -> tuple[dict[str, CategoryResult], list[tuple[str, str]]]:
        """Resolves as many tickets as possible via keyword rules then
        batched embedding similarity. Returns (resolved, still_unresolved) -
        the caller (pipeline.py) sends still_unresolved through the
        LangGraph LLM-classification node."""
        resolved: dict[str, CategoryResult] = {}
        needs_embedding: list[tuple[str, str]] = []

        for ticket_id, text in items:
            text = (text or "").strip()
            if not text:
                resolved[ticket_id] = CategoryResult(ticket_id=ticket_id, category="Uncategorized", confidence=0.0, method="fallback")
                continue
            if match := _keyword_match(text):
                resolved[ticket_id] = CategoryResult(ticket_id=ticket_id, category=match[0], confidence=match[1], method="keyword_rule")
                continue
            needs_embedding.append((ticket_id, text))

        if self._embeddings is not None and needs_embedding:
            embedding_matched, still_unmatched = await self._batch_embedding_match(needs_embedding)
            resolved.update(embedding_matched)
        else:
            still_unmatched = needs_embedding

        return resolved, still_unmatched
