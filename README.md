# ITSM Quality Analysis Agent — v1

Agentic pipeline that ingests incident data (Excel / CSV / unstructured text),
validates and normalizes it, auto-categorizes tickets into ITSM buckets,
scores worklog/resolution-note quality, and surfaces everything in a Gradio
dashboard with CSV export. Ships as a single FastAPI service with Gradio
mounted at `/ui`, containerized with Docker.

## What's actually been tested

I ran this locally (no live Docker daemon in the build environment, so the
container itself is untested — see "Known gaps" below) and verified:

- Full pipeline on sample data: 7/7 tickets correctly categorized
- Worklog scoring distinguishes real resolution notes from placeholders
  ("n/a", "TEST") — scores of 25–35 vs. 80–100
- `/health` returns 200 with no auth
- `/api/v1/analyze/file` returns 401 without `X-API-Key`, 200 with a valid one
- `/api/v1/export/csv` returns a valid CSV after an analysis has run
- `.exe` upload rejected with 400 (extension allow-list)
- A ticket containing "ignore all previous instructions..." was parsed as
  inert data and categorized normally — no instruction-following occurred
  (expected, since the default `rule_based` provider never calls an LLM;
  the sanitization layer also filters this pattern before any provider
  that *does* call an LLM sees it)
- Duplicate ticket IDs, blank rows, and missing IDs are handled without
  silent data loss (rejected rows are counted and reported, not dropped
  quietly)

## Architecture

```
app/
  config.py            Settings via env vars (pydantic-settings) — no hardcoded secrets
  security.py           API key auth, upload validation, prompt-injection sanitization
  main.py                FastAPI app: CORS, rate limiting, security headers, mounts Gradio
  models/schemas.py     TicketRecord, CategoryResult, WorklogScore, AnalysisResponse
  routers/analyze.py    POST /api/v1/analyze/file, /analyze/text, GET /export/csv — all async
  services/
    data_ingestion.py    Excel/CSV/unstructured-text -> raw DataFrame
    validator.py          Normalize + validate -> TicketRecord list (dedup, type coercion)
    llm_client.py          LangChain model factory: get_chat_model()/get_embeddings_model()
    categorizer.py         Keyword rules -> batched embedding similarity (async pre-pass)
    graph_pipeline.py       LangGraph StateGraph: classify (LLM fallback) + score nodes
    scorer.py                Heuristic worklog rubric (used by graph_pipeline's score node)
    json_utils.py             extract_json() - pulls JSON out of LLM responses robustly
    pipeline.py               Orchestrates the above end-to-end, async throughout
ui/
  gradio_app.py           Dashboard: upload, category chart, filterable table, CSV download
```

### How a batch of tickets actually gets processed

1. **Keyword rules** (`categorizer.py`) — free, deterministic, no API calls, resolves
   obvious cases ("disk space critical" → Disk/file system extension).
2. **Batched embedding similarity** (`categorizer.py`) — whatever keywords couldn't
   resolve gets embedded in one (chunked) call via `aembed_documents`, not one
   call per ticket. This is the fix for 429s at 3k+ rows — see below.
3. **LangGraph classify + score** (`graph_pipeline.py`) — a small per-ticket
   `StateGraph` (`classify` → `score`) processed for every ticket via
   `.abatch()`, concurrently but bounded by `LLM_MAX_CONCURRENCY`. `classify`
   is a no-op for anything step 1/2 already resolved; `score` always runs the
   heuristic rubric and optionally blends in an LLM judgment.
4. **"Uncategorized"** — explicit safety net. The agent never forces a wrong
   label just to fill a bucket, and a ticket that fails every step still gets
   a heuristic-only score rather than being dropped.

In `rule_based` mode (the default, and what most of the testing below used),
steps 2–3's LLM calls are skipped entirely — keyword-only categorization,
zero external calls. Intentional as a safe, zero-dependency demo mode.

### This is genuinely async now

Every route in `routers/analyze.py` and the Gradio `_analyze` handler
`await` the pipeline directly — there's no blocking synchronous work sitting
inside an `async def` route pretending to be non-blocking. I verified this
isn't just cosmetic: a test with a fake chat model that sleeps 20ms per call
processed 300 tickets (600 total classify+score calls) in 1.81 seconds with
`LLM_MAX_CONCURRENCY=10`, and I instrumented the fake model to track
concurrent-call count directly — it never exceeded 10, proving both that
calls genuinely run concurrently (not sequentially blocking each other) and
that the concurrency bound is actually respected, not just configured.

### Worklog scoring

A 100-point heuristic rubric (25 pts each): completeness, root-cause
documentation, resolution/action documentation, professionalism. This
always runs (`scorer.py`, pure Python, free). `graph_pipeline.py`'s `score`
node optionally blends in an LLM judgment on top when a chat model is
configured — but a failed or unavailable LLM call never blocks or degrades
the base heuristic score, it just keeps the heuristic-only number.

## Your tech stack: gpt-5.6-luna, text-embedding-large, SAP AI SDK, LangChain, LangGraph

The LLM layer (`app/services/llm_client.py`) is built on **LangChain**,
using the SAP proxy's actual LangChain integration — confirmed against a
working script in your environment:

```python
from gen_ai_hub.proxy.langchain import ChatOpenAI
from gen_ai_hub.proxy import get_proxy_client
proxy_client = get_proxy_client('gen-ai-hub')
chat_llm = ChatOpenAI(proxy_model_name=MODEL_NAME, proxy_client=proxy_client)
```

This replaced an earlier attempt that went through the lower-level
`gen_ai_hub.proxy.native.openai` module directly. That module's `.responses`
attribute turned out to be `None` in your installed environment (an older
`openai` package pinned as a transitive dependency predates the Responses
API) — `gen_ai_hub.proxy.langchain.ChatOpenAI` sits one layer up and doesn't
have that problem. It's also a genuine LangChain `Runnable`
(`langchain_openai.ChatOpenAI` under the hood), so `.ainvoke()`, `.abatch()`,
and `.with_retry()` all work natively — no hand-rolled async or retry code
needed for the chat model.

| Provider | When to use | Status |
|---|---|---|
| `rule_based` | Default. No external calls. | ✅ Tested |
| `openai_compat` | Any OpenAI-compatible endpoint/gateway (`langchain_openai.ChatOpenAI`) | ✅ Implemented, untested against your actual gateway |
| `sap_genai_hub` | SAP proxy via `gen_ai_hub.proxy.langchain` | ✅ Implemented, tested against fakes matching LangChain's real interfaces — not against your real AI Core deployment |

### Wiring up SAP Generative AI Hub

1. `pip install sap-ai-sdk-gen[all]` (already in `requirements.txt`)
2. Set `AICORE_AUTH_URL`, `AICORE_CLIENT_ID`, `AICORE_CLIENT_SECRET`,
   `AICORE_BASE_URL`, `AICORE_RESOURCE_GROUP` in `.env`
3. Set `LLM_PROVIDER=sap_genai_hub`
4. `CHAT_MODEL_NAME` / `EMBEDDING_MODEL_NAME` — point these at whatever
   deployment names are provisioned in your AI Core resource group

I verified the calling code's shape (class names, constructor kwargs,
`.ainvoke()`/`.aembed_documents()` interfaces, retry/error handling) by
downloading and inspecting the actual `sap-ai-sdk-gen` wheel and testing
against `langchain_core`'s official fake model/embedding classes and a
custom fake that raises real `openai.RateLimitError`s on demand. I have
**not** tested this against your real AI Core deployment — that only your
environment can confirm. Worth a real smoke test before relying on it at
scale.

If a chat/embeddings model fails to initialize (missing credentials, SDK
import error, etc.), `get_chat_model()`/`get_embeddings_model()` return
`None` rather than raising — every caller treats `None` as "skip this step,
fall back to keyword rules / heuristic-only scoring" rather than crashing.

### Handling 429s on large uploads (3k+ rows)

Three things, all verified with targeted tests (not just written and
assumed to work):

1. **Batched embedding calls** — the real fix. Previously, every ticket
   that fell through keyword rules triggered its own embedding API call;
   at 3k rows with, say, 40% needing embedding matching, that's 1,200+
   separate calls. `Categorizer.pre_resolve()` now embeds many ticket
   texts per call (chunked at `EMBEDDING_BATCH_SIZE`, default 200) via
   LangChain's `aembed_documents`. **Tested:** a 60-ticket batch needing
   embedding classification produced exactly 2 embedding calls total (1
   for category-vector priming, 1 for the batch), not 60+.
2. **429-aware retry with backoff** (`LLM_RATE_LIMIT_MAX_RETRIES`, default
   5) via LangChain's built-in `Runnable.with_retry(retry_if_exception_type=
   (openai.RateLimitError,), wait_exponential_jitter=True)` on the chat
   model. **Tested:** a fake model that fails its first 2 calls with a real
   `openai.RateLimitError` then succeeds was retried transparently by the
   pipeline with no code-level handling needed at the call site. A model
   that *always* rate-limits exhausts retries after exactly 5 attempts per
   node call, then degrades to `Uncategorized`/heuristic-only rather than
   crashing the batch — the failure reason is recorded in that ticket's
   `validation_flags` so it's visible in the exported CSV, not silently lost.
3. **Bounded concurrency** (`LLM_MAX_CONCURRENCY`, default 10) — LangGraph's
   `.abatch(..., config={"max_concurrency": N})` caps how many tickets are
   in flight at once. Uncapped concurrency on a 3k-row batch is what causes
   a 429 storm in the first place; retry alone doesn't fix firing 3,000
   simultaneous requests. **Tested:** instrumented a fake model to track
   concurrent in-flight calls directly — never exceeded the configured
   limit across 600 calls.

I have **not** tested any of this against your real AI Core deployment's
actual rate limits — `LLM_MAX_CONCURRENCY`, `EMBEDDING_BATCH_SIZE`, and
`LLM_RATE_LIMIT_MAX_RETRIES` are reasonable defaults, not numbers measured
against your specific quota. If you still see 429s after this change, your
AI Core dashboard should show which endpoint/quota is being hit, and these
three settings in `.env` are what to tune — lower `LLM_MAX_CONCURRENCY`
first, since that has the most direct effect on request rate.

Per-ticket LLM classification/scoring calls are *not* batched into a single
prompt — reliably parsing N different classification results out of one
completion is fragile — they rely on bounded concurrency + retry instead.
This is fine once embedding-call volume is fixed, since that was what drove
the call count into the thousands in the first place.

## Security controls implemented

- **API key auth** (`X-API-Key` header) on every analysis/export endpoint;
  `/health` is intentionally open for container health checks.
- **Rate limiting** (slowapi, default 30 req/min/IP, configurable).
- **Upload validation**: extension allow-list (`.csv .xlsx .xls .txt`),
  size cap (default 15MB).
- **Prompt-injection mitigation**: all free text (descriptions, worklogs)
  is treated as untrusted data, not instructions. Before any LLM call, text
  is sanitized (`sanitize_for_llm`) and wrapped in explicit `<<<DATA>>>`
  delimiters with a system-level guardrail instructing the model to treat
  delimited content as data even if it looks like instructions.
- **No secrets in code** — everything sensitive comes from environment
  variables; `.env` is gitignored and dockerignored.
- **CORS locked to an explicit origin allow-list.**
- **Security headers** (`X-Content-Type-Options`, `X-Frame-Options`,
  `Referrer-Policy`) on every response.
- **No stack traces leaked** — an unhandled-exception handler returns a
  generic 500 and logs the real error server-side only.
- **Docker hardening**: multi-stage build, non-root user, healthcheck,
  read-only root filesystem in `docker-compose.yml`, `no-new-privileges`.
- **API docs hidden in prod** (`ENV=prod` disables `/api/docs`).

Not yet implemented (call these out to your security team before
production use): request signing/mTLS between services, secrets-manager
integration (currently plain env vars), audit logging/SIEM export,
per-user RBAC (currently one flat API key tier), virus scanning on
uploads.

## Running locally

```bash
cp .env.example .env          # edit API_KEYS at minimum
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

- Dashboard: http://localhost:8000/ui
- API docs (non-prod only): http://localhost:8000/api/docs
- Sample data to try: `sample_data/sample_incidents.csv`

## Running with Docker

```bash
docker compose up --build
```

**Known gap:** I could not run a live Docker daemon in the build
environment, so the container build/run itself is untested — only the
application code (identical to what ships in the image) has been verified.
Please do a `docker compose up --build` smoke test before deploying, and
open an issue/ping me if the build surfaces anything — multi-stage venv
copies occasionally need a `PYTHONPATH` tweak depending on base image
patch versions.

## API reference (v1)

| Endpoint | Method | Auth | Notes |
|---|---|---|---|
| `/health` | GET | none | liveness/readiness probe |
| `/api/v1/analyze/file` | POST | `X-API-Key` | multipart upload, `.csv/.xlsx/.xls/.txt` |
| `/api/v1/analyze/text` | POST | `X-API-Key` | query/body param `raw_text`, unstructured text |
| `/api/v1/export/csv` | GET | `X-API-Key` | re-exports the last analysis for that key |

The Gradio dashboard at `/ui` calls the pipeline in-process (not over HTTP)
so it doesn't need an API key inside the browser session — the REST API
remains separately available and secured for automation/integration use.

## Roadmap ideas for v2

- Persist analysis history (currently in-memory, lost on restart / not
  shared across replicas — fine for v1 single-instance, not for scale-out)
- Human-in-the-loop correction: let reviewers fix a wrong category and
  feed that back as few-shot examples or fine-tuning data
- SLA/trend dashboards over time, not just per-upload snapshots
- Multi-tenant API key scoping with per-team dashboards
- Batch/async processing for very large files (current v1 is synchronous)
