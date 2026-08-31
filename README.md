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
  routers/analyze.py    POST /api/v1/analyze/file, /analyze/text, GET /export/csv
  services/
    data_ingestion.py    Excel/CSV/unstructured-text -> raw DataFrame
    validator.py          Normalize + validate -> TicketRecord list (dedup, type coercion)
    llm_client.py          Pluggable LLM/embedding interface (see below)
    categorizer.py         Keyword rules -> embedding similarity -> LLM -> "Uncategorized"
    scorer.py               Heuristic worklog rubric, optional LLM blending
    pipeline.py              Orchestrates the above end-to-end
ui/
  gradio_app.py           Dashboard: upload, category chart, filterable table, CSV download
```

### Categorization strategy (cheapest/most-reliable first)

1. **Keyword rules** — fast, free, deterministic, high precision for obvious
   phrasing ("disk space critical" → Disk/file system extension).
2. **Embedding similarity** — semantic match against category descriptions
   using `EMBEDDING_MODEL_NAME`, for paraphrased text the keywords miss.
3. **LLM classification** — only called if 1 and 2 are inconclusive, to keep
   cost and latency down.
4. **"Uncategorized"** — explicit safety net. The agent never forces a
   wrong label just to fill a bucket.

In `rule_based` mode (the default, and what I tested above), steps 2–3 are
skipped entirely — you get keyword-only categorization with zero external
calls. This is intentional as a safe, zero-dependency demo mode.

### Worklog scoring

A 100-point heuristic rubric (25 pts each): completeness, root-cause
documentation, resolution/action documentation, professionalism (no
ALL-CAPS shouting, has a timestamp/action trail for longer entries).
This always runs. If an LLM provider is configured, its score is blended
in (averaged) for nuance — but a failed or unavailable LLM call never
blocks or degrades the base heuristic score.

## Your tech stack: gpt-5.6luna, text-embedding-large, SAP Gen AI SDK

I don't have visibility into `gpt-5.6luna` or a `text-embedding-large`
model as public/recognized model identifiers, and I can't get live SAP AI
Core credentials in this environment — so I couldn't test real calls
against either. Rather than guess at endpoint shapes, I built the LLM
layer as a **pluggable interface** (`app/services/llm_client.py`) with
three backends selected by `LLM_PROVIDER` in `.env`:

| Provider | When to use | Status |
|---|---|---|
| `rule_based` | Default. No external calls. What I tested. | ✅ Working |
| `openai_compat` | Any OpenAI-compatible endpoint/gateway | ✅ Implemented, untested against your actual gateway |
| `sap_genai_hub` | SAP Generative AI Hub SDK / AI Core | ✅ Implemented, tested against a mock of the SDK's call signature — not against your real AI Core deployment |

### Wiring up SAP Generative AI Hub

The `sap_genai_hub` provider is now **implemented for real**, matching the
exact calling convention from your existing codebase
(`gen_ai_hub.proxy.native.openai`, `chat.completions.create(model_name=...,
messages=...)` — note `model_name=`, not `model=`, which is what
`OpenAICompatClient` uses for the plain openai-python client).

1. `pip install generative-ai-hub-sdk` (already in `requirements.txt`)
2. Set `AICORE_AUTH_URL`, `AICORE_CLIENT_ID`, `AICORE_CLIENT_SECRET`,
   `AICORE_BASE_URL`, `AICORE_RESOURCE_GROUP` in `.env` — the SDK reads
   these directly, there's no separate client object to construct
3. Set `LLM_PROVIDER=sap_genai_hub`
4. `CHAT_MODEL_NAME` defaults to `gpt-5.6-luna`, `EMBEDDING_MODEL_NAME` to
   `text-embedding-large` — point these at whatever deployment names are
   provisioned in your AI Core resource group

I verified this against a mock of `gen_ai_hub.proxy.native.openai` that
enforces the same call signature your SDK expects (`model_name=` kwarg,
`response.choices[0].message.content` shape) — classify, score_worklog,
and embed all round-tripped correctly and fed cleanly into the
categorizer/scorer pipeline. I have **not** tested this against your real
SAP AI Core deployment, since I don't have credentials for it — the mock
only proves the calling code is shaped correctly, not that your specific
deployment/model will respond the way the prompts expect. Worth a real
smoke test on your end before relying on it.

If the SDK call fails or isn't configured, `get_llm_client()` automatically
falls back to `rule_based` rather than crashing the app — fail-safe, not
fail-open.

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
