import logging

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from app.config import get_settings
from app.routers import analyze

settings = get_settings()

logging.basicConfig(level=settings.LOG_LEVEL)
logger = logging.getLogger(__name__)

limiter = Limiter(key_func=get_remote_address, default_limits=[settings.RATE_LIMIT])

app = FastAPI(
    title=settings.APP_NAME,
    version="1.0.0",
    docs_url="/api/docs" if settings.ENV != "prod" else None,  # hide docs in prod
    redoc_url=None,
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in settings.ALLOWED_ORIGINS.split(",") if o.strip()],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["X-API-Key", "Content-Type"],
)


@app.middleware("http")
async def security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    return response


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    # Never leak stack traces / internals to the client.
    logger.exception("Unhandled exception on %s", request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Internal server error."})


@app.get("/health", tags=["ops"])
async def health():
    return {"status": "ok", "app": settings.APP_NAME, "provider": settings.LLM_PROVIDER}


app.include_router(analyze.router)

# --- Mount Gradio dashboard at the app root so its bundled static assets
# (which reference absolute "/assets/..." paths) resolve correctly. The
# REST API lives under /api/v1/... and /health, both registered above and
# therefore matched before this catch-all mount. ---
from ui.gradio_app import CUSTOM_CSS, build_ui  # noqa: E402  (import after app creation intentional)
import gradio as gr  # noqa: E402

gr.mount_gradio_app(app, build_ui(), path="/", theme=gr.themes.Soft(primary_hue="blue"), css=CUSTOM_CSS)
