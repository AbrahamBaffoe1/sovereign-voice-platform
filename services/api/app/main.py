"""FastAPI application assembly, lifecycle management, metrics, error mapping, and route registration."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Request
from fastapi.responses import ORJSONResponse, PlainTextResponse
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

from app.api.routes.corpus import router as corpus_router
from app.api.routes.health import router as health_router
from app.api.routes.languages import router as languages_router
from app.api.routes.speech import router as speech_router
from app.api.routes.voices import router as voices_router
from app.api.ws.conversation import router as conversation_router
from app.config import get_settings
from app.core.errors import (
    ConfigurationError,
    ConflictError,
    EngineUnavailableError,
    InvalidAudioError,
    InvalidRequestError,
    ModelInferenceError,
    ResourceNotFoundError,
    UnsupportedLanguageError,
    VoicePlatformError,
)
from app.core.logging import configure_logging
from app.core.metrics import refresh_gpu_metrics
from app.core.security import require_api_key
from app.dependencies import build_container

REQUESTS = Counter("voice_http_requests_total", "HTTP requests", ["method", "path", "status"])
LATENCY = Histogram("voice_http_request_seconds", "HTTP request latency", ["method", "path"])

settings = get_settings()
configure_logging(settings.log_level)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Build the process-wide service container when FastAPI starts and close the optional shared LLM
    HTTP client on shutdown. Heavy speech models remain lazy and are not loaded merely by starting
    the API."""
    app.state.container = build_container(settings)
    yield
    if app.state.container.llm is not None:
        await app.state.container.llm.close()


app = FastAPI(
    title="Sovereign Voice Platform",
    version="0.1.0",
    default_response_class=ORJSONResponse,
    lifespan=lifespan,
)


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    """Measure every HTTP request by method/path and record the final status code. WebSocket turn
    metrics belong at the pipeline/protocol layer rather than this HTTP middleware."""
    path = request.url.path
    with LATENCY.labels(request.method, path).time():
        response = await call_next(request)
    REQUESTS.labels(request.method, path, str(response.status_code)).inc()
    return response


@app.exception_handler(VoicePlatformError)
async def platform_error_handler(_: Request, exc: VoicePlatformError):
    """Translate domain exceptions into stable HTTP status classes without teaching engines about
    FastAPI. Unexpected programming errors are intentionally left to the framework rather than
    disguised as domain failures."""
    status_code = 422
    if isinstance(exc, InvalidAudioError):
        status_code = 400
    elif isinstance(exc, ConflictError):
        status_code = 409
    elif isinstance(exc, InvalidRequestError):
        status_code = 422
    elif isinstance(exc, ResourceNotFoundError):
        status_code = 404
    elif isinstance(exc, EngineUnavailableError):
        status_code = 503
    elif isinstance(exc, ModelInferenceError):
        status_code = 502
    elif isinstance(exc, ConfigurationError):
        status_code = 500
    elif isinstance(exc, UnsupportedLanguageError):
        status_code = 422
    return ORJSONResponse(
        status_code=status_code,
        content={"error": type(exc).__name__, "detail": str(exc)},
    )


@app.get("/metrics", include_in_schema=False)
async def metrics() -> PlainTextResponse:
    """Expose Prometheus text-format metrics without ORJSON encoding so a scraper receives the
    canonical content type and payload."""
    refresh_gpu_metrics()
    return PlainTextResponse(generate_latest().decode(), media_type=CONTENT_TYPE_LATEST)


app.include_router(health_router)
protected = [Depends(require_api_key(settings))]
app.include_router(languages_router, dependencies=protected)
app.include_router(corpus_router, dependencies=protected)
app.include_router(speech_router, dependencies=protected)
app.include_router(voices_router, dependencies=protected)
app.include_router(conversation_router)
