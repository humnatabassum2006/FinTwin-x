"""
FinTwin-X API — FastAPI application factory.

Security posture (see docs/threat_model.md):
  * JWT auth endpoints + optional auth on read endpoints
  * PBKDF2 password hashing, no plaintext credentials anywhere
  * per-IP rate limiting, security headers, request audit log
  * Pydantic validation on every request body (strict, extra fields rejected)
  * no real bank credentials are ever used: the platform runs on synthetic data
"""
from __future__ import annotations

import time
import uuid

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

from apps.api.routes import agent, analytics, auth as auth_routes, meta, risk, simulation
from apps.api.security import seed_demo_user
from common import settings
from common.logging_utils import audit, get_logger

log = get_logger("api")

HTTP_REQUESTS = Counter(
    "fintwin_http_requests_total",
    "HTTP requests handled by FinTwin-X",
    ("method", "route", "status"),
)
HTTP_LATENCY = Histogram(
    "fintwin_http_request_duration_seconds",
    "FinTwin-X HTTP request duration",
    ("method", "route"),
)

DESCRIPTION = """
**FinTwin-X** — AI Financial Digital Twin & Probabilistic Stress-Testing Engine.

Everything here runs on **synthetic data**. No real banking credentials are used
or stored. Predictions are model estimates with probabilities and assumptions —
never guarantees.
"""


def create_app() -> FastAPI:
    app = FastAPI(
        title="FinTwin-X API",
        version="2.0.0",
        description=DESCRIPTION,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.CORS_ORIGINS),
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ------------------------------------------------------------------ middleware
    @app.middleware("http")
    async def security_and_audit(request: Request, call_next):
        rid = request.headers.get("x-request-id", uuid.uuid4().hex[:12])
        start = time.time()
        response = await call_next(request)
        elapsed = time.time() - start
        duration_ms = round(elapsed * 1000, 1)
        route_obj = request.scope.get("route")
        route_path = getattr(route_obj, "path", request.url.path)
        HTTP_REQUESTS.labels(request.method, route_path, str(response.status_code)).inc()
        HTTP_LATENCY.labels(request.method, route_path).observe(elapsed)
        response.headers["X-Request-ID"] = rid
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "SAMEORIGIN"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
        if request.url.path.startswith(("/docs", "/redoc")):
            response.headers["Content-Security-Policy"] = (
                "default-src 'self' https://cdn.jsdelivr.net; img-src 'self' data: https://fastapi.tiangolo.com; "
                "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
                "script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net"
            )
        else:
            response.headers["Content-Security-Policy"] = (
                "default-src 'self'; img-src 'self' data:; style-src 'self' 'unsafe-inline'; "
                "script-src 'self' 'unsafe-inline'; connect-src 'self'"
            )
        if settings.ENVIRONMENT in {"prod", "production"}:
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["X-Response-Time-ms"] = str(duration_ms)
        if not request.url.path.startswith(("/docs", "/redoc", "/openapi.json", "/static", "/metrics/prometheus")):
            audit.write(
                actor=(request.client.host if request.client else "anonymous"),
                action="http_request",
                resource=f"{request.method} {request.url.path}",
                status=str(response.status_code),
                request_id=rid, duration_ms=duration_ms,
            )
        return response

    # ------------------------------------------------------------------ routers
    app.include_router(meta.router)
    app.include_router(auth_routes.router)
    app.include_router(analytics.router)
    app.include_router(risk.router)
    app.include_router(simulation.router)
    app.include_router(agent.router)

    @app.get("/metrics/prometheus", include_in_schema=False)
    def prometheus_metrics():
        return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

    # ------------------------------------------------------------------ static UI
    # Production image contains a statically exported Next.js app.  Serving it
    # from FastAPI keeps browser/API traffic same-origin and gives one public URL.
    web_out = settings.ROOT / "apps" / "web" / "out"
    legacy_static = settings.ROOT / "apps" / "api" / "static"
    legacy_static.mkdir(parents=True, exist_ok=True)
    app.mount("/static", StaticFiles(directory=str(legacy_static)), name="static")

    if web_out.exists() and (web_out / "index.html").exists():
        # Keep this mount last so API/OpenAPI routes above always win.
        app.mount("/", StaticFiles(directory=str(web_out), html=True), name="web")
    else:
        @app.get("/", include_in_schema=False)
        def index():
            dash = legacy_static / "dashboard.html"
            if dash.exists():
                return FileResponse(dash)
            return JSONResponse({"message": "FinTwin-X API is running. See /docs."})

    # ------------------------------------------------------------------ startup
    @app.on_event("startup")
    def on_startup():
        if settings.ENABLE_DEMO_USER:
            seed_demo_user()
        try:
            from rag.retrieval.retriever import get_retriever
            r = get_retriever()
            log.info("RAG vector store ready (%s)", r.backend)
        except Exception as exc:                                # noqa: BLE001
            log.warning("RAG store not ready: %s", exc)
        log.info("FinTwin-X API started (llm=%s)", bool(settings.LLM_API_KEY))

    return app


app = create_app()
