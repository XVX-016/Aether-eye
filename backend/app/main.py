from __future__ import annotations

import os
import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import aircraft_inference, change_inference, health, inference, intelligence, onnx_inference, vit_explainability
from app.core.config import get_settings


def create_app() -> FastAPI:
    settings = get_settings()
    if os.name == "nt" and settings.enforce_backend_python:
        running = Path(sys.executable).resolve()
        expected = Path(settings.backend_python_executable).resolve()
        if str(running).lower() != str(expected).lower():
            raise RuntimeError(
                f"Backend must run with {expected}; current interpreter is {running}."
            )

    app = FastAPI(
        title=settings.app_name,
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # Basic CORS: adjust allowed origins as needed.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
        allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?$",
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health.router, prefix="/api")
    app.include_router(inference.router, prefix="/api")
    app.include_router(onnx_inference.router, prefix="/api")
    app.include_router(vit_explainability.router, prefix="/api")
    app.include_router(aircraft_inference.router, prefix="/api")
    app.include_router(change_inference.router, prefix="/api")
    app.include_router(intelligence.router, prefix="/api")

    return app


app = create_app()

