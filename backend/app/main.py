from __future__ import annotations

import os
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import aircraft_inference, change_inference, health, inference, intelligence, onnx_inference, vit_explainability
from app.core.config import get_settings
from app.database.session import init_db, async_session
import asyncio
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from satellite_ingestion.stac_watcher import run_once as run_stac_watcher
from app.core.tasks import create_scene_job, process_scene_job
from app.services.activity_service import aggregate_aircraft_activity


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title=settings.app_name,
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    @app.on_event("startup")
    async def on_startup():
        await init_db()
        scheduler = AsyncIOScheduler()
        app.state.scheduler = scheduler

        async def stac_job():
            scene_ids = await run_stac_watcher()
            for scene_id in scene_ids:
                job_id = create_scene_job(scene_id)
                asyncio.create_task(process_scene_job(scene_id, job_id))

        async def activity_job():
            async with async_session() as session:
                await aggregate_aircraft_activity(
                    session,
                    window_hours=settings.activity_window_hours,
                    surge_factor=settings.activity_surge_factor,
                    min_count=settings.activity_min_count,
                )
                await session.commit()

        if settings.enable_stac_watcher:
            scheduler.add_job(
                stac_job,
                "interval",
                minutes=settings.stac_poll_minutes,
                id="stac_watcher",
                max_instances=1,
            )
        if settings.enable_activity_aggregator:
            scheduler.add_job(
                activity_job,
                "interval",
                hours=settings.activity_window_hours,
                id="activity_aggregator",
                max_instances=1,
            )
        scheduler.start()

    @app.on_event("shutdown")
    async def on_shutdown():
        scheduler = getattr(app.state, "scheduler", None)
        if scheduler:
            scheduler.shutdown()

    if os.name == "nt" and settings.enforce_backend_python:
        running = Path(sys.executable).resolve()
        expected = Path(settings.backend_python_executable).resolve()
        if str(running).lower() != str(expected).lower():
            raise RuntimeError(
                f"Backend must run with {expected}; current interpreter is {running}."
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

