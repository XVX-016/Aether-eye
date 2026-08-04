from __future__ import annotations

import importlib
import logging
import os
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
ml_core_root = REPO_ROOT / "ml_core"
for path in (str(REPO_ROOT), str(ml_core_root)):
    if path not in sys.path:
        sys.path.insert(0, path)

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import func, select

from app.core.config import get_settings
from app.database.crud import backfill_aoi_daily_counts
from app.database.models import AoiDailyCount
from app.database.session import init_db, async_session
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from services.flight_feed import fetch_flights_for_sites
from services.intel_feed import fetch_and_store_articles, retag_existing_articles

logger = logging.getLogger(__name__)


def verify_change_model_assets() -> None:
    # Read config
    cfg_path = REPO_ROOT / "backend/configs/inference/change_detector.yaml"
    if not cfg_path.exists():
        logger.warning(f"Change detector config missing: {cfg_path}")
        return

    with cfg_path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}

    # Resolve onnx path relative to repo root
    onnx_path = REPO_ROOT / cfg["onnx_path"]
    if not onnx_path.exists():
        logger.warning(f"Change model ONNX missing: {onnx_path}")
        return
    logger.info(f"Change model verified: {onnx_path}")


def verify_aircraft_classifier() -> None:
    cfg_path = REPO_ROOT / "backend/configs/inference/aircraft_classifier.yaml"
    if not cfg_path.exists():
        logger.warning(f"Aircraft classifier config missing: {cfg_path}")
        return

    with cfg_path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}

    onnx_path = REPO_ROOT / cfg["onnx_path"]
    if not onnx_path.exists():
        logger.warning(f"Aircraft classifier ONNX missing: {onnx_path}")
        return
    logger.info(f"Aircraft classifier verified: {onnx_path}")


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title=settings.app_name,
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "version": "1.1.0"}

    @app.get("/health/models")
    async def model_health():
        change_onnx = REPO_ROOT / "ml_core/artifacts/change_model_v2/change_model_v2_int8.onnx"
        aircraft_onnx = REPO_ROOT / "ml_core/artifacts/aircraft_classifier_v2/aircraft_classifier_v2_int8.onnx"
        return {
            "status": "ok",
            "change_model": "v2-int8" if change_onnx.exists() else "missing",
            "aircraft_classifier": "v2-military" if aircraft_onnx.exists() else "missing",
            "repo_root": str(REPO_ROOT),
        }

    @app.on_event("startup")
    async def on_startup():
        await init_db()
        # Verify model files exist (no memory cost — models lazy-load on first request)
        try:
            verify_change_model_assets()
        except Exception as exc:
            logger.warning("Change model verification skipped: %s", exc)
        try:
            verify_aircraft_classifier()
        except Exception as exc:
            logger.warning("Aircraft classifier verification skipped: %s", exc)
        # NOTE: YOLO detector, change model, and aircraft classifier are NOT
        # loaded into memory here. They lazy-load via @lru_cache on first
        # inference request. This keeps startup RAM at ~150MB for free-tier.
        try:
            async with async_session() as session:
                existing_detection_count = await session.scalar(
                    select(func.count()).select_from(AoiDailyCount).where(AoiDailyCount.event_type == "detection")
                )
                if int(existing_detection_count or 0) == 0:
                    backfilled = await backfill_aoi_daily_counts(session)
                    await session.commit()
                    logger.info("Backfilled %s aoi_daily_count rows", backfilled)
        except Exception as exc:
            logger.warning("AOI daily count backfill skipped due to error: %s", exc)
        if settings.enable_intel_fetch_on_startup:
            try:
                async with async_session() as session:
                    stored = await fetch_and_store_articles(session)
                    logger.info("Intel feed: stored %s new articles", stored)
                    retagged = await retag_existing_articles(session)
                    logger.info("Intel re-tagged %s existing articles", retagged)
            except Exception as exc:
                logger.warning("Intel feed startup fetch skipped due to error: %s", exc)
        else:
            logger.info("Intel feed startup fetch disabled")
        if settings.enable_flight_fetch_on_startup:
            try:
                async with async_session() as session:
                    stored = await fetch_flights_for_sites(session)
                    logger.info("Flight feed: stored %d states", stored)
                    await session.commit()
            except Exception as exc:
                logger.warning("Flight feed startup failed: %s", exc)
        else:
            logger.info("Flight feed startup fetch disabled")
        scheduler = AsyncIOScheduler()
        app.state.scheduler = scheduler

        async def stac_job():
            from pipeline.stac_watcher import run_watcher

            await run_watcher()

        async def activity_job():
            from app.services.activity_service import aggregate_aircraft_activity

            async with async_session() as session:
                await aggregate_aircraft_activity(
                    session,
                    window_hours=settings.activity_window_hours,
                    surge_factor=settings.activity_surge_factor,
                    min_count=settings.activity_min_count,
                )
                await session.commit()

        async def intel_feed_job():
            try:
                async with async_session() as session:
                    stored = await fetch_and_store_articles(session)
                    logger.info("Intel feed: stored %s new articles", stored)
                    retagged = await retag_existing_articles(session)
                    logger.info("Intel re-tagged %s existing articles", retagged)
            except Exception as exc:
                logger.warning("Intel feed scheduled fetch skipped due to error: %s", exc)

        async def flight_feed_job():
            try:
                async with async_session() as session:
                    stored = await fetch_flights_for_sites(session)
                    logger.info("Flight feed: stored %d states", stored)
                    await session.commit()
            except Exception as exc:
                logger.warning("Flight feed scheduled fetch skipped due to error: %s", exc)

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
        scheduler.add_job(
            intel_feed_job,
            "interval",
            minutes=30,
            id="intel_feed_fetch",
            max_instances=1,
        )
        scheduler.add_job(
            flight_feed_job,
            "interval",
            minutes=5,
            id="flight_feed_fetch",
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

    route_modules = [
        "health",
        "inference",
        "onnx_inference",
        "vit_explainability",
        "aircraft_inference",
        "change_inference",
        "intelligence",
        "live_aircraft",
        "operations",
    ]
    for module_name in route_modules:
        try:
            module = importlib.import_module(f"app.api.routes.{module_name}")
            app.include_router(module.router, prefix="/api")
        except Exception as exc:
            logger.warning("Skipping route module %s due to import error: %s", module_name, exc)

    return app


app = create_app()

