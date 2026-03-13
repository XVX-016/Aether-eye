from __future__ import annotations

from typing import Any, Iterable

from satellite_ingestion.stac_query import SentinelIngestor
from app.core.config import get_settings
from app.database.session import async_session
from app.services.ingestion_service import ingest_items_for_aoi, load_stac_config


async def run_once() -> list[str]:
    settings = get_settings()
    cfg = load_stac_config(settings)
    aoi_list = cfg.get("aoi_list", [])
    ingestor = SentinelIngestor(output_dir=cfg.get("download_dir", "data/sentinel2_raw"))

    scene_ids: list[str] = []
    async with async_session() as session:
        for aoi in aoi_list:
            if not aoi.get("enabled", True):
                continue
            aoi_id = aoi["id"]
            bbox = aoi["bbox"]
            time_range = aoi.get("time_range")
            max_cloud_cover = cfg.get("max_cloud_cover", 20)
            max_items = cfg.get("max_items", 20)
            collection = cfg.get("collections", ["sentinel-2-l2a"])[0]
            asset_key = cfg.get("asset_key", "visual")

            # If time_range not provided, ingestion_service will compute based on cursor.
            items = None
            if time_range:
                items = ingestor.query_items(
                    bbox=bbox,
                    time_range=time_range,
                    max_cloud_cover=max_cloud_cover,
                    max_items=max_items,
                )

            added = await ingest_items_for_aoi(
                session=session,
                aoi_id=aoi_id,
                bbox=bbox,
                collection=collection,
                asset_key=asset_key,
                max_cloud_cover=max_cloud_cover,
                max_items=max_items,
                ingestor=ingestor,
                items_override=items,
            )
            scene_ids.extend(added)
        await session.commit()
    return scene_ids
# DEPRECATED: superseded by backend/pipeline/
