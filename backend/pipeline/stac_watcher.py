from __future__ import annotations

import argparse
import asyncio
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys
from typing import Any

import httpx
import yaml
from sqlalchemy import select

REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = Path(__file__).resolve().parents[1]
for path in (str(REPO_ROOT), str(BACKEND_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from app.core.config import get_settings
from app.core.tasks import create_scene_job, process_scene_job
from app.database.crud import save_scene, upsert_aoi
from app.database.models import SatelliteScene
from app.database.session import async_session
from pipeline.site_registry import load_sites

logger = logging.getLogger(__name__)

DEFAULT_COPERNICUS_STAC_URL = "https://catalogue.dataspace.copernicus.eu/stac"


def _load_stac_yaml() -> dict[str, Any]:
    settings = get_settings()
    path = Path(settings.stac_config_path)
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


async def _sync_sites_to_registry() -> None:
    cfg = _load_stac_yaml()
    cloud_threshold = float(cfg.get("cloud_cover_max", cfg.get("max_cloud_cover", 30)))
    async with async_session() as session:
        for site in load_sites():
            bbox = site.get("bbox")
            if not bbox:
                continue
            await upsert_aoi(
                session,
                aoi_id=str(site["id"]),
                name=str(site.get("name") or site["id"]),
                bbox=[float(x) for x in bbox],
                scan_frequency_hrs=int(site.get("scan_frequency_hrs", 6)),
                cloud_threshold=cloud_threshold,
                enabled=True,
            )
        await session.commit()


async def search_stac(
    *,
    client: httpx.AsyncClient,
    stac_url: str,
    bbox: list[float],
    lookback_days: int,
    cloud_cover_max: int,
    collection: str,
    max_items: int,
) -> list[dict[str, Any]]:
    now = datetime.now(timezone.utc)
    start = now - timedelta(days=lookback_days)
    datetime_range = f"{start.isoformat().replace('+00:00', 'Z')}/{now.isoformat().replace('+00:00', 'Z')}"
    search_url = stac_url.rstrip("/") + "/search"
    payload = {
        "collections": [collection],
        "bbox": bbox,
        "datetime": datetime_range,
        "limit": max_items,
        "query": {"eo:cloud_cover": {"lte": cloud_cover_max}},
    }
    response = await client.post(search_url, json=payload)
    response.raise_for_status()
    data = response.json()
    return list(data.get("features") or [])


async def discover_scenes_for_all_sites(
    client: httpx.AsyncClient,
    lookback_days: int = 14,
    cloud_cover_max: int = 30,
    *,
    stac_url: str = DEFAULT_COPERNICUS_STAC_URL,
    collection: str = "sentinel-2-l2a",
    max_items: int = 20,
) -> list[dict[str, Any]]:
    sites = load_sites()
    all_scenes: list[dict[str, Any]] = []
    seen_scene_ids: set[str] = set()

    for site in sites:
        try:
            bbox = [float(x) for x in site["bbox"]]
            scenes: list[dict[str, Any]] = []
            for attempt in range(2):
                try:
                    scenes = await search_stac(
                        client=client,
                        stac_url=stac_url,
                        bbox=bbox,
                        lookback_days=lookback_days,
                        cloud_cover_max=cloud_cover_max,
                        collection=collection,
                        max_items=max_items,
                    )
                    break
                except httpx.HTTPStatusError as exc:
                    if exc.response.status_code == 429 and attempt == 0:
                        logger.warning("STAC 429 for %s, retrying in 30s", site["id"])
                        await asyncio.sleep(30)
                        continue
                    logger.warning("STAC search failed for %s: %s", site["id"], exc)
                    scenes = []
                    break
                except Exception as exc:
                    logger.warning("STAC search failed for %s: %s", site["id"], exc)
                    scenes = []
                    break

            for scene in scenes:
                scene_id = str(scene.get("id") or "")
                if not scene_id or scene_id in seen_scene_ids:
                    continue
                properties = scene.setdefault("properties", {})
                properties["aoi_id"] = site["id"]
                properties["aoi_name"] = site["name"]
                all_scenes.append(scene)
                seen_scene_ids.add(scene_id)

            logger.info("STAC site %s: %d scenes", site["id"], len(scenes))
            await asyncio.sleep(3)
        except Exception as exc:
            logger.warning("STAC search failed for %s: %s", site["id"], exc)
            continue

    return all_scenes


async def _insert_new_scenes(*, dry_run: bool = False) -> list[str]:
    await _sync_sites_to_registry()
    cfg = _load_stac_yaml()
    stac_url = str(cfg.get("stac_url") or DEFAULT_COPERNICUS_STAC_URL)
    lookback_days = int(cfg.get("lookback_days", 14))
    cloud_cover_max = int(cfg.get("cloud_cover_max", cfg.get("max_cloud_cover", 30)))
    collections = cfg.get("collections") or ["sentinel-2-l2a"]
    collection = str(collections[0])
    max_items = int(cfg.get("max_items", 20))
    discovered_scene_ids: list[str] = []
    total_discovered = 0
    total_new = 0

    timeout = httpx.Timeout(30.0)
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
        scenes = await discover_scenes_for_all_sites(
            client,
            lookback_days=lookback_days,
            cloud_cover_max=cloud_cover_max,
            stac_url=stac_url,
            collection=collection,
            max_items=max_items,
        )

    total_discovered = len(scenes)

    async with async_session() as session:
        for item in scenes:
            scene_id = str(item.get("id") or "")
            if not scene_id:
                continue
            exists = await session.execute(select(SatelliteScene).where(SatelliteScene.scene_id == scene_id))
            if exists.scalar_one_or_none():
                discovered_scene_ids.append(scene_id)
                continue

            properties = item.get("properties") or {}
            cloud_cover = properties.get("eo:cloud_cover")
            if cloud_cover is not None and float(cloud_cover) > cloud_cover_max:
                continue

            assets = item.get("assets") or {}
            asset = assets.get("visual") or assets.get("rendered_preview") or next(iter(assets.values()), None)
            asset_href = asset.get("href") if isinstance(asset, dict) else getattr(asset, "href", None)
            if not asset_href:
                continue

            dt = properties.get("datetime") or item.get("datetime")
            if isinstance(dt, str):
                dt = datetime.fromisoformat(dt.replace("Z", "+00:00"))
            elif not isinstance(dt, datetime):
                dt = datetime.now(timezone.utc)

            if not dry_run:
                await save_scene(
                    session,
                    scene_id=scene_id,
                    source="copernicus-dataspace",
                    collection=collection,
                    aoi_id=properties.get("aoi_id"),
                    aoi_name=properties.get("aoi_name"),
                    dt=dt,
                    bbox=item.get("bbox"),
                    cloud_cover=cloud_cover,
                    asset_href=asset_href,
                    geotiff_path=None,
                    status="DISCOVERED",
                    processed=False,
                )
                total_new += 1
            discovered_scene_ids.append(scene_id)

        if dry_run:
            await session.rollback()
        else:
            await session.commit()

    logger.info(
        "STAC scan complete: %d sites, %d total scenes discovered, %d new",
        len(load_sites()),
        total_discovered,
        total_new,
    )
    return discovered_scene_ids


async def run_watcher(*, dry_run: bool = False) -> list[str]:
    scene_ids = await _insert_new_scenes(dry_run=dry_run)
    if dry_run:
        return scene_ids
    for scene_id in scene_ids:
        job_id = create_scene_job(scene_id)
        asyncio.create_task(process_scene_job(scene_id, job_id))
    return scene_ids


async def run_once(*, dry_run: bool = False) -> list[str]:
    return await run_watcher(dry_run=dry_run)


def main() -> None:
    parser = argparse.ArgumentParser(description="Query Copernicus Data Space STAC for new scenes.")
    parser.add_argument("--dry-run", action="store_true", help="Query and list candidate scenes without inserting or dispatching jobs.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    scene_ids = asyncio.run(run_watcher(dry_run=args.dry_run))
    print(f"discovered={len(scene_ids)}")
    if scene_ids:
        print(scene_ids[:10])


if __name__ == "__main__":
    main()
