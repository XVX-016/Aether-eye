from __future__ import annotations

import asyncio
import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = Path(__file__).resolve().parents[1]
for path in (str(REPO_ROOT), str(BACKEND_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from pystac_client import Client
from requests import Session
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import yaml
from sqlalchemy import select

from app.core.config import get_settings
from app.core.tasks import create_scene_job, process_scene_job
from app.database.crud import list_enabled_aois, save_scene, upsert_aoi
from app.database.models import SatelliteScene
from app.database.session import async_session


DEFAULT_COPERNICUS_STAC_URL = "https://catalogue.dataspace.copernicus.eu/stac"


def _retrying_session() -> Session:
    session = Session()
    retry = Retry(
        total=5,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "OPTIONS"],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def _load_stac_yaml() -> dict[str, Any]:
    settings = get_settings()
    path = Path(settings.stac_config_path)
    if not path.exists():
        return {"aoi_list": []}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {"aoi_list": []}


async def _sync_aois_from_config() -> None:
    cfg = _load_stac_yaml()
    aois = cfg.get("aoi_list", []) or []
    async with async_session() as session:
        for aoi in aois:
            bbox = aoi.get("bbox")
            if not bbox:
                continue
            await upsert_aoi(
                session,
                aoi_id=str(aoi.get("id") or aoi.get("name")),
                name=str(aoi.get("name") or aoi.get("id")),
                bbox=[float(x) for x in bbox],
                scan_frequency_hrs=int(aoi.get("scan_frequency_hrs", 6)),
                cloud_threshold=float(aoi.get("cloud_threshold", cfg.get("max_cloud_cover", 20))),
                enabled=bool(aoi.get("enabled", True)),
            )
        await session.commit()


async def _insert_new_scenes(*, dry_run: bool = False) -> list[str]:
    await _sync_aois_from_config()
    cfg = _load_stac_yaml()
    stac_url = str(cfg.get("stac_url") or DEFAULT_COPERNICUS_STAC_URL)
    lookback_days = int(cfg.get("lookback_days", 1))
    collections = cfg.get("collections") or ["sentinel-2-l2a"]
    max_items = int(cfg.get("max_items", 20))
    client = Client.open(stac_url)
    now = datetime.now(timezone.utc)
    start = now - timedelta(days=lookback_days)
    datetime_range = f"{start.date().isoformat()}/{now.date().isoformat()}"
    discovered_scene_ids: list[str] = []

    async with async_session() as session:
        aois = await list_enabled_aois(session)
        for aoi in aois:
            bbox = aoi.get("bbox")
            if not bbox:
                continue
            cloud_threshold = float(aoi.get("cloud_threshold", cfg.get("max_cloud_cover", 20.0)))
            search = client.search(
                collections=collections,
                bbox=bbox,
                datetime=datetime_range,
                max_items=max_items,
            )
            items = list(search.items())
            for item in items:
                exists = await session.execute(select(SatelliteScene).where(SatelliteScene.scene_id == item.id))
                if exists.scalar_one_or_none():
                    continue
                cloud_cover = item.properties.get("eo:cloud_cover")
                if cloud_cover is not None and float(cloud_cover) >= cloud_threshold:
                    continue
                asset = item.assets.get("visual") or next(iter(item.assets.values()), None)
                if asset is None:
                    continue
                dt = item.datetime or item.properties.get("datetime")
                if isinstance(dt, str):
                    dt = datetime.fromisoformat(dt.replace("Z", "+00:00"))
                if not dry_run:
                    await save_scene(
                        session,
                        scene_id=item.id,
                        source="copernicus-dataspace",
                        collection="sentinel-2-l2a",
                        aoi_id=aoi["aoi_id"],
                        aoi_name=aoi["name"],
                        dt=dt,
                        bbox=item.bbox,
                        cloud_cover=cloud_cover,
                        asset_href=asset.href,
                        geotiff_path=None,
                        status="DISCOVERED",
                        processed=False,
                    )
                discovered_scene_ids.append(item.id)
        if dry_run:
            await session.rollback()
        else:
            await session.commit()
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
    scene_ids = asyncio.run(run_watcher(dry_run=args.dry_run))
    print(f"discovered={len(scene_ids)}")
    if scene_ids:
        print(scene_ids[:10])


if __name__ == "__main__":
    main()
