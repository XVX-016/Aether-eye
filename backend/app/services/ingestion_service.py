from __future__ import annotations

import json
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Iterable

import yaml
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings
from app.database.models import IngestionState, SatelliteScene
from satellite_ingestion.stac_query import SentinelIngestor


def load_stac_config(settings: Settings) -> dict[str, Any]:
    cfg_path = Path(settings.stac_config_path)
    cfg: dict[str, Any] = {}
    if cfg_path.exists():
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    if settings.stac_aois_json:
        try:
            cfg["aoi_list"] = json.loads(settings.stac_aois_json)
        except Exception:
            pass
    cfg.setdefault("collections", ["sentinel-2-l2a"])
    cfg.setdefault("asset_key", "visual")
    cfg.setdefault("max_cloud_cover", 20)
    cfg.setdefault("poll_minutes", settings.stac_poll_minutes)
    cfg.setdefault("download_dir", "data/sentinel2_raw")
    cfg.setdefault("max_items", 20)
    cfg.setdefault("aoi_list", [])
    return cfg


def _get_item_datetime(item: Any) -> datetime:
    dt = item.datetime or item.properties.get("datetime") or item.properties.get("start_datetime")
    if isinstance(dt, str):
        return datetime.fromisoformat(dt.replace("Z", "+00:00"))
    return dt


def _scene_key(item: Any, collection: str) -> tuple[str, str]:
    return (item.id, collection)


def filter_new_items(items: Iterable[Any], existing_keys: set[tuple[str, str]], collection: str) -> list[Any]:
    new_items: list[Any] = []
    for item in items:
        if _scene_key(item, collection) in existing_keys:
            continue
        new_items.append(item)
    return new_items


async def _get_last_timestamp(session: AsyncSession, aoi_id: str) -> datetime | None:
    result = await session.execute(select(IngestionState).where(IngestionState.aoi_id == aoi_id))
    state = result.scalar_one_or_none()
    return state.last_timestamp if state else None


async def _set_last_timestamp(session: AsyncSession, aoi_id: str, ts: datetime | None) -> None:
    result = await session.execute(select(IngestionState).where(IngestionState.aoi_id == aoi_id))
    state = result.scalar_one_or_none()
    if state is None:
        state = IngestionState(aoi_id=aoi_id, last_timestamp=ts)
        session.add(state)
    else:
        state.last_timestamp = ts


async def ingest_items_for_aoi(
    *,
    session: AsyncSession,
    aoi_id: str,
    bbox: list[float],
    collection: str,
    asset_key: str,
    max_cloud_cover: float,
    max_items: int,
    ingestor: SentinelIngestor,
    items_override: Iterable[Any] | None = None,
) -> list[str]:
    last_ts = await _get_last_timestamp(session, aoi_id)
    now = datetime.now(timezone.utc)
    if last_ts:
        time_range = f"{last_ts.isoformat()}/{now.isoformat()}"
    else:
        # default to last 7 days if no cursor
        start = now - timedelta(days=7)
        time_range = f"{start.isoformat()}/{now.isoformat()}"

    if items_override is None:
        items = ingestor.query_items(
            bbox=bbox,
            time_range=time_range,
            max_cloud_cover=max_cloud_cover,
            max_items=max_items,
        )
    else:
        items = list(items_override)

    existing = await session.execute(
        select(SatelliteScene.scene_id, SatelliteScene.collection)
        .where(SatelliteScene.collection == collection)
    )
    existing_keys = {(row[0], row[1]) for row in existing.all()}

    added: list[str] = []
    newest_ts = last_ts
    for item in filter_new_items(items, existing_keys, collection):
        if _scene_key(item, collection) in existing_keys:
            continue
        dt = _get_item_datetime(item)
        if dt and (newest_ts is None or dt > newest_ts):
            newest_ts = dt
        asset = item.assets.get(asset_key)
        if not asset:
            continue
        scene = SatelliteScene(
            scene_id=item.id,
            collection=collection,
            datetime=dt,
            bbox=item.bbox,
            cloud_cover=item.properties.get("eo:cloud_cover"),
            asset_href=asset.href,
            status="NEW",
        )
        session.add(scene)
        added.append(scene.scene_id)

    if newest_ts:
        await _set_last_timestamp(session, aoi_id, newest_ts)
    return added
