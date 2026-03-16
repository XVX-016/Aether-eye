from __future__ import annotations

import argparse
import asyncio
import math
import random
import sys
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path

from geoalchemy2.elements import WKTElement
from sqlalchemy import delete
from sqlalchemy.dialects.postgresql import insert

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT))

from app.database.crud import save_detection, save_event, save_scene, upsert_aoi  # noqa: E402
from app.database.models import AoiDailyCount, IntelArticle, ObjectEvent, SatelliteScene, TileDetection  # noqa: E402
from app.database.session import async_session  # noqa: E402
from pipeline.site_registry import load_sites  # noqa: E402


UTC = timezone.utc
SCENE_PREFIX = "DEMO_S2A_MSIL2A_"
DEMO_INTEL_TIER = 99
RNG = random.Random(42)


@dataclass(frozen=True)
class SceneSpec:
    site_id: str
    tile_code: str
    offset_days: int


SITE_IDS = [
    "al_dhafra",
    "al_udeid",
    "strait_hormuz_north",
    "kadena",
    "ramstein",
    "dubai_airport",
]

SCENE_SPECS: list[SceneSpec] = [
    SceneSpec("al_dhafra", "T40RBM", 14),
    SceneSpec("al_dhafra", "T40RBM", 7),
    SceneSpec("al_udeid", "T39RXH", 13),
    SceneSpec("al_udeid", "T39RXH", 5),
    SceneSpec("strait_hormuz_north", "T40RDN", 12),
    SceneSpec("strait_hormuz_north", "T40RDN", 2),
    SceneSpec("kadena", "T52RTP", 11),
    SceneSpec("kadena", "T52RTP", 4),
    SceneSpec("ramstein", "T32UMB", 9),
    SceneSpec("dubai_airport", "T40RCN", 3),
]

SITE_DAY_RANGES: dict[str, tuple[int, int]] = {
    "al_dhafra": (120, 280),
    "al_udeid": (100, 250),
    "strait_hormuz_north": (80, 200),
    "kadena": (90, 220),
    "ramstein": (70, 180),
    "dubai_airport": (60, 160),
}

INTEL_HEADLINES: dict[str, list[str]] = {
    "al_udeid": [
        "CENTCOM Conducts Joint Air Operations from Al Udeid Base",
        "US Air Force Deploys Additional F-35s to Qatar Amid Regional Tensions",
        "Al Udeid Air Base Expands Drone Operations in Middle East Theater",
        "379th Air Expeditionary Wing Completes Record Sortie Count",
        "Qatar and US Renew Defense Cooperation Agreement Through 2030",
    ],
    "strait_hormuz_north": [
        "IRGC Navy Conducts Exercises Near Strait of Hormuz",
        "Commercial Shipping Reports Increased Iranian Naval Presence",
        "US Fifth Fleet Issues Advisory for Hormuz Transit Corridor",
        "Bandar Abbas Port Sees Unusual Naval Asset Concentration",
        "Satellite Imagery Shows New Vessel Deployments at Iranian Base",
    ],
    "al_dhafra": [
        "UAE Air Force Conducts Large-Scale Exercise at Al Dhafra",
        "B-2 Spirit Bombers Reportedly Staged Through Gulf Region",
        "Al Dhafra Hosts Multinational Air Combat Training Exercise",
    ],
    "kadena": [
        "USAF Kadena Increases Readiness Posture Amid Pacific Tensions",
        "F-15 Squadron Rotation Completed at Kadena Air Base Okinawa",
        "US Japan Alliance Conducts Combined Air Defense Exercise",
    ],
    "ramstein": [
        "USAFE Command Coordinates European Air Mobility Operations",
        "Ramstein Air Base Central to NATO Eastern Flank Logistics",
    ],
    "diego_garcia": [
        "B-52 Strategic Bombers Deployed to Diego Garcia for Deterrence",
        "Indian Ocean Base Supports Ongoing Maritime Security Operations",
    ],
}


def point_wkt(lat: float, lon: float) -> WKTElement:
    return WKTElement(f"POINT({lon} {lat})", srid=4326)


def scene_datetime(offset_days: int) -> datetime:
    target_day = datetime.now(UTC).date() - timedelta(days=offset_days)
    return datetime.combine(target_day, time(6, 53, 1), tzinfo=UTC)


def build_scene_id(scene_dt: datetime, tile_code: str) -> str:
    processed_dt = scene_dt + timedelta(days=1, hours=5, minutes=37, seconds=43)
    return (
        f"{SCENE_PREFIX}{scene_dt:%Y%m%d}T065301_R020_"
        f"{tile_code}_{processed_dt:%Y%m%d}T123044"
    )


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def random_point(bbox: list[float]) -> tuple[float, float]:
    min_lon, min_lat, max_lon, max_lat = bbox
    lat = RNG.uniform(min_lat, max_lat)
    lon = RNG.uniform(min_lon, max_lon)
    return lat, lon


def random_bbox_around(lat: float, lon: float) -> list[float]:
    lat_delta = RNG.uniform(0.0008, 0.0035)
    lon_delta = RNG.uniform(0.0008, 0.0035)
    return [
        round(lon - lon_delta, 6),
        round(lat - lat_delta, 6),
        round(lon + lon_delta, 6),
        round(lat + lat_delta, 6),
    ]


def scene_detection_range(site: dict[str, object]) -> tuple[int, int]:
    site_type = str(site.get("type", ""))
    priority = str(site.get("priority", ""))
    if priority == "critical" and site_type == "MILITARY_AIRBASE":
        return (150, 300)
    if priority == "high" and site_type in {"CIVIL_AIRPORT", "STRATEGIC_PORT"}:
        return (80, 180)
    return (40, 100)


def event_priority(site: dict[str, object]) -> str:
    priority = str(site.get("priority", ""))
    if priority == "critical":
        return "HIGH"
    if priority == "high":
        return RNG.choice(["HIGH", "MEDIUM"])
    return "MEDIUM"


def event_type_pick() -> str:
    roll = RNG.random()
    if roll < 0.60:
        return "ACTIVITY_SURGE"
    if roll < 0.85:
        return "NEW_OBJECT"
    return "ELEVATED_ACTIVITY"


def intel_source_for_index(index: int) -> tuple[str, int]:
    options = [
        ("DEMO Reuters", DEMO_INTEL_TIER),
        ("DEMO Breaking Defense", DEMO_INTEL_TIER),
        ("DEMO Aviationist", DEMO_INTEL_TIER),
        ("DEMO Strategic Monitor", DEMO_INTEL_TIER),
    ]
    return options[index % len(options)]


async def clear_existing_demo_rows(db) -> None:
    demo_scene_ids = (
        await db.execute(
            delete(TileDetection)
            .where(TileDetection.scene_id.like(f"{SCENE_PREFIX}%"))
            .returning(TileDetection.id)
        )
    ).scalars().all()
    await db.execute(delete(ObjectEvent).where(ObjectEvent.scene_id.like(f"{SCENE_PREFIX}%")))
    await db.execute(delete(SatelliteScene).where(SatelliteScene.scene_id.like(f"{SCENE_PREFIX}%")))
    await db.execute(delete(IntelArticle).where(IntelArticle.source_tier == DEMO_INTEL_TIER))
    await db.flush()


async def seed_aois(db, sites_by_id: dict[str, dict[str, object]]) -> None:
    for site_id in SITE_IDS:
        site = sites_by_id[site_id]
        await upsert_aoi(
            db,
            aoi_id=site_id,
            name=str(site["name"]),
            bbox=list(site["bbox"]),
            scan_frequency_hrs=6,
            cloud_threshold=30.0,
            enabled=True,
        )


async def seed_scenes(db, sites_by_id: dict[str, dict[str, object]]) -> list[tuple[str, dict[str, object], datetime]]:
    seeded: list[tuple[str, dict[str, object], datetime]] = []
    for spec in SCENE_SPECS:
        site = sites_by_id[spec.site_id]
        dt = scene_datetime(spec.offset_days)
        scene_id = build_scene_id(dt, spec.tile_code)
        await save_scene(
            db,
            scene_id=scene_id,
            source="demo-seed",
            collection="sentinel-2-l2a",
            aoi_id=spec.site_id,
            aoi_name=str(site["name"]),
            dt=dt,
            bbox=list(site["bbox"]),
            cloud_cover=round(RNG.uniform(5.0, 25.0), 1),
            asset_href=f"demo://{scene_id}.SAFE",
            geotiff_path=f"/demo/{scene_id}.tif",
            status="PROCESSED",
            processed=True,
        )
        seeded.append((scene_id, site, dt))
    await db.commit()
    return seeded


async def seed_detections(db, scenes: list[tuple[str, dict[str, object], datetime]]) -> int:
    total = 0
    for scene_id, site, _ in scenes:
        lower, upper = scene_detection_range(site)
        count = RNG.randint(lower, upper)
        bbox = list(site["bbox"])
        for index in range(count):
            lat, lon = random_point(bbox)
            score = clamp(RNG.normalvariate(0.35, 0.14), 0.05, 0.95)
            await save_detection(
                db,
                scene_id=scene_id,
                tile_x=index % 32,
                tile_y=(index // 32) % 32,
                lat=lat,
                lon=lon,
                model_type="change_detection",
                change_score=round(score, 4),
                confidence=round(score, 4),
                detection_class="terrain_change",
                bbox=random_bbox_around(lat, lon),
                metadata_json={"changed_pixels": RNG.randint(100, 50000)},
            )
        total += count
    await db.commit()
    return total


async def seed_events(db, scenes: list[tuple[str, dict[str, object], datetime]]) -> int:
    total = 0
    for scene_index, (scene_id, site, scene_dt) in enumerate(scenes):
        count = RNG.randint(4, 10)
        bbox = list(site["bbox"])
        for event_index in range(count):
            lat, lon = random_point(bbox)
            event_type = event_type_pick()
            surge_factor = round(RNG.uniform(1.5, 4.5), 2) if event_type == "ACTIVITY_SURGE" else 1.0
            event_id = f"demo_evt_{scene_index:02d}_{event_index:02d}_{spec_safe(scene_id)}"
            event = await save_event(
                db,
                event_id=event_id,
                event_type=event_type,
                scene_id=scene_id,
                lat=lat,
                lon=lon,
                confidence=round(RNG.uniform(0.45, 0.92), 4),
                priority=event_priority(site),
                detection_class="terrain_change",
                metadata_json={
                    "surge_factor": surge_factor,
                    "demo_data": True,
                    "site_id": site["id"],
                },
            )
            event.timestamp = scene_dt + timedelta(minutes=event_index * 7)
            event.created_at = event.timestamp
        total += count
    await db.commit()
    return total


async def seed_baselines(db) -> int:
    today = datetime.now(UTC).date()
    rows: list[dict[str, object]] = []
    for site_id, (lower, upper) in SITE_DAY_RANGES.items():
        for day_offset in range(29, -1, -1):
            target_date = today - timedelta(days=day_offset)
            count = RNG.randint(lower, upper)
            if day_offset == 0 and site_id == "al_dhafra":
                count = upper + 110
            if day_offset == 0 and site_id == "strait_hormuz_north":
                count = upper + 85
            rows.append(
                {
                    "aoi_id": site_id,
                    "date": target_date,
                    "event_type": "detection",
                    "count": count,
                }
            )

    stmt = insert(AoiDailyCount).values(rows)
    stmt = stmt.on_conflict_do_update(
        constraint="uq_aoi_daily_counts_aoi_date_event",
        set_={"count": stmt.excluded.count},
    )
    await db.execute(stmt)
    await db.commit()
    return len(rows)


async def seed_intel(db) -> int:
    now = datetime.now(UTC)
    total = 0
    for site_id, headlines in INTEL_HEADLINES.items():
        for idx, title in enumerate(headlines):
            source, tier = intel_source_for_index(total)
            published_at = now - timedelta(hours=(idx * 3) + (total % 5) + 1)
            article = IntelArticle(
                site_id=site_id,
                title=title,
                url=f"https://demo.local/intel/{site_id}/{slugify(title)}",
                source=source,
                source_tier=tier,
                published_at=published_at,
                fetched_at=now,
                summary=f"Demo intelligence brief for {site_id.replace('_', ' ')}. Offline-safe seeded article.",
            )
            db.add(article)
            total += 1
    await db.commit()
    return total


def slugify(value: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "-" for ch in value).strip("-")


def spec_safe(scene_id: str) -> str:
    return scene_id.replace("-", "_")[-18:]


async def run(reset: bool) -> None:
    sites_by_id = {site["id"]: site for site in load_sites()}
    missing = [site_id for site_id in SITE_IDS if site_id not in sites_by_id]
    if missing:
        raise RuntimeError(f"Missing site config for: {', '.join(missing)}")

    async with async_session() as db:
        if reset:
            await clear_existing_demo_rows(db)
            await db.commit()
        else:
            await clear_existing_demo_rows(db)
            await db.commit()

        await seed_aois(db, sites_by_id)
        await db.commit()

        scenes = await seed_scenes(db, sites_by_id)
        detections = await seed_detections(db, scenes)
        events = await seed_events(db, scenes)
        baselines = await seed_baselines(db)
        intel = await seed_intel(db)

    print(f"Seeding scenes...      {len(scenes)} created")
    print(f"Seeding detections...  {detections} created")
    print(f"Seeding events...      {events} created")
    print(f"Seeding baselines...   {baselines} rows upserted")
    print(f"Seeding intel...       {intel} articles created")
    print("-" * 41)
    print("Demo data ready.")
    print("Dashboard: http://localhost:3000/operations")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Seed Aether-Eye demo data.")
    parser.add_argument("--reset", action="store_true", help="Delete existing demo rows before reseeding.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(run(reset=args.reset))
