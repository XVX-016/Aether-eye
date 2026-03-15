from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from typing import Any

import feedparser
import httpx
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

try:
    from app.database.models import IntelArticle
    from pipeline.site_registry import load_sites
except ModuleNotFoundError:
    from backend.app.database.models import IntelArticle
    from backend.pipeline.site_registry import load_sites

logger = logging.getLogger(__name__)


RSS_SOURCES = [
    {"url": "https://feeds.reuters.com/reuters/worldNews", "source": "Reuters", "tier": 1},
    {"url": "https://feeds.bbci.co.uk/news/world/rss.xml", "source": "BBC News", "tier": 1},
    {"url": "https://breakingdefense.com/feed", "source": "Breaking Defense", "tier": 2},
    {"url": "https://www.defensenews.com/arc/outboundfeeds/rss/", "source": "Defense News", "tier": 2},
    {"url": "https://www.aljazeera.com/xml/rss/all.xml", "source": "Al Jazeera", "tier": 2},
    {"url": "https://www.arabnews.com/rss.xml", "source": "Arab News", "tier": 2},
    {"url": "https://www.thenationalnews.com/rss", "source": "The National", "tier": 2},
    {"url": "https://rss.app/feeds/tvYWGGqHBSGMYBKN.xml", "source": "War Zone", "tier": 3},
]

ALIASES: dict[str, list[str]] = {
    "al_udeid": ["al udeid", "centcom", "qatar air", "usaf qatar", "55th wing", "379th", "air operations center"],
    "al_dhafra": ["al dhafra", "uae air force", "abu dhabi air", "khalifa city", "barak missile"],
    "strait_hormuz_north": ["bandar abbas", "hormuz", "persian gulf blockade", "iriaf", "iranian navy", "irgc navy"],
    "kadena": ["kadena", "okinawa base", "18th wing", "usaf japan", "ryukyu"],
    "andersen_guam": ["andersen", "guam bomber", "usaf guam", "pacific strike", "b-52 pacific", "b-2 guam"],
    "ramstein": ["ramstein", "usafe", "air force europe", "kaiserslautern", "86th airlift"],
    "incirlik": ["incirlik", "turkey air base", "usaf turkey", "39th air base wing", "nato turkey"],
    "diego_garcia": ["diego garcia", "biot", "indian ocean base", "british indian ocean", "b-2 diego"],
    "al_asad": ["al asad", "al-asad", "anbar province base", "iraq air base", "ain al-asad"],
    "norfolk_naval": ["norfolk naval", "naval station norfolk", "carrier strike group norfolk", "2nd fleet"],
    "pearl_harbor": ["pearl harbor", "joint base pearl", "pacific fleet", "indopacom"],
    "changi_naval": ["changi naval", "singapore navy", "rsan changi", "strait of malacca base"],
    "rota_naval": ["naval station rota", "rota spain", "6th fleet", "destroyer squadron rota"],
    "aden_port": ["port of aden", "aden gulf", "houthi attack", "red sea shipping", "bab el-mandeb", "houthi drone", "houthi missile"],
    "jeddah_port": ["jeddah port", "saudi port", "red sea saudi"],
    "dubai_airport": ["dubai airport", "dxb", "emirates airline", "dubai aviation"],
    "bagram": ["bagram", "afghanistan air base", "parwan province", "kabul air"],
}

COUNTRY_MILITARY: dict[str, list[str]] = {
    "qatar": ["al_udeid"],
    "uae": ["al_dhafra", "dubai_airport"],
    "iran": ["strait_hormuz_north"],
    "iraq": ["al_asad"],
    "turkey": ["incirlik"],
    "japan": ["kadena"],
    "germany": ["ramstein"],
    "singapore": ["changi_naval"],
    "yemen": ["aden_port"],
}

MILITARY_TERMS = ["military", "air force", "navy", "troops", "strike", "missile", "drone", "exercise", "deployment", "base"]


def geo_tag_article(text: str) -> str | None:
    text_lower = text.lower()
    sites = load_sites()
    scores: dict[str, int] = {}

    for site in sites:
        for tag in site.get("tags", []):
            if str(tag).lower() in text_lower:
                scores[str(site["id"])] = scores.get(str(site["id"]), 0) + 2

    for site_id, aliases in ALIASES.items():
        for alias in aliases:
            if alias in text_lower:
                scores[site_id] = scores.get(site_id, 0) + 1

    if any(term in text_lower for term in MILITARY_TERMS):
        for country, site_ids in COUNTRY_MILITARY.items():
            if country in text_lower:
                for site_id in site_ids:
                    scores[site_id] = scores.get(site_id, 0) + 1

    if not scores:
        return None
    return max(scores, key=lambda site_id: scores[site_id])


def _coerce_published(entry: Any) -> datetime | None:
    for key in ("published", "updated", "created"):
        value = entry.get(key)
        if not value:
            continue
        try:
            parsed = parsedate_to_datetime(value)
        except (TypeError, ValueError, IndexError):
            continue
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    return None


async def fetch_and_store_articles(db: AsyncSession) -> int:
    total = 0

    async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
        for source in RSS_SOURCES:
            try:
                response = await client.get(source["url"])
                response.raise_for_status()
                feed = feedparser.parse(response.content)
                source_count = 0

                for entry in feed.entries:
                    title = str(entry.get("title") or "").strip()
                    url = str(entry.get("link") or "").strip()
                    summary = str(entry.get("summary") or entry.get("description") or "").strip()
                    if not title or not url:
                        continue

                    text = f"{title} {summary}".strip()
                    site_id = geo_tag_article(text)
                    published = entry.get("published_parsed")
                    pub_dt = (
                        datetime(*published[:6], tzinfo=timezone.utc)
                        if published
                        else _coerce_published(entry) or datetime.now(timezone.utc)
                    )

                    try:
                        result = await db.execute(
                            insert(IntelArticle)
                            .values(
                                site_id=site_id,
                                title=title[:500],
                                url=url,
                                source=source["source"],
                                source_tier=source["tier"],
                                published_at=pub_dt,
                                summary=summary[:4000] or None,
                            )
                            .on_conflict_do_nothing(index_elements=["url"])
                            .returning(IntelArticle.id)
                        )
                        if result.scalar_one_or_none() is not None:
                            source_count += 1
                    except Exception:
                        continue

                await db.commit()
                total += source_count
                logger.info("Intel feed %s: %d articles", source["source"], source_count)
            except Exception as exc:
                logger.warning("Intel feed failed [%s] %s: %s", source["source"], source["url"], exc)
                continue

    return total


async def get_articles_for_site(
    db: AsyncSession,
    site_id: str,
    hours: int = 48,
) -> list[dict[str, Any]]:
    since = datetime.now(timezone.utc) - timedelta(hours=hours)
    rows = (
        await db.execute(
            select(IntelArticle)
            .where(IntelArticle.site_id == site_id)
            .where(IntelArticle.fetched_at >= since)
            .order_by(IntelArticle.published_at.desc().nullslast(), IntelArticle.id.desc())
            .limit(20)
        )
    ).scalars().all()

    return [
        {
            "title": row.title,
            "url": row.url,
            "source": row.source,
            "source_tier": row.source_tier,
            "published_at": row.published_at.isoformat() if row.published_at else None,
        }
        for row in rows
    ]


async def get_global_articles(
    db: AsyncSession,
    hours: int = 48,
    limit: int = 20,
) -> list[dict[str, Any]]:
    since = datetime.now(timezone.utc) - timedelta(hours=hours)
    rows = (
        await db.execute(
            select(IntelArticle)
            .where(IntelArticle.site_id.is_(None))
            .where(IntelArticle.fetched_at >= since)
            .order_by(IntelArticle.published_at.desc().nullslast(), IntelArticle.id.desc())
            .limit(limit)
        )
    ).scalars().all()

    return [
        {
            "title": row.title,
            "url": row.url,
            "source": row.source,
            "source_tier": row.source_tier,
            "published_at": row.published_at.isoformat() if row.published_at else None,
            "site_id": None,
        }
        for row in rows
    ]


async def retag_existing_articles(db: AsyncSession) -> int:
    articles = (await db.execute(select(IntelArticle))).scalars().all()
    retagged = 0
    for article in articles:
        new_site_id = geo_tag_article(f"{article.title} {article.summary or ''}")
        if new_site_id != article.site_id:
            article.site_id = new_site_id
            retagged += 1
    await db.commit()
    return retagged
