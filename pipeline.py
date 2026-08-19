from __future__ import annotations

import asyncio
import logging
import time
from datetime import UTC, date, datetime, timedelta
from urllib.parse import urlparse

from ddgs import DDGS
from ddgs.exceptions import DDGSException

import db
import llm

logger = logging.getLogger(__name__)

_lock = asyncio.Lock()
_last_run: datetime | None = None


def _is_recent(publish_date: str | None, hours: int = 48) -> bool:
    if not publish_date:
        return True
    try:
        for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%SZ"):
            try:
                dt = datetime.strptime(publish_date, fmt).replace(tzinfo=UTC)
                cutoff = datetime.now(UTC) - timedelta(hours=hours)
                return dt >= cutoff
            except ValueError:
                continue
        return True
    except (ValueError, TypeError):
        logger.debug("Date parsing failed for publish_date=%s", publish_date)
        return True


_HUB_PATHS = {
    "/", "/news", "/ai", "/technology", "/tech", "/latest",
    "/artificial-intelligence", "/machine-learning", "/business",
}

_HUB_SUFFIXES = (
    "/topics/", "/topic/", "/section/", "/category/",
    "/latest", "/index.html", "/news/", "/ai/",
)

# Known hub pages — exact path matches
_KNOWN_HUBS = {
    "www.reuters.com/technology/artificial-intelligence",
    "www.bbc.co.uk/news/topics/ce1qrvleleqt",
    "www.sciencedaily.com/news/computers_math/artificial_intelligence",
    "www.theguardian.com/technology/artificialintelligenceai",
    "www.artificialintelligence-news.com",
    "www.startuphub.ai/news",
}

_DOMAIN_HUBS = {
    "startuphub.ai",
    "artificialintelligence-news.com",
}


def _is_hub_page(url: str) -> bool:
    """Aggressively filter out hub/section/index pages — keep only real articles."""
    parsed = urlparse(url)
    domain = parsed.netloc.replace("www.", "").lower()
    path = parsed.path.rstrip("/").lower()

    # Known hub domains
    if domain in _DOMAIN_HUBS:
        return True

    # Known hub full paths
    check_path = f"{domain}{path}"
    if check_path in _KNOWN_HUBS:
        return True

    # Bare domain root (e.g. "www.reuters.com" with no path)
    if not path or path == "":
        return True

    # Common hub paths
    if path in _HUB_PATHS:
        return True

    # Section/topic/category paths
    if any(path.endswith(s) or path.startswith(s.rstrip("/")) for s in _HUB_SUFFIXES):
        return True

    # Paths that are just one segment with no article slug (e.g. "/ai", "/technology", "/news")
    segments = [s for s in path.split("/") if s]
    if len(segments) == 1 and not any(c.isdigit() for c in segments[0]):
        return True

    # If path has no date-like segment and is short, likely a hub
    has_date = any(
        part for part in segments
        if len(part) >= 4 and (part[:4].isdigit() or "-" in part)
    )
    return bool(not has_date and len(segments) <= 2)


async def _search_ddgs(queries: list[str]) -> list[dict]:
    results = []

    def _do_search():
        ddgs = DDGS()
        all_results = []
        for i, q in enumerate(queries):
            try:
                hits = ddgs.text(q, timelimit="d", max_results=20)
                if isinstance(hits, list):
                    all_results.extend(hits)
            except (OSError, ValueError, DDGSException) as e:
                logger.warning("DDGS search failed for %r: %s", q, e)
            # Brief pause between queries to avoid rate limiting
            if i < len(queries) - 1:
                time.sleep(2)
        return all_results

    raw = await asyncio.to_thread(_do_search)
    for r in raw:
        url = r.get("href", "")
        if not url:
            continue
        parsed = urlparse(url)
        source = parsed.netloc.replace("www.", "") if parsed.netloc else "unknown"
        results.append(
            {
                "url": url,
                "title": r.get("title", ""),
                "source": source,
                "summary": r.get("body", ""),
                "publish_date": None,
            }
        )
    return results


async def run_pipeline() -> dict:
    global _last_run

    if _lock.locked():
        logger.warning("Pipeline already running, rejecting trigger")
        return {"status": "already_running"}

    async with _lock:
        logger.info("Pipeline started at %s", datetime.now(UTC).isoformat())
        try:
            # Step 0: Ask LLM what to search for
            preferences = await db.get_recent_preferences(days=7)
            queries = await llm.generate_search_queries(preferences)
            logger.info("LLM generated search queries: %s", queries)

            # Step 1: Search
            logger.info("Running DDGS searches with %d queries", len(queries))
            candidates = await _search_ddgs(queries)
            logger.info("DDGS returned %d raw candidates", len(candidates))

            # Step 2: Freshness filter
            fresh = [c for c in candidates if _is_recent(c.get("publish_date"))]
            logger.info("After freshness filter: %d candidates", len(fresh))

            # Step 3: Dedup
            existing_urls = await db.get_all_article_urls()
            unique = [c for c in fresh if c["url"] not in existing_urls]
            logger.info("After dedup: %d candidates", len(unique))

            # Step 3.5: Filter hub pages
            articles = [c for c in unique if not _is_hub_page(c["url"])]
            logger.info("After hub filter: %d candidates", len(articles))

            if not articles:
                logger.warning("No valid candidates after filtering, skipping edition")
                _last_run = datetime.now(UTC)
                return {"status": "no_candidates"}

            # Step 4: LLM curation — score each article individually
            curated = await llm.curate_articles(articles, preferences)
            logger.info("LLM curated %d articles", len(curated))

            if not curated:
                curated = llm._fallback_by_date(articles)
                logger.warning("LLM returned nothing, using fallback: %d articles", len(curated))

            # Step 5: Store edition
            today = date.today()  # noqa: DTZ011
            edition_id = await db.store_edition(today, curated)
            logger.info("Stored edition %d for %s with %d articles", edition_id, today, len(curated))

            _last_run = datetime.now(UTC)
            return {"status": "success", "article_count": len(curated), "edition_date": today.isoformat()}

        except Exception as e:
            logger.exception("Pipeline failed")
            _last_run = datetime.now(UTC)
            return {"status": "error", "error": str(e)}


def get_last_run() -> datetime | None:
    return _last_run
