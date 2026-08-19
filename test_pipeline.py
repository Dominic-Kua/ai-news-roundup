import asyncio
import os
from datetime import UTC, date, datetime, timedelta
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio


@pytest_asyncio.fixture(autouse=True)
def setup_test_db(tmp_path):
    os.environ["DB_PATH"] = str(tmp_path / "test.db")
    yield
    os.environ.pop("DB_PATH", None)


def test_is_recent_with_valid_date():
    from pipeline import _is_recent

    now = datetime.now(UTC)
    recent = (now - timedelta(hours=24)).strftime("%Y-%m-%d")
    assert _is_recent(recent) is True

    old = (now - timedelta(hours=72)).strftime("%Y-%m-%d")
    assert _is_recent(old) is False


def test_is_recent_with_none_date():
    from pipeline import _is_recent

    assert _is_recent(None) is True


def test_is_recent_with_unparseable_date():
    from pipeline import _is_recent

    assert _is_recent("not-a-date") is True


@pytest.mark.asyncio
async def test_pipeline_stores_edition():
    import db
    import pipeline

    await db.init_db()

    curated = [
        {"url": f"https://example.com/2026/08/18/article-{i}", "title": f"AI News {i}", "source": "example.com", "summary": f"Summary {i}", "publish_date": None}
        for i in range(1, 6)
    ]

    with (
        patch.object(pipeline, "_search_ddgs", new_callable=AsyncMock, return_value=curated),
        patch("llm.generate_search_queries", new_callable=AsyncMock, return_value=["test query"]),
        patch("llm.curate_articles", new_callable=AsyncMock, return_value=curated),
    ):
        result = await pipeline.run_pipeline()

    assert result["status"] == "success"
    assert result["article_count"] == 5

    edition = await db.get_edition(datetime.now(UTC).date())
    assert edition is not None
    assert len(edition) == 5


@pytest.mark.asyncio
async def test_pipeline_rejects_concurrent_run():
    import pipeline

    pipeline._lock = asyncio.Lock()
    await pipeline._lock.acquire()

    result = await pipeline.run_pipeline()
    assert result["status"] == "already_running"

    pipeline._lock.release()


@pytest.mark.asyncio
async def test_pipeline_handles_no_candidates():
    import db
    import pipeline

    await db.init_db()

    with (
        patch.object(pipeline, "_search_ddgs", new_callable=AsyncMock, return_value=[]),
        patch("llm.generate_search_queries", new_callable=AsyncMock, return_value=["test query"]),
    ):
        result = await pipeline.run_pipeline()

    assert result["status"] == "no_candidates"


@pytest.mark.asyncio
async def test_pipeline_deduplicates_against_existing():
    import db
    import pipeline

    await db.init_db()

    # Pre-store an article
    existing = [{"url": "https://example.com/2026/08/17/old-article", "title": "Old", "source": "x", "summary": "S", "publish_date": None}]
    await db.store_edition(date(2026, 8, 17), existing)

    # Pipeline should skip the duplicate
    candidates = [
        {"url": "https://example.com/2026/08/17/old-article", "title": "Old", "source": "x", "summary": "S", "publish_date": None},
        {"url": "https://example.com/2026/08/18/new-article", "title": "New", "source": "x", "summary": "S", "publish_date": None},
    ]

    curated = [candidates[1]]

    with (
        patch.object(pipeline, "_search_ddgs", new_callable=AsyncMock, return_value=candidates),
        patch("llm.generate_search_queries", new_callable=AsyncMock, return_value=["test query"]),
        patch("llm.curate_articles", new_callable=AsyncMock, return_value=curated),
    ):
        result = await pipeline.run_pipeline()

    assert result["status"] == "success"
    assert result["article_count"] == 1
