import os
from datetime import date

import pytest
import pytest_asyncio

# Use a temp DB for all tests
_test_db = None


@pytest_asyncio.fixture(autouse=True)
def setup_test_db(tmp_path):
    global _test_db
    _test_db = str(tmp_path / "test.db")
    os.environ["DB_PATH"] = _test_db
    yield
    os.environ.pop("DB_PATH", None)


@pytest.mark.asyncio
async def test_init_db_creates_tables():
    import db
    await db.init_db()
    import aiosqlite

    async with aiosqlite.connect(_test_db) as conn:
        cursor = await conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in await cursor.fetchall()}
    assert "articles" in tables
    assert "editions" in tables
    assert "edition_articles" in tables
    assert "preferences" in tables


@pytest.mark.asyncio
async def test_store_and_get_edition():
    import db

    await db.init_db()
    articles = [
        {"url": "https://example.com/1", "title": "Article 1", "source": "example.com", "summary": "Sum 1", "publish_date": "2026-08-18"},
        {"url": "https://example.com/2", "title": "Article 2", "source": "example.com", "summary": "Sum 2", "publish_date": "2026-08-17"},
    ]
    edition_id = await db.store_edition(date(2026, 8, 18), articles)
    assert edition_id is not None

    result = await db.get_edition(date(2026, 8, 18))
    assert result is not None
    assert len(result) == 2
    assert result[0]["title"] == "Article 1"  # ordered by publish_date DESC, 2026-08-18 first
    assert result[1]["title"] == "Article 2"


@pytest.mark.asyncio
async def test_get_edition_returns_none_for_missing():
    import db

    await db.init_db()
    result = await db.get_edition(date(2020, 1, 1))
    assert result is None


@pytest.mark.asyncio
async def test_dedup_by_url():
    import db

    await db.init_db()
    articles = [{"url": "https://example.com/1", "title": "A", "source": "x", "summary": "S", "publish_date": None}]
    await db.store_edition(date(2026, 8, 18), articles)

    # Store same URL again — should not duplicate
    await db.store_edition(date(2026, 8, 19), articles)
    urls = await db.get_all_article_urls()
    assert len(urls) == 1


@pytest.mark.asyncio
async def test_store_and_get_preferences():
    import db

    await db.init_db()
    articles = [{"url": "https://example.com/1", "title": "A", "source": "x", "summary": "S", "publish_date": None}]
    await db.store_edition(date(2026, 8, 18), articles)
    edition = await db.get_edition(date(2026, 8, 18))
    article_id = edition[0]["id"]

    await db.store_preference(article_id, True, "AI research")
    prefs = await db.get_recent_preferences(days=7)
    assert len(prefs) == 1
    assert prefs[0]["liked"] is True
    assert prefs[0]["topic_fingerprint"] == "AI research"


@pytest.mark.asyncio
async def test_get_article():
    import db

    await db.init_db()
    articles = [{"url": "https://example.com/1", "title": "Test Title", "source": "test.com", "summary": "Test summary", "publish_date": "2026-08-18"}]
    await db.store_edition(date(2026, 8, 18), articles)
    edition = await db.get_edition(date(2026, 8, 18))
    article_id = edition[0]["id"]

    article = await db.get_article(article_id)
    assert article is not None
    assert article["title"] == "Test Title"
    assert article["source"] == "test.com"


@pytest.mark.asyncio
async def test_get_article_returns_none_for_missing():
    import db

    await db.init_db()
    article = await db.get_article(99999)
    assert article is None


@pytest.mark.asyncio
async def test_get_last_edition_date():
    import db

    await db.init_db()
    assert await db.get_last_edition_date() is None

    articles = [{"url": "https://example.com/1", "title": "A", "source": "x", "summary": "S", "publish_date": None}]
    await db.store_edition(date(2026, 8, 18), articles)
    await db.store_edition(date(2026, 8, 15), articles)

    last = await db.get_last_edition_date()
    assert last == date(2026, 8, 18)
