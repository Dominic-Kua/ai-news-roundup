import os
from datetime import date
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient


@pytest_asyncio.fixture(autouse=True)
def setup_test_db(tmp_path):
    os.environ["DB_PATH"] = str(tmp_path / "test.db")
    yield
    os.environ.pop("DB_PATH", None)


@pytest_asyncio.fixture
async def client():
    import db
    from main import app

    await db.init_db()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.mark.asyncio
async def test_health_endpoint(client):
    resp = await client.get("/api/health")
    assert resp.status_code == 200
    data = resp.json()
    assert "scheduler_running" in data
    assert "last_run" in data
    assert "ollama_reachable" in data


@pytest.mark.asyncio
async def test_index_empty(client):
    resp = await client.get("/")
    assert resp.status_code == 200
    assert "No articles yet" in resp.text


@pytest.mark.asyncio
async def test_archive_empty(client):
    resp = await client.get("/archive/2026-08-18")
    assert resp.status_code == 200
    assert "No edition" in resp.text


@pytest.mark.asyncio
async def test_archive_invalid_date(client):
    resp = await client.get("/archive/not-a-date")
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_preference_requires_fields(client):
    resp = await client.post("/api/preferences", json={"article_id": 1})
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_preference_stores(client):
    import db

    articles = [{"url": "https://example.com/1", "title": "Test", "source": "x", "summary": "S", "publish_date": None}]
    await db.store_edition(date(2026, 8, 18), articles)
    edition = await db.get_edition(date(2026, 8, 18))
    article_id = edition[0]["id"]

    resp = await client.post("/api/preferences", json={"article_id": article_id, "liked": True})

    assert resp.status_code == 200
    data = resp.json()
    assert data["liked"] is True

    prefs = await db.get_recent_preferences()
    assert len(prefs) == 1
    assert prefs[0]["topic_fingerprint"] == "Test"


@pytest.mark.asyncio
async def test_trigger_returns_202(client):
    with patch("pipeline.run_pipeline", new_callable=AsyncMock, return_value={"status": "success"}):
        resp = await client.post("/api/trigger")
    assert resp.status_code == 202
    assert resp.json()["status"] == "accepted"
