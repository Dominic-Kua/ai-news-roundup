import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest


@pytest.mark.asyncio
async def test_score_article_returns_number():
    import llm

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {"response": "8"}

    with patch("httpx.AsyncClient") as mock_client:
        instance = AsyncMock()
        instance.post = AsyncMock(return_value=mock_response)
        instance.__aenter__ = AsyncMock(return_value=instance)
        instance.__aexit__ = AsyncMock(return_value=False)
        mock_client.return_value = instance

        score = await llm.score_article(
            {"title": "GPT-5 Released", "source": "openai.com", "summary": "New model"},
            [],
        )

    assert score == 8


@pytest.mark.asyncio
async def test_score_article_clamps_to_10():
    import llm

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {"response": "15"}

    with patch("httpx.AsyncClient") as mock_client:
        instance = AsyncMock()
        instance.post = AsyncMock(return_value=mock_response)
        instance.__aenter__ = AsyncMock(return_value=instance)
        instance.__aexit__ = AsyncMock(return_value=False)
        mock_client.return_value = instance

        score = await llm.score_article({"title": "X", "source": "x", "summary": "Y"}, [])

    assert score == 10


@pytest.mark.asyncio
async def test_score_article_fallback_on_error():
    import llm

    with patch("httpx.AsyncClient") as mock_client:
        instance = AsyncMock()
        instance.post = AsyncMock(side_effect=httpx.ConnectError("refused"))
        instance.__aenter__ = AsyncMock(return_value=instance)
        instance.__aexit__ = AsyncMock(return_value=False)
        mock_client.return_value = instance

        score = await llm.score_article({"title": "X", "source": "x", "summary": "Y"}, [])

    assert score == 5  # default fallback


@pytest.mark.asyncio
async def test_curate_articles_scores_and_sorts():
    import llm

    scores = {"Article A": 9, "Article B": 3, "Article C": 7}

    async def fake_score(article, prefs):
        return scores.get(article["title"], 5)

    with patch("llm.score_article", side_effect=fake_score):
        candidates = [
            {"url": "https://a.com/1", "title": "Article A", "source": "a.com", "summary": "S1"},
            {"url": "https://a.com/2", "title": "Article B", "source": "a.com", "summary": "S2"},
            {"url": "https://a.com/3", "title": "Article C", "source": "a.com", "summary": "S3"},
        ]
        result = await llm.curate_articles(candidates, [])

    assert len(result) == 3
    assert result[0]["title"] == "Article A"
    assert result[0]["score"] == 9
    assert result[2]["title"] == "Article B"
    assert result[2]["score"] == 3


@pytest.mark.asyncio
async def test_curate_articles_returns_top_7():
    import llm

    async def fake_score(article, prefs):
        return hash(article["title"]) % 10 + 1

    candidates = [
        {"url": f"https://a.com/{i}", "title": f"Article {i}", "source": "a.com", "summary": f"S{i}"}
        for i in range(20)
    ]

    with patch("llm.score_article", side_effect=fake_score):
        result = await llm.curate_articles(candidates, [])

    assert len(result) == 7


@pytest.mark.asyncio
async def test_curate_articles_includes_preferences():
    import llm

    pref_topics = []

    async def fake_score(article, prefs):
        if prefs:
            pref_topics.extend([p.get("topic_fingerprint", "") for p in prefs])
        return 7

    preferences = [{"liked": True, "topic_fingerprint": "AI research", "title": "", "source": "", "created_at": ""}]
    candidates = [{"url": "https://a.com/1", "title": "T1", "source": "a.com", "summary": "S1"}]

    with patch("llm.score_article", side_effect=fake_score):
        result = await llm.curate_articles(candidates, preferences)

    assert len(result) == 1
    assert "AI research" in pref_topics


@pytest.mark.asyncio
async def test_generate_search_queries():
    import llm

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {
        "response": json.dumps(["GPT-5 release date", "Claude 4 update", "AI regulation 2026"])
    }

    with patch("httpx.AsyncClient") as mock_client:
        instance = AsyncMock()
        instance.post = AsyncMock(return_value=mock_response)
        instance.__aenter__ = AsyncMock(return_value=instance)
        instance.__aexit__ = AsyncMock(return_value=False)
        mock_client.return_value = instance

        queries = await llm.generate_search_queries([])

    assert len(queries) == 3
    assert "GPT-5 release date" in queries


@pytest.mark.asyncio
async def test_generate_search_queries_fallback_on_error():
    import llm

    with patch("httpx.AsyncClient") as mock_client:
        instance = AsyncMock()
        instance.post = AsyncMock(side_effect=httpx.ConnectError("refused"))
        instance.__aenter__ = AsyncMock(return_value=instance)
        instance.__aexit__ = AsyncMock(return_value=False)
        mock_client.return_value = instance

        queries = await llm.generate_search_queries([])

    assert len(queries) == 7  # fallback queries


@pytest.mark.asyncio
async def test_check_ollama_returns_true():
    import llm

    mock_response = MagicMock()
    mock_response.status_code = 200

    with patch("httpx.AsyncClient") as mock_client:
        instance = AsyncMock()
        instance.get = AsyncMock(return_value=mock_response)
        instance.__aenter__ = AsyncMock(return_value=instance)
        instance.__aexit__ = AsyncMock(return_value=False)
        mock_client.return_value = instance

        result = await llm.check_ollama()

    assert result is True


@pytest.mark.asyncio
async def test_check_ollama_returns_false_on_error():
    import llm

    with patch("httpx.AsyncClient") as mock_client:
        instance = AsyncMock()
        instance.get = AsyncMock(side_effect=httpx.ConnectError("refused"))
        instance.__aenter__ = AsyncMock(return_value=instance)
        instance.__aexit__ = AsyncMock(return_value=False)
        mock_client.return_value = instance

        result = await llm.check_ollama()

    assert result is False


@pytest.mark.asyncio
async def test_generate_fingerprint():
    import llm

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {"response": "autonomous driving research"}

    with patch("httpx.AsyncClient") as mock_client:
        instance = AsyncMock()
        instance.post = AsyncMock(return_value=mock_response)
        instance.__aenter__ = AsyncMock(return_value=instance)
        instance.__aexit__ = AsyncMock(return_value=False)
        mock_client.return_value = instance

        result = await llm.generate_fingerprint({"title": "Tesla FSD Update", "source": "techcrunch.com", "summary": "New self-driving features"})

    assert result == "autonomous driving research"


@pytest.mark.asyncio
async def test_generate_fingerprint_fallback_on_error():
    import llm

    with patch("httpx.AsyncClient") as mock_client:
        instance = AsyncMock()
        instance.post = AsyncMock(side_effect=httpx.ConnectError("refused"))
        instance.__aenter__ = AsyncMock(return_value=instance)
        instance.__aexit__ = AsyncMock(return_value=False)
        mock_client.return_value = instance

        result = await llm.generate_fingerprint({"title": "Test Article", "source": "x", "summary": "y"})

    assert result == "Test Article"
