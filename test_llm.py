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
async def test_curate_articles_single_call_scores_and_sorts():
    """Batched curation must use exactly ONE LLM call."""
    import llm

    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {
        "response": json.dumps([{"i": 0, "score": 9}, {"i": 1, "score": 3}, {"i": 2, "score": 7}])
    }

    with patch("httpx.AsyncClient") as mock_client:
        instance = AsyncMock()
        instance.post = AsyncMock(return_value=mock_response)
        instance.__aenter__ = AsyncMock(return_value=instance)
        instance.__aexit__ = AsyncMock(return_value=False)
        mock_client.return_value = instance

        candidates = [
            {"url": "https://a.com/1", "title": "Article A", "source": "a.com", "summary": "S1"},
            {"url": "https://a.com/2", "title": "Article B", "source": "a.com", "summary": "S2"},
            {"url": "https://a.com/3", "title": "Article C", "source": "a.com", "summary": "S3"},
        ]
        result = await llm.curate_articles(candidates, [])

        assert instance.post.call_count == 1

    assert len(result) == 3
    assert result[0]["title"] == "Article A"
    assert result[0]["score"] == 9
    assert result[2]["title"] == "Article B"
    assert result[2]["score"] == 3


@pytest.mark.asyncio
async def test_curate_articles_returns_top_7_single_call():
    import llm

    ranking = [{"i": i, "score": (i % 10) + 1} for i in range(20)]
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {"response": json.dumps(ranking)}

    with patch("httpx.AsyncClient") as mock_client:
        instance = AsyncMock()
        instance.post = AsyncMock(return_value=mock_response)
        instance.__aenter__ = AsyncMock(return_value=instance)
        instance.__aexit__ = AsyncMock(return_value=False)
        mock_client.return_value = instance

        candidates = [
            {"url": f"https://a.com/{i}", "title": f"Article {i}", "source": "a.com", "summary": f"S{i}"}
            for i in range(20)
        ]
        result = await llm.curate_articles(candidates, [])

        assert instance.post.call_count == 1

    assert len(result) == 7


@pytest.mark.asyncio
async def test_curate_articles_includes_preferences_in_single_prompt():
    import llm

    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {"response": json.dumps([{"i": 0, "score": 7}])}

    with patch("httpx.AsyncClient") as mock_client:
        instance = AsyncMock()
        instance.post = AsyncMock(return_value=mock_response)
        instance.__aenter__ = AsyncMock(return_value=instance)
        instance.__aexit__ = AsyncMock(return_value=False)
        mock_client.return_value = instance

        preferences = [{"liked": True, "topic_fingerprint": "AI research", "title": "", "source": "", "created_at": ""}]
        candidates = [{"url": "https://a.com/1", "title": "T1", "source": "a.com", "summary": "S1"}]
        result = await llm.curate_articles(candidates, preferences)

        assert instance.post.call_count == 1
        sent_prompt = instance.post.call_args[1]["json"]["prompt"]
        assert "AI research" in sent_prompt

    assert len(result) == 1


@pytest.mark.asyncio
async def test_curate_articles_falls_back_without_extra_calls():
    """LLM failure must fall back heuristically, not retry with more calls."""
    import llm

    with patch("httpx.AsyncClient") as mock_client:
        instance = AsyncMock()
        instance.post = AsyncMock(side_effect=httpx.ConnectError("refused"))
        instance.__aenter__ = AsyncMock(return_value=instance)
        instance.__aexit__ = AsyncMock(return_value=False)
        mock_client.return_value = instance

        candidates = [
            {"url": f"https://a.com/{i}", "title": f"GPT breakthrough {i}", "source": "a.com", "summary": "S"}
            for i in range(5)
        ]
        result = await llm.curate_articles(candidates, [])

        assert instance.post.call_count == 1  # one attempt, then zero-LLM fallback

    assert len(result) == 5


@pytest.mark.asyncio
async def test_generate_search_queries_zero_llm_calls():
    """Query generation must not touch the LLM at all."""
    import llm

    with patch("httpx.AsyncClient") as mock_client:
        queries = await llm.generate_search_queries([])
        mock_client.assert_not_called()

    assert len(queries) == 7


@pytest.mark.asyncio
async def test_generate_search_queries_uses_liked_topics_without_llm():
    import llm

    preferences = [
        {"liked": True, "topic_fingerprint": "quantum computing", "title": "Q", "source": "", "created_at": ""}
    ]
    with patch("httpx.AsyncClient") as mock_client:
        queries = await llm.generate_search_queries(preferences)
        mock_client.assert_not_called()

    assert len(queries) == 7
    assert any("quantum computing" in q for q in queries)


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
