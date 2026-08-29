from __future__ import annotations

import json
import logging

import httpx

logger = logging.getLogger(__name__)

OLLAMA_BASE = "http://localhost:11434"
MODEL = "llama3.1:8b"
TIMEOUT = 300.0


async def generate_search_queries(preferences: list[dict]) -> list[str]:
    """Ask the LLM what AI news topics are worth searching for today."""
    preference_block = ""
    if preferences:
        liked = [p for p in preferences if p["liked"]]
        disliked = [p for p in preferences if not p["liked"]]
        parts = []
        if liked:
            topics = ", ".join(p["topic_fingerprint"] or p["title"] for p in liked[:10])
            parts.append(f"User LIKED: {topics}")
        if disliked:
            topics = ", ".join(p["topic_fingerprint"] or p["title"] for p in disliked[:10])
            parts.append(f"User DISLIKED: {topics}")
        preference_block = "\n".join(parts)

    nl = "\n"
    pref_section = f"\nUSER PREFERENCES:\n{preference_block}{nl}" if preference_block else ""

    prompt = f"""You are an AI news editor planning today's search. Generate 7 diverse DuckDuckGo search queries to find the most interesting AI news from the last 24 hours.

{pref_section}RULES:
- Each query should target SPECIFIC recent news, not general hub pages
- Focus on: new model releases, research breakthroughs, product launches, major funding, policy/regulation, open-source releases
- Avoid generic queries like "AI news" — be specific (e.g. "GPT-5 release date", "Anthropic Claude update 2026")
- If user liked certain topics, include queries for similar areas
- If user disliked certain topics, avoid those areas
- Queries should be short, 3-6 words each

Return ONLY a JSON array of 7 query strings. No explanation, no markdown fences."""

    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            resp = await client.post(
                f"{OLLAMA_BASE}/api/generate",
                json={"model": MODEL, "prompt": prompt, "stream": False},
            )
            resp.raise_for_status()
            text = resp.json().get("response", "").strip()
            # Strip markdown fences if present
            if text.startswith("```"):
                text = text.split("\n", 1)[1] if "\n" in text else text[3:]
            text = text.removesuffix("```").strip()
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return [str(q) for q in parsed[:7] if q]
            return []
    except (httpx.HTTPError, httpx.TimeoutException, json.JSONDecodeError, TypeError) as e:
        logger.warning("LLM query generation failed: %s, using fallback queries", e)
        return _fallback_queries()


def _fallback_queries() -> list[str]:
    return [
        "new AI model release",
        "machine learning breakthrough research",
        "AI startup funding announcement",
        "generative AI product launch",
        "artificial intelligence regulation",
        "open source AI model",
        "AI safety alignment research",
    ]


async def score_article(article: dict, preferences: list[dict]) -> int:
    """Score a single article 1-10 for relevance and novelty."""
    preference_block = ""
    if preferences:
        liked = [p for p in preferences if p["liked"]]
        disliked = [p for p in preferences if not p["liked"]]
        parts = []
        if liked:
            topics = ", ".join(p["topic_fingerprint"] or p["title"] for p in liked[:10])
            parts.append(f"User LIKED: {topics}")
        if disliked:
            topics = ", ".join(p["topic_fingerprint"] or p["title"] for p in disliked[:10])
            parts.append(f"User DISLIKED: {topics}")
        preference_block = "\n".join(parts)

    nl = "\n"
    pref_section = f"\nUSER PREFERENCES:\n{preference_block}{nl}" if preference_block else ""

    prompt = f"""Score this AI news article from 1-10 for relevance and novelty.

Title: {article.get('title', '')}
Source: {article.get('source', '')}
Summary: {article.get('summary', '')}
{pref_section}
RULES:
- 10 = groundbreaking research or major product launch
- 7-9 = significant development, novel angle
- 4-6 = routine industry news, funding rounds, minor updates
- 1-3 = rehashed old news, filler, off-topic, hub/homepage pages

Return ONLY the number. No explanation."""

    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            resp = await client.post(
                f"{OLLAMA_BASE}/api/generate",
                json={"model": MODEL, "prompt": prompt, "stream": False},
            )
            resp.raise_for_status()
            text = resp.json().get("response", "").strip()
            score = int("".join(c for c in text if c.isdigit()) or "5")
            return max(1, min(10, score))
    except (httpx.HTTPError, httpx.TimeoutException, ValueError, TypeError) as e:
        logger.warning("Scoring failed for %s: %s", article.get("title", "?")[:40], e)
        return 5


async def curate_articles(candidates: list[dict], preferences: list[dict]) -> list[dict]:
    """Score each candidate individually, return top 7."""
    scored = []
    for i, c in enumerate(candidates):
        score = await score_article(c, preferences)
        c["score"] = score
        scored.append(c)
        if (i + 1) % 10 == 0:
            logger.info("Scored %d/%d articles", i + 1, len(candidates))

    scored.sort(key=lambda x: x.get("score", 0), reverse=True)
    return scored[:7]


async def check_ollama() -> bool:
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{OLLAMA_BASE}/api/tags")
            return resp.status_code == 200
    except (httpx.HTTPError, httpx.TimeoutException):
        return False


def _fallback_by_date(candidates: list[dict]) -> list[dict]:
    def sort_key(c: dict) -> str:
        return c.get("publish_date") or "0000"

    sorted_cands = sorted(candidates, key=sort_key, reverse=True)
    return sorted_cands[:7]
