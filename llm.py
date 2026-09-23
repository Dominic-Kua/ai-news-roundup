from __future__ import annotations

import json
import logging
import re

import httpx

logger = logging.getLogger(__name__)

OLLAMA_BASE = "http://localhost:11434"
MODEL = "llama3.1:8b"
TIMEOUT = 300.0

# Hard cap on how many candidates are sent to the LLM in a single prompt.
# Keeps the prompt small, fast, and cheap. Pipeline pre-trims to this size;
# this is a defensive second cap.
MAX_LLM_CANDIDATES = 30
EDITION_SIZE = 7


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


def _preference_derived_queries(preferences: list[dict], limit: int = 2) -> list[str]:
    """Derive up to `limit` search queries from liked topics without any LLM call."""
    queries: list[str] = []
    liked = [p for p in preferences if p.get("liked")]
    for p in liked[:limit]:
        topic = (p.get("topic_fingerprint") or p.get("title") or "").strip()
        if not topic:
            continue
        # Keep it short: first 6 words max, strip punctuation noise.
        words = re.findall(r"[A-Za-z0-9][A-Za-z0-9\-+.#]*", topic)[:6]
        if words:
            queries.append(" ".join(words))
    return queries


async def generate_search_queries(preferences: list[dict]) -> list[str]:
    """Return search queries with ZERO LLM calls.

    Previously this cost 1 LLM call per pipeline run. Static queries +
    preference-derived keywords achieve the same coverage deterministically,
    so the LLM is skipped entirely here and reserved for the single
    batched curation call.
    """
    base = _fallback_queries()
    extra = _preference_derived_queries(preferences or [])
    # Prepend preference queries, keep total at 7, dedup case-insensitively.
    seen: set[str] = set()
    queries: list[str] = []
    for q in [*extra, *base]:
        key = q.lower()
        if key not in seen:
            seen.add(key)
            queries.append(q)
        if len(queries) >= 7:
            break
    return queries


def _preference_block(preferences: list[dict]) -> str:
    if not preferences:
        return ""
    liked = [p for p in preferences if p.get("liked")]
    disliked = [p for p in preferences if not p.get("liked")]

    def _topic(p: dict) -> str:
        return str(p.get("topic_fingerprint") or p.get("title") or "").strip()

    parts = []
    if liked:
        topics = ", ".join(t for t in (_topic(p) for p in liked[:10]) if t)
        if topics:
            parts.append(f"User LIKED: {topics}")
    if disliked:
        topics = ", ".join(t for t in (_topic(p) for p in disliked[:10]) if t)
        if topics:
            parts.append(f"User DISLIKED: {topics}")
    if not parts:
        return ""
    return "\nUSER PREFERENCES:\n" + "\n".join(parts) + "\n"


async def score_article(article: dict, preferences: list[dict]) -> int:
    """Legacy single-article scorer (1 LLM call). Kept for backwards compat.

    The pipeline no longer uses this — see :func:`curate_articles` for the
    single-call batched version. Kept so existing callers/tests keep working.
    """
    pref_section = _preference_block(preferences)

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


def _format_candidates_for_prompt(candidates: list[dict]) -> str:
    lines = []
    for i, c in enumerate(candidates):
        title = str(c.get("title", ""))[:200].replace("\n", " ")
        source = str(c.get("source", ""))[:80]
        summary = str(c.get("summary", ""))[:300].replace("\n", " ")
        lines.append(f"[{i}] Title: {title} | Source: {source} | Summary: {summary}")
    return "\n".join(lines)


def _parse_ranking_response(text: str, n: int) -> dict[int, int] | None:
    """Parse batched ranking response into {index: score}. Returns None on failure."""
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1] if "\n" in t else t[3:]
    t = t.removesuffix("```").strip()
    try:
        parsed = json.loads(t)
    except json.JSONDecodeError:
        # Fallback: extract all integers 0..n as ranked indices in order.
        nums = [int(x) for x in re.findall(r"\d+", t)]
        nums = [x for x in nums if 0 <= x < n]
        if not nums:
            return None
        # Treat as ranked order: first = highest score.
        return {idx: (n - rank) for rank, idx in enumerate(dict.fromkeys(nums))}
    scores: dict[int, int] = {}
    if isinstance(parsed, list):
        for rank, item in enumerate(parsed):
            if isinstance(item, dict):
                idx = item.get("i", item.get("index", item.get("id")))
                sc = item.get("score", item.get("rating"))
            elif isinstance(item, int):
                idx, sc = item, (n - rank)
            else:
                continue
            try:
                idx = int(idx)
                sc = int(sc)
            except (TypeError, ValueError):
                continue
            if 0 <= idx < n:
                scores[idx] = max(1, min(10, sc))
    return scores or None


async def curate_articles(candidates: list[dict], preferences: list[dict]) -> list[dict]:
    """Rank candidates and return top 7 using exactly ONE LLM call.

    Previously this made N sequential LLM calls (one per article). Now all
    candidates are sent in a single batched prompt. On LLM failure, falls
    back to a zero-LLM heuristic ranking (no extra calls).
    """
    if not candidates:
        return []

    shortlist = candidates[:MAX_LLM_CANDIDATES]
    pref_section = _preference_block(preferences or [])
    listing = _format_candidates_for_prompt(shortlist)

    prompt = f"""You are an AI news editor. Score each of the {len(shortlist)} articles below from 1-10 for relevance and novelty.
{pref_section}
RULES:
- 10 = groundbreaking research or major product launch
- 7-9 = significant development, novel angle
- 4-6 = routine industry news, funding rounds, minor updates
- 1-3 = rehashed old news, filler, off-topic, hub/homepage pages
- Penalize clickbait, duplicates, and off-topic items
- If user preferences are given, boost matching topics and penalize disliked ones

ARTICLES:
{listing}

Return ONLY a JSON array like [{{"i": 0, "score": 8}}, {{"i": 1, "score": 5}}] covering every index 0-{len(shortlist) - 1}. No explanation, no markdown fences."""

    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            resp = await client.post(
                f"{OLLAMA_BASE}/api/generate",
                json={"model": MODEL, "prompt": prompt, "stream": False},
            )
            resp.raise_for_status()
            text = resp.json().get("response", "").strip()
        scores = _parse_ranking_response(text, len(shortlist))
        if not scores:
            logger.warning("LLM ranking unparseable, using heuristic fallback")
            return heuristic_rank(candidates, preferences)[:EDITION_SIZE]
        for i, c in enumerate(shortlist):
            c["score"] = scores.get(i, 5)
        shortlist_sorted = sorted(shortlist, key=lambda x: x.get("score", 0), reverse=True)
        return shortlist_sorted[:EDITION_SIZE]
    except (httpx.HTTPError, httpx.TimeoutException, ValueError, TypeError, KeyError) as e:
        logger.warning("Batched curation failed (%s), using heuristic fallback", e)
        return heuristic_rank(candidates, preferences)[:EDITION_SIZE]


_HIGH_VALUE_KEYWORDS = frozenset(
    {
        "breakthrough", "release", "launch", "open-source", "open source",
        "frontier", "sota", "state-of-the-art", "benchmark", "paper",
        "regulation", "funding", "acquisition", "gpt", "claude", "gemini",
        "llama", "mistral", "deepseek", "agent", "robot", "chip",
    }
)

_LOW_VALUE_KEYWORDS = frozenset(
    {"sponsored", "webinar", "ebook", "coupon", "deal", "horoscope", "sports", "celebrity", "hub", "homepage"}
)


def heuristic_rank(candidates: list[dict], preferences: list[dict] | None = None) -> list[dict]:
    """Zero-LLM heuristic ranking. Used as pre-filter and LLM-failure fallback."""
    prefs = preferences or []
    liked_terms: set[str] = set()
    disliked_terms: set[str] = set()
    for p in prefs:
        topic = str(p.get("topic_fingerprint") or p.get("title") or "").lower()
        words = set(re.findall(r"[a-z0-9][a-z0-9\-+]*", topic))
        if p.get("liked"):
            liked_terms |= words
        else:
            disliked_terms |= words

    scored = []
    seen_sources: dict[str, int] = {}
    for c in candidates:
        text = f"{c.get('title', '')} {c.get('summary', '')}".lower()
        words = set(re.findall(r"[a-z0-9][a-z0-9\-+]*", text))
        score = 5
        score += sum(1 for k in _HIGH_VALUE_KEYWORDS if k in text)
        score -= sum(2 for k in _LOW_VALUE_KEYWORDS if k in text)
        score += len(words & liked_terms)
        score -= 2 * len(words & disliked_terms)
        # Source diversity bonus: prefer varied sources.
        src = str(c.get("source", ""))
        count = seen_sources.get(src, 0)
        score -= count  # penalize 2nd/3rd article from same source
        seen_sources[src] = count + 1
        # Longer summaries tend to be real articles, not stubs.
        if len(str(c.get("summary", ""))) > 120:
            score += 1
        c["score"] = max(1, min(10, score))
        scored.append(c)
    scored.sort(key=lambda x: x.get("score", 0), reverse=True)
    return scored


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
    return sorted_cands[:EDITION_SIZE]
