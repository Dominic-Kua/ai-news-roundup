# AI News Roundup

Daily curated AI news briefing served on your local network. An LLM searches the web, scores articles for relevance, and serves the top 7 on a web page you can browse over coffee.

## How it works

1. **6:00 AM** — Pipeline runs automatically (or trigger manually via `POST /api/trigger`)
2. **Search** — 7 static search queries (no LLM call); up to 2 are derived from your liked topics via keyword extraction
3. **Filter** — Results are filtered for freshness (≤48h), deduplicated by URL and near-duplicate title, hub/homepage pages are removed, then heuristically pre-trimmed to 30 candidates so the LLM prompt stays small
4. **Score** — All candidates are ranked in a single batched LLM call (1 call per run) by Llama 3.1 8B via Ollama; on LLM failure a zero-LLM heuristic ranking is used
5. **Serve** — Top 7 articles appear at `http://localhost:8000`
6. **Learn** — Like/dislike buttons store the article title as the topic fingerprint (no LLM call) to bias future searches and ranking

## Prerequisites

- Python 3.11+
- [Ollama](https://ollama.com) running locally

## Minimum machine specs

The pipeline runs exactly 1 LLM call per day (single batched ranking; query generation and fallback ranking use zero-LLM heuristics).

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| RAM | 8 GB | 16 GB |
| Disk | 2 GB free | 5 GB free |
| CPU | 4 cores | 8+ cores |
| GPU | Not required | Any NVIDIA with 4+ VRAM (CUDA) |

Tested on macOS (M1 Pro, 16 GB). CPU is perfectly fine — the pipeline runs at 6:00 AM and finishes well before an 8:00 AM read.

## Changing the model

Any Ollama-compatible model works. Edit the `MODEL` constant in `llm.py`:

```python
MODEL = "llama3.1:8b"  # change this
```

Then pull the model:

```bash
ollama pull <model-name>
```

**Model tradeoffs:**

| Model | RAM | Quality |
|-------|-----|---------|
| `tinyllama` | 1 GB | Basic — misses nuance, good for testing |
| `llama3.1:8b` | 5 GB | Good — understands context and preferences |
| `llama3.1:70b` | 40 GB | Excellent — overkill for this use case |
| `phi3` | 2 GB | Decent — good balance of speed and quality |
| `gemma2:9b` | 6 GB | Good — strong at instruction following |

The model handles one task: batched ranking of up to 30 candidates 1-10 in a single call. Better instruction following still helps, so `llama3.1:8b` is the sweet spot for most setups.

## Setup

```bash
# Install dependencies
uv sync

# Pull the default model
ollama pull llama3.1:8b

# Start the server
uv run python main.py
```

The server runs at `http://localhost:8000` by default. Configure with environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `HOST` | `0.0.0.0` | Bind address |
| `PORT` | `8000` | Listen port |
| `DB_PATH` | `./data/news.db` | SQLite database path |

## API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Today's briefing |
| `/archive/{date}` | GET | Browse past editions (e.g. `/archive/2026-08-18`) |
| `/api/trigger` | POST | Manually trigger the pipeline |
| `/api/preferences` | POST | Record like/dislike (`{"article_id": int, "liked": bool}`) |
| `/api/health` | GET | Scheduler status, last run, Ollama connectivity |

## Project structure

```
main.py          FastAPI app + scheduler
pipeline.py      Search → filter → score → store
llm.py           Ollama integration (batched ranking, heuristic fallback)
db.py            SQLite async layer
templates/       Jinja2 HTML templates
static/          CSS
data/            Database (gitignored)
```

## LLM prompts

The system uses one prompt, sent to Ollama with `stream: false`. When the user has liked/disliked articles, topic fingerprints are injected into the prompt. Search query generation and the fallback ranking use zero-LLM heuristics (no prompt).

**Batched article ranking** — runs once per pipeline, scores up to 30 candidates in a single call:

```
You are an AI news editor. Score each of the N articles below from 1-10
for relevance and novelty.

USER PREFERENCES:
User LIKED: <topic fingerprints>
User DISLIKED: <topic fingerprints>

RULES:
- 10 = groundbreaking research or major product launch
- 7-9 = significant development, novel angle
- 4-6 = routine industry news, funding rounds, minor updates
- 1-3 = rehashed old news, filler, off-topic, hub/homepage pages
- Penalize clickbait, duplicates, and off-topic items
- If user preferences are given, boost matching topics and penalize disliked ones

ARTICLES:
[0] Title: <title> | Source: <source> | Summary: <summary>
...

Return ONLY a JSON array like [{"i": 0, "score": 8}, {"i": 1, "score": 5}]
covering every index. No explanation, no markdown fences.
```

**Legacy single-article scorer** — `score_article()` is kept as a backwards-compat wrapper but is no longer used by the pipeline.

## Running tests

```bash
uv run python -m pytest -v
```
