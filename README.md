# AI News Roundup

Daily curated AI news briefing served on your local network. An LLM searches the web, scores articles for relevance, and serves the top 7 on a web page you can browse over coffee.

## How it works

1. **6:00 AM** — Pipeline runs automatically (or trigger manually via `POST /api/trigger`)
2. **Search** — LLM generates 7 search queries based on trending topics and your preferences
3. **Filter** — Results are filtered for freshness (≤48h), deduplicated, and hub/homepage pages are removed
4. **Score** — Each article is scored 1-10 by Llama 3.1 8B via Ollama
5. **Serve** — Top 7 articles appear at `http://localhost:8000`
6. **Learn** — Like/dislike buttons generate topic fingerprints that bias future searches

## Prerequisites

- Python 3.11+
- [Ollama](https://ollama.com) running locally

## Minimum machine specs

The pipeline runs ~14 LLM calls per day (1 query generation + 7 scoring + 6 fingerprinting). With `llama3.1:8b` on CPU, expect:

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| RAM | 8 GB | 16 GB |
| Disk | 2 GB free | 5 GB free |
| CPU | 4 cores | 8+ cores |
| GPU | Not required | Any NVIDIA with 4+ VRAM (CUDA) |

**Timings on CPU (Apple M2, 16 GB):** ~10s per scoring call, full pipeline ~5 minutes.
**Timings on GPU (NVIDIA, 4 GB VRAM):** ~2s per scoring call, full pipeline ~1 minute.

The pipeline runs at 6:00 AM and finishes well before an 8:00 AM read, so CPU is perfectly fine.

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

| Model | RAM | Speed (CPU) | Quality |
|-------|-----|-------------|---------|
| `tinyllama` | 1 GB | ~1s/call | Basic — misses nuance, good for testing |
| `llama3.1:8b` | 5 GB | ~10s/call | Good — understands context and preferences |
| `llama3.1:70b` | 40 GB | ~60s/call | Excellent — overkill for this use case |
| `phi3` | 2 GB | ~3s/call | Decent — good balance of speed and quality |
| `gemma2:9b` | 6 GB | ~12s/call | Good — strong at instruction following |

The model handles three tasks: generating search queries, scoring articles 1-10, and generating topic fingerprints. All three benefit from better instruction following, so `llama3.1:8b` is the sweet spot for most setups.

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
llm.py           Ollama integration (query gen, scoring, fingerprints)
db.py            SQLite async layer
templates/       Jinja2 HTML templates
static/          CSS
data/            Database (gitignored)
```

## LLM prompts

The system uses three prompts, all sent to Ollama with `stream: false`. When the user has liked/disliked articles, topic fingerprints are injected into each prompt.

**Search query generation** — runs once per pipeline, generates 7 DuckDuckGo queries:

```
You are an AI news editor planning today's search. Generate 7 diverse
DuckDuckGo search queries to find the most interesting AI news from
the last 24 hours.

USER PREFERENCES:
User LIKED: <topic fingerprints>
User DISLIKED: <topic fingerprints>

RULES:
- Each query should target SPECIFIC recent news, not general hub pages
- Focus on: new model releases, research breakthroughs, product launches,
  major funding, policy/regulation, open-source releases
- Avoid generic queries like "AI news" — be specific (e.g. "GPT-5 release
  date", "Anthropic Claude update 2026")
- If user liked certain topics, include queries for similar areas
- If user disliked certain topics, avoid those areas
- Queries should be short, 3-6 words each

Return ONLY a JSON array of 7 query strings. No explanation, no markdown fences.
```

**Article scoring** — runs once per candidate article, returns 1-10:

```
Score this AI news article from 1-10 for relevance and novelty.

Title: <title>
Source: <source>
Summary: <summary>

USER PREFERENCES:
User LIKED: <topic fingerprints>
User DISLIKED: <topic fingerprints>

RULES:
- 10 = groundbreaking research or major product launch
- 7-9 = significant development, novel angle
- 4-6 = routine industry news, funding rounds, minor updates
- 1-3 = rehashed old news, filler, off-topic, hub/homepage pages

Return ONLY the number. No explanation.
```

**Topic fingerprinting** — runs once per liked/disliked article, generates a 2-6 word label:

```
Generate a short topic fingerprint (2-6 words) for this news article:
Title: <title>
Source: <source>
Summary: <summary>

Return ONLY the fingerprint text, no quotes, no explanation.
```

## Running tests

```bash
uv run python -m pytest -v
```
