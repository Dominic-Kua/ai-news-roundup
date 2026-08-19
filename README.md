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
- [Ollama](https://ollama.com) running locally with `llama3.1:8b` pulled:
  ```bash
  ollama pull llama3.1:8b
  ```

## Setup

```bash
# Install dependencies
uv sync

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

## Running tests

```bash
uv run python -m pytest -v
```
