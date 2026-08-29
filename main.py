from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from datetime import date, timedelta

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import db
import llm
import pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", "8000"))

scheduler = AsyncIOScheduler()
templates = Jinja2Templates(directory="templates")


@asynccontextmanager
async def lifespan(app: FastAPI):
    await db.init_db()
    scheduler.add_job(pipeline.run_pipeline, CronTrigger(hour=6, minute=0), id="daily_pipeline")
    scheduler.start()
    logger.info("Scheduler started, daily pipeline at 06:00")
    yield
    scheduler.shutdown(wait=False)


app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    today = date.today()  # noqa: DTZ011
    articles = await db.get_edition(today)
    last_run = pipeline.get_last_run()
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "articles": articles,
            "edition_date": today.isoformat(),
            "last_run": last_run.isoformat() if last_run else None,
        },
    )


@app.get("/archive/{edition_date}", response_class=HTMLResponse)
async def archive(request: Request, edition_date: str):
    try:
        d = date.fromisoformat(edition_date)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format")

    articles = await db.get_edition(d)
    prev_date = (d - timedelta(days=1)).isoformat()
    next_date = (d + timedelta(days=1)).isoformat()
    return templates.TemplateResponse(
        request,
        "archive.html",
        {
            "articles": articles,
            "edition_date": d.isoformat(),
            "prev_date": prev_date,
            "next_date": next_date,
        },
    )


@app.post("/api/preferences")
async def record_preference(request: Request):
    body = await request.json()
    article_id = body.get("article_id")
    liked = body.get("liked")
    if article_id is None or liked is None:
        raise HTTPException(status_code=400, detail="article_id and liked required")

    article = await db.get_article(article_id)
    fingerprint = article["title"] if article else None
    await db.store_preference(article_id, liked, fingerprint)
    return {"status": "ok", "liked": liked}


@app.post("/api/trigger")
async def trigger_pipeline():
    if pipeline._lock.locked():
        raise HTTPException(status_code=409, detail="Pipeline already running")

    import asyncio
    asyncio.create_task(pipeline.run_pipeline())
    return JSONResponse(status_code=202, content={"status": "accepted", "message": "Pipeline triggered"})


@app.get("/api/health")
async def health():
    ollama_ok = await llm.check_ollama()
    last_run = pipeline.get_last_run()
    return {
        "scheduler_running": scheduler.running,
        "last_run": last_run.isoformat() if last_run else None,
        "ollama_reachable": ollama_ok,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT)
