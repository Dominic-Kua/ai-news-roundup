from __future__ import annotations

import os
from datetime import UTC, date, datetime, timedelta

import aiosqlite


def _get_db_path() -> str:
    return os.environ.get("DB_PATH", "./data/news.db")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS articles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    url TEXT UNIQUE NOT NULL,
    title TEXT NOT NULL,
    source TEXT NOT NULL,
    summary TEXT NOT NULL,
    publish_date TEXT,
    fetched_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS editions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TEXT UNIQUE NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS edition_articles (
    edition_id INTEGER NOT NULL REFERENCES editions(id),
    article_id INTEGER NOT NULL REFERENCES articles(id),
    PRIMARY KEY (edition_id, article_id)
);

CREATE TABLE IF NOT EXISTS preferences (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    article_id INTEGER NOT NULL REFERENCES articles(id),
    liked INTEGER NOT NULL,
    topic_fingerprint TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
"""


async def init_db() -> None:
    db_path = _get_db_path()
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    async with aiosqlite.connect(db_path) as db:
        await db.executescript(_SCHEMA)
        await db.commit()


async def store_edition(edition_date: date, articles: list[dict]) -> int:
    db_path = _get_db_path()
    async with aiosqlite.connect(db_path) as db:
        date_str = edition_date.isoformat()
        cursor = await db.execute(
            "INSERT OR IGNORE INTO editions (date) VALUES (?)",
            (date_str,),
        )
        if cursor.rowcount == 0:
            row = await db.execute("SELECT id FROM editions WHERE date = ?", (date_str,))
            edition_row = await row.fetchone()
            edition_id = edition_row[0]
        else:
            edition_id = cursor.lastrowid

        for article in articles:
            cursor = await db.execute(
                """INSERT OR IGNORE INTO articles (url, title, source, summary, publish_date)
                   VALUES (?, ?, ?, ?, ?)""",
                (
                    article["url"],
                    article["title"],
                    article["source"],
                    article["summary"],
                    article.get("publish_date"),
                ),
            )
            if cursor.rowcount > 0:
                article_id = cursor.lastrowid
            else:
                row = await db.execute("SELECT id FROM articles WHERE url = ?", (article["url"],))
                existing = await row.fetchone()
                article_id = existing[0]

            await db.execute(
                "INSERT OR IGNORE INTO edition_articles (edition_id, article_id) VALUES (?, ?)",
                (edition_id, article_id),
            )

        await db.commit()
        return edition_id


async def get_edition(edition_date: date) -> list[dict] | None:
    db_path = _get_db_path()
    async with aiosqlite.connect(db_path) as db:
        db.row_factory = aiosqlite.Row
        row = await db.execute("SELECT id FROM editions WHERE date = ?", (edition_date.isoformat(),))
        edition_row = await row.fetchone()
        if not edition_row:
            return None

        edition_id = edition_row[0]
        cursor = await db.execute(
            """SELECT a.id, a.url, a.title, a.source, a.summary, a.publish_date,
                      p.liked, p.topic_fingerprint
               FROM articles a
               JOIN edition_articles ea ON a.id = ea.article_id
               LEFT JOIN preferences p ON p.article_id = a.id
               WHERE ea.edition_id = ?
               ORDER BY a.publish_date DESC""",
            (edition_id,),
        )
        rows = await cursor.fetchall()
        articles = []
        for r in rows:
            articles.append(
                {
                    "id": r[0],
                    "url": r[1],
                    "title": r[2],
                    "source": r[3],
                    "summary": r[4],
                    "publish_date": r[5],
                    "liked": r[6],
                    "topic_fingerprint": r[7],
                }
            )
        return articles


async def get_article(article_id: int) -> dict | None:
    db_path = _get_db_path()
    async with aiosqlite.connect(db_path) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT id, url, title, source, summary, publish_date FROM articles WHERE id = ?",
            (article_id,),
        )
        row = await cursor.fetchone()
        if not row:
            return None
        return {
            "id": row[0],
            "url": row[1],
            "title": row[2],
            "source": row[3],
            "summary": row[4],
            "publish_date": row[5],
        }


async def get_all_article_urls() -> set[str]:
    db_path = _get_db_path()
    async with aiosqlite.connect(db_path) as db:
        cursor = await db.execute("SELECT url FROM articles")
        rows = await cursor.fetchall()
        return {r[0] for r in rows}


async def store_preference(article_id: int, liked: bool, fingerprint: str | None = None) -> None:
    db_path = _get_db_path()
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            "INSERT INTO preferences (article_id, liked, topic_fingerprint) VALUES (?, ?, ?)",
            (article_id, 1 if liked else 0, fingerprint),
        )
        await db.commit()


async def get_recent_preferences(days: int = 7) -> list[dict]:
    db_path = _get_db_path()
    cutoff = (datetime.now(UTC) - timedelta(days=days)).isoformat()
    async with aiosqlite.connect(db_path) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            """SELECT p.liked, p.topic_fingerprint, p.created_at, a.title, a.source
               FROM preferences p
               JOIN articles a ON p.article_id = a.id
               WHERE p.created_at >= ?
               ORDER BY p.created_at DESC""",
            (cutoff,),
        )
        rows = await cursor.fetchall()
        return [
            {
                "liked": bool(r[0]),
                "topic_fingerprint": r[1],
                "created_at": r[2],
                "title": r[3],
                "source": r[4],
            }
            for r in rows
        ]


async def get_last_edition_date() -> date | None:
    db_path = _get_db_path()
    async with aiosqlite.connect(db_path) as db:
        cursor = await db.execute("SELECT date FROM editions ORDER BY date DESC LIMIT 1")
        row = await cursor.fetchone()
        if row:
            return date.fromisoformat(row[0])
        return None
