"""Журнал всех обращений: что спросили, что ответили, источники и время этапов."""
import json

import aiosqlite

from config import DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS journal (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    user_id INTEGER,
    kind TEXT,          -- voice / text
    question TEXT,
    answer TEXT,
    sources TEXT,       -- JSON-список файлов
    timings TEXT        -- JSON: секунды по этапам
)
"""


async def init() -> None:
    async with aiosqlite.connect(DB_PATH) as conn:
        await conn.execute(SCHEMA)
        await conn.commit()


async def save(user_id: int, kind: str, question: str, answer: str,
               sources: list[str], timings: dict) -> None:
    async with aiosqlite.connect(DB_PATH) as conn:
        await conn.execute(
            "INSERT INTO journal (user_id, kind, question, answer, sources, timings) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (user_id, kind, question, answer,
             json.dumps(sources, ensure_ascii=False),
             json.dumps({k: round(v, 2) for k, v in timings.items()}, ensure_ascii=False)),
        )
        await conn.commit()
