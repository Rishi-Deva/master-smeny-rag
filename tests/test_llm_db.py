import json
import sqlite3

import httpx

import config
import db
import llm

FRAGMENTS = [
    {"source": "reglament_brak.md", "text": "Сообщить мастеру за 10 минут.", "score": 0.9},
    {"source": "peredacha_smeny.md", "text": "Лист передачи смены.", "score": 0.4},
]


async def test_answer_sends_context_with_sources(monkeypatch):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return httpx.Response(200, json={"message": {"content": "  Ответ модели \n"}})

    real_client = httpx.AsyncClient
    monkeypatch.setattr(llm.httpx, "AsyncClient",
                        lambda **kw: real_client(transport=httpx.MockTransport(handler), **kw))

    reply = await llm.answer("Опять брак", FRAGMENTS)

    assert reply == "Ответ модели"
    assert seen["model"] == config.LLM_MODEL
    assert seen["stream"] is False
    system, user = seen["messages"]
    assert system["role"] == "system" and "ничего не выдумывай" in system["content"]
    assert "[reglament_brak.md]" in user["content"] and "Опять брак" in user["content"]


async def test_demo_mode_does_not_call_model(demo_mode, monkeypatch):
    monkeypatch.setattr(llm.httpx, "AsyncClient", None)  # любой вызов сети упадёт
    reply = await llm.answer("вопрос", FRAGMENTS)
    assert "демо-режим" in reply and "[reglament_brak.md]" in reply
    assert "не найдено" in llm.demo_answer([])


async def test_journal_saves_json_fields(tmp_db):
    await db.init()
    await db.save(42, "voice", "вопрос", "ответ", ["a.md", "б.md"],
                  {"расшифровка": 1.234, "поиск": 0.056})
    row = sqlite3.connect(tmp_db).execute(
        "SELECT user_id, kind, question, answer, sources, timings FROM journal").fetchone()
    assert row[:4] == (42, "voice", "вопрос", "ответ")
    assert json.loads(row[4]) == ["a.md", "б.md"]
    assert json.loads(row[5]) == {"расшифровка": 1.23, "поиск": 0.06}
