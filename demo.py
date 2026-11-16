"""Прогон того же потока, что и в боте, но из консоли и без Telegram.

    python demo.py "На второй линии опять брак по толщине листа, что делать?"
    DEMO_MODE=1 python demo.py "..."     # без Ollama: заглушки вместо моделей

Голос здесь не расшифровывается: на вход сразу идёт текст.
"""
import asyncio
import sys
import time

import config
import db
import llm
from rag import Index

DEFAULT_QUESTION = "На второй линии опять брак по толщине листа, третий раз за смену, что делать?"


async def run(question: str) -> str:
    await db.init()
    index = Index()
    timings: dict[str, float] = {}

    t = time.perf_counter()
    n = await index.build()
    timings["индекс"] = time.perf_counter() - t

    t = time.perf_counter()
    fragments = await index.search(question)
    timings["поиск"] = time.perf_counter() - t

    t = time.perf_counter()
    reply = await llm.answer(question, fragments)
    timings["ответ"] = time.perf_counter() - t

    sources = sorted({f["source"] for f in fragments})
    await db.save(0, "demo", question, reply, sources, timings)

    found = "\n".join(f"  {f['score']:.2f}  {f['source']}" for f in fragments)
    timing_line = " · ".join(f"{k} {v:.2f} с" for k, v in timings.items())
    mode = "DEMO_MODE (заглушки)" if config.DEMO_MODE else f"Ollama: {config.LLM_MODEL} + {config.EMBED_MODEL}"
    return (f"Режим: {mode}\nФрагментов в индексе: {n}\n"
            f"Вопрос: {question}\n\nНайдено:\n{found}\n\n{reply}\n\n"
            f"📎 Источники: {', '.join(sources)}\n⏱ {timing_line}")


if __name__ == "__main__":
    print(asyncio.run(run(" ".join(sys.argv[1:]) or DEFAULT_QUESTION)))
