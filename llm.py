"""Ответ локальной модели через Ollama на основе найденных фрагментов."""
import httpx

import config

SYSTEM = """Ты — помощник сотрудников производственного предприятия. Отвечай по-русски, коротко и по делу.
Опирайся только на фрагменты документов, которые тебе дали. Если ответа в них нет — прямо скажи об этом и ничего не выдумывай.
Формат ответа:
1. Суть проблемы — 1–2 предложения.
2. Что говорят документы — со ссылкой на файл в квадратных скобках.
3. Следующий шаг — что сделать и кого уведомить."""


def demo_answer(fragments: list[dict]) -> str:
    """Заглушка для DEMO_MODE: модель не вызывается, показываем лучший найденный фрагмент."""
    if not fragments:
        return "[демо-режим] Подходящих фрагментов не найдено."
    best = fragments[0]
    return (f"[демо-режим: модель не вызывалась]\n"
            f"Ближайший фрагмент [{best['source']}]:\n{best['text']}")


async def answer(question: str, fragments: list[dict]) -> str:
    if config.DEMO_MODE:
        return demo_answer(fragments)
    context = "\n\n".join(f"[{f['source']}]\n{f['text']}" for f in fragments)
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user",
         "content": f"Документы:\n{context}\n\nСообщение сотрудника:\n{question}"},
    ]
    async with httpx.AsyncClient(timeout=config.OLLAMA_TIMEOUT) as client:
        r = await client.post(
            f"{config.OLLAMA_URL}/api/chat",
            json={
                "model": config.LLM_MODEL,
                "messages": messages,
                "stream": False,
                "options": {"num_ctx": config.LLM_NUM_CTX, "temperature": 0.2},
            },
        )
        r.raise_for_status()
    return r.json()["message"]["content"].strip()
