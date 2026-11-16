import asyncio
import logging
import tempfile
import time
from pathlib import Path

from aiogram import Bot, Dispatcher, F, Router
from aiogram.filters import Command, CommandStart
from aiogram.types import BotCommand, Message

import config
import db
import llm
import stt
from rag import Index

log = logging.getLogger(__name__)
router = Router()
index = Index()

TELEGRAM_LIMIT = 4096  # максимум символов в одном сообщении Telegram
ERROR_TEXT = "⚠️ Что-то пошло не так, ошибка записана в лог."

START_TEXT = (
    "Здравствуйте! Я помощник по производству.\n\n"
    "Отправьте голосовое или текст о проблеме — я найду связанные регламенты "
    "и договорённости и подскажу следующий шаг.\n\n"
    "/docs — какие документы я знаю"
)


def split_message(text: str, limit: int = TELEGRAM_LIMIT) -> list[str]:
    """Режет длинный текст на части не длиннее limit, по возможности по переводу строки."""
    parts = []
    while len(text) > limit:
        cut = text.rfind("\n", 0, limit)
        if cut <= 0:
            cut = limit
        parts.append(text[:cut])
        text = text[cut:].lstrip("\n")
    if text:
        parts.append(text)
    return parts


async def send_long(message: Message, text: str) -> None:
    for part in split_message(text):
        await message.answer(part)


def user_id(message: Message) -> int:
    # from_user бывает пустым, например у сообщений от имени канала
    return message.from_user.id if message.from_user else 0


async def process(message: Message, kind: str, text: str, timings: dict) -> None:
    t = time.perf_counter()
    fragments = await index.search(text)
    timings["поиск"] = time.perf_counter() - t

    t = time.perf_counter()
    reply = await llm.answer(text, fragments)
    timings["ответ"] = time.perf_counter() - t

    sources = sorted({f["source"] for f in fragments})
    await db.save(user_id(message), kind, text, reply, sources, timings)

    timing_line = " · ".join(f"{k} {v:.1f} с" for k, v in timings.items())
    await send_long(message, f"{reply}\n\n📎 Источники: {', '.join(sources)}\n⏱ {timing_line}")


@router.message(CommandStart())
async def on_start(message: Message) -> None:
    await message.answer(START_TEXT)


@router.message(Command("docs"))
async def on_docs(message: Message) -> None:
    names = sorted({c["source"] for c in index.chunks})
    await send_long(message, "📚 Документы в базе:\n" + "\n".join(f"• {n}" for n in names))


@router.message(Command("reindex"))
async def on_reindex(message: Message) -> None:
    if user_id(message) not in config.ADMIN_IDS:
        return
    await message.answer("🔄 Перестраиваю индекс…")
    t = time.perf_counter()
    try:
        n = await index.build()
    except Exception:
        log.exception("Ошибка перестройки индекса")
        await message.answer("⚠️ Индекс не перестроен: проверьте, что Ollama запущена и в docs/ есть документы. "
                             "Подробности в логе.")
        return
    await message.answer(f"✅ Готово: {n} фрагментов за {time.perf_counter() - t:.1f} с")


@router.message(F.voice)
async def on_voice(message: Message, bot: Bot) -> None:
    status = await message.answer("🎧 Расшифровываю голосовое…")
    timings: dict[str, float] = {}
    try:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "voice.ogg"
            await bot.download(message.voice, destination=path)
            t = time.perf_counter()
            text = await stt.transcribe(str(path))
            timings["расшифровка"] = time.perf_counter() - t

        if not text:
            await status.edit_text("Не получилось разобрать голосовое, попробуйте ещё раз.")
            return

        shown = text if len(text) <= 3000 else text[:3000] + "…"
        await status.edit_text(f"📝 Расшифровка:\n{shown}\n\n🔎 Ищу в документах…")
        await process(message, "voice", text, timings)
    except Exception:
        log.exception("Ошибка обработки голосового")
        await status.edit_text(ERROR_TEXT)


@router.message(F.text & ~F.text.startswith("/"))
async def on_text(message: Message) -> None:
    status = await message.answer("🔎 Ищу в документах…")
    try:
        await process(message, "text", message.text, {})
    except Exception:
        log.exception("Ошибка обработки текста")
        await status.edit_text(ERROR_TEXT)
        return
    await status.delete()


async def main() -> None:
    logging.basicConfig(level=logging.INFO)
    if not config.BOT_TOKEN:
        raise SystemExit("Не задан BOT_TOKEN в .env (для проверки без Telegram есть demo.py)")
    await db.init()
    if not index.load():
        await index.build()
    log.info("Загружаю модель whisper…")
    await asyncio.to_thread(stt.load_model)

    bot = Bot(config.BOT_TOKEN)
    await bot.set_my_commands([
        BotCommand(command="start", description="Начало"),
        BotCommand(command="docs", description="Документы в базе"),
    ])
    dp = Dispatcher()
    dp.include_router(router)
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
