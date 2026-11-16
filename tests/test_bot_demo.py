import sqlite3
from types import SimpleNamespace

import bot
import config
import db
import demo
from rag import Index


class FakeMessage:
    def __init__(self, user_id=1, text=""):
        self.from_user = SimpleNamespace(id=user_id)
        self.text = text
        self.sent: list[str] = []

    async def answer(self, text):
        self.sent.append(text)
        return self


async def test_process_replies_with_sources_and_timings(demo_mode, tmp_db, tmp_index, monkeypatch):
    await db.init()
    index = Index()
    await index.build()
    monkeypatch.setattr(bot, "index", index)

    msg = FakeMessage(text="брак по толщине листа")
    await bot.process(msg, "text", msg.text, {"расшифровка": 0.5})

    reply = msg.sent[-1]
    assert "📎 Источники:" in reply
    assert "⏱ расшифровка 0.5 с · поиск" in reply and "ответ" in reply
    kind, sources = sqlite3.connect(tmp_db).execute("SELECT kind, sources FROM journal").fetchone()
    assert kind == "text" and "reglament_brak.md" in sources


async def test_reindex_only_for_admin(monkeypatch):
    built = []

    class FakeIndex:
        async def build(self):
            built.append(1)
            return 7

    monkeypatch.setattr(bot, "index", FakeIndex())
    monkeypatch.setattr(config, "ADMIN_IDS", {100})

    stranger = FakeMessage(user_id=5)
    await bot.on_reindex(stranger)
    assert stranger.sent == [] and built == []

    admin = FakeMessage(user_id=100)
    await bot.on_reindex(admin)
    assert built == [1] and "7 фрагментов" in admin.sent[-1]


async def test_demo_script_runs_end_to_end(demo_mode, tmp_db, tmp_index):
    out = await demo.run("Что делать, если нет паспорта качества на лист?")
    assert "Режим: DEMO_MODE" in out
    assert "usloviya_postavki_lista.md" in out.split("Найдено:")[1].splitlines()[1]
    rows = sqlite3.connect(tmp_db).execute("SELECT kind FROM journal").fetchall()
    assert rows == [("demo",)]


class FakeStatus:
    def __init__(self):
        self.edits: list[str] = []
        self.deleted = False

    async def edit_text(self, text):
        self.edits.append(text)

    async def delete(self):
        self.deleted = True


class StatusMessage(FakeMessage):
    """Первый ответ — «статус», который бот потом редактирует или удаляет."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.status = FakeStatus()

    async def answer(self, text):
        self.sent.append(text)
        return self.status


def test_split_message_respects_telegram_limit():
    text = "\n".join("строка %d " % i + "x" * 90 for i in range(100))
    parts = bot.split_message(text, limit=1000)
    assert all(len(p) <= 1000 for p in parts)
    assert "".join(p.replace("\n", "") for p in parts) == text.replace("\n", "")
    assert bot.split_message("y" * 2500, limit=1000) == ["y" * 1000, "y" * 1000, "y" * 500]
    assert bot.split_message("коротко") == ["коротко"]


async def test_long_reply_is_split(monkeypatch, tmp_db):
    await db.init()

    class OneHit:
        async def search(self, text):
            return [{"source": "a.md", "text": "t", "score": 1.0}]

    async def long_answer(question, fragments):
        return "ответ\n" * 1500                         # ~9000 символов

    monkeypatch.setattr(bot, "index", OneHit())
    monkeypatch.setattr(bot.llm, "answer", long_answer)
    msg = FakeMessage(text="вопрос")
    await bot.process(msg, "text", msg.text, {})
    assert len(msg.sent) >= 3 and all(len(p) <= bot.TELEGRAM_LIMIT for p in msg.sent)
    assert "📎 Источники: a.md" in msg.sent[-1]


async def test_text_error_replaces_status(monkeypatch):
    class Broken:
        async def search(self, text):
            raise RuntimeError("Ollama недоступна")

    monkeypatch.setattr(bot, "index", Broken())
    msg = StatusMessage(text="вопрос")
    await bot.on_text(msg)
    assert msg.status.edits == [bot.ERROR_TEXT] and not msg.status.deleted


async def test_text_success_deletes_status(demo_mode, tmp_db, tmp_index, monkeypatch):
    await db.init()
    index = Index()
    await index.build()
    monkeypatch.setattr(bot, "index", index)
    msg = StatusMessage(text="брак")
    await bot.on_text(msg)
    assert msg.status.deleted and msg.status.edits == []


async def test_reindex_error_is_reported_to_admin(monkeypatch):
    class Broken:
        async def build(self):
            raise RuntimeError("нет связи")

    monkeypatch.setattr(bot, "index", Broken())
    monkeypatch.setattr(config, "ADMIN_IDS", {100})
    admin = FakeMessage(user_id=100)
    await bot.on_reindex(admin)
    assert "Индекс не перестроен" in admin.sent[-1]


async def test_message_without_user_is_handled(monkeypatch):
    msg = FakeMessage()
    msg.from_user = None
    monkeypatch.setattr(config, "ADMIN_IDS", {100})
    await bot.on_reindex(msg)                           # не падает и молчит
    assert msg.sent == [] and bot.user_id(msg) == 0
