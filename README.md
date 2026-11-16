# Мастер смены — голосовой RAG-помощник для производства

**In English.** A Telegram bot that answers shop-floor questions from internal documents.
Voice messages are transcribed locally with faster-whisper; documents are split into chunks and embedded with bge-m3 via Ollama.
The top-3 chunks by cosine similarity go to a local Qwen 2.5 3B model (Ollama), which answers only from them and cites the source files.
Every request is logged to SQLite with per-stage timings; `DEMO_MODE=1` runs the same pipeline from the console with stubs instead of models.
Python, aiogram 3, NumPy, httpx, pytest (22 offline tests); the sample documents are fictional.

Сотрудник цеха отправляет в Telegram голосовое или текст о проблеме («опять брак по толщине листа, что делать?»). Бот:

1. расшифровывает голос локально (faster-whisper);
2. находит подходящие фрагменты регламентов и договорённостей (эмбеддинги bge-m3, косинусное сходство);
3. просит локальную модель (Qwen 2.5 3B через Ollama) коротко объяснить ситуацию и предложить следующий шаг **только по найденным фрагментам**;
4. отвечает со списком источников и временем каждого этапа;
5. пишет вопрос, ответ, источники и замеры в журнал SQLite.

Модели локальные: распознавание, эмбеддинги и ответ считаются на своём сервере, платных внешних API нет. Наружу ходит только Telegram — вопросы и ответы бота (с фрагментами документов) проходят через его серверы. Модель whisper при первом запуске скачивается с Hugging Face.

> Документы в `docs/` — **вымышленные** образцы, написанные для демонстрации. Предприятие, поставщик и цифры придуманы.

## Как устроено

```
голосовое (.ogg) ──► stt.py  faster-whisper, CPU, int8 ──┐
текст ───────────────────────────────────────────────────┤
                                                         ▼
docs/*.md ──► rag.py: куски по абзацам (~800 симв.) ──► эмбеддинги bge-m3 (Ollama)
              index.npz (+ имя модели) ◄── сохраняется   │
                                                         ▼
                                    top-3 фрагмента по косинусному сходству
                                                         ▼
                      llm.py: Qwen 2.5 3B (Ollama), temperature 0.2, «не выдумывай»
                                                         ▼
          bot.py: ответ + источники + время этапов ──► db.py: журнал journal.db
```

| Файл | Что делает |
|---|---|
| `bot.py` | Telegram-бот на aiogram 3: `/start`, `/docs`, `/reindex` (только админ), голос и текст |
| `stt.py` | Расшифровка голосовых через faster-whisper в отдельном потоке |
| `rag.py` | Нарезка документов, эмбеддинги через Ollama `/api/embed`, индекс в `index.npz`, поиск. Индекс помнит, какой моделью посчитан; если модель другая, он перестраивается |
| `llm.py` | Запрос к Ollama `/api/chat` с найденными фрагментами |
| `db.py` | Журнал обращений в SQLite |
| `demo.py` | Тот же поток из консоли, без Telegram |

## Стек
Python 3.10+ (проверено на 3.14), aiogram 3, Ollama (Qwen 2.5 3B, bge-m3), faster-whisper, NumPy, httpx, SQLite (aiosqlite), pytest.

Зависимости разделены: `requirements.txt` — ядро (бот без голоса, demo.py), `requirements-voice.txt` — плюс faster-whisper для голосовых, `requirements-dev.txt` — ядро и pytest.

## Быстрая проверка без моделей и Telegram (демо-режим)

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements-dev.txt      # без faster-whisper: ~130 МБ
DEMO_MODE=1 python demo.py "На второй линии опять брак по толщине листа, третий раз за смену, что делать?"
```

`DEMO_MODE=1` заменяет Ollama заглушками: эмбеддинги — хеш слов (ловит только совпадение слов, не смысл), вместо ответа модели выводится ближайший фрагмент. Это нужно, чтобы посмотреть поток данных, а не качество ответов. Демо-индекс пишется в отдельный файл `index.demo.npz` и не смешивается с настоящим.

Пример вывода (реальный запуск этой команды, сокращён):

```
Режим: DEMO_MODE (заглушки)
Фрагментов в индексе: 4
Вопрос: На второй линии опять брак по толщине листа, третий раз за смену, что делать?

Найдено:
  0.16  reglament_brak.md
  0.15  usloviya_postavki_lista.md
  0.11  instrukciya_ostanovka_linii.md

[демо-режим: модель не вызывалась]
Ближайший фрагмент [reglament_brak.md]:
# Регламент действий при обнаружении брака
...
📎 Источники: instrukciya_ostanovka_linii.md, reglament_brak.md, usloviya_postavki_lista.md
⏱ индекс 0.01 с · поиск 0.00 с · ответ 0.00 с
```

## Полный запуск

### 1. Ollama и модели
```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen2.5:3b     # ~1.9 ГБ
ollama pull bge-m3         # ~1.2 ГБ
```

### 2. Проект
```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements-voice.txt   # ядро + faster-whisper (~0.5 ГБ зависимостей)
cp .env.example .env       # токен от @BotFather и свой Telegram ID в ADMIN_IDS
```

### 3. Запуск
```bash
python demo.py "вопрос"    # проверить поиск и модель из консоли
python bot.py              # Telegram-бот
```
При первом запуске строится индекс документов, а faster-whisper скачивает модель (`small`, ~0.5 ГБ).

Свои документы: положите `.md` или `.txt` в `docs/` и отправьте боту `/reindex`.

Время этапов есть в каждом ответе и в журнале:
```bash
sqlite3 journal.db "SELECT kind, timings FROM journal ORDER BY id DESC LIMIT 10;"
```

## Тесты

```bash
pip install -r requirements-dev.txt
pytest
```

22 теста, без сети и без моделей; результат не зависит от `DEMO_MODE` в `.env`. Проверяются:
- нарезка текста, батчи и нормализация эмбеддингов (Ollama подменяется `httpx.MockTransport`);
- построение, сохранение и загрузка индекса, в том числе отказ загружать индекс, посчитанный другой моделью;
- запрос к модели и журнал;
- ответ бота с источниками и замерами, разбиение длинных ответов под лимит Telegram, сообщение об ошибке вместо зависшего статуса;
- `/reindex` только для админа и ответ админу при ошибке, сообщения без отправителя;
- прогон `demo.py`.

## Ограничения

- **Скорость на CPU.** Всё считается на процессоре: расшифровка, эмбеддинги и ответ 3B-модели могут занимать заметное время. Точных замеров в репозитории нет — бот показывает время этапов, по ним и стоит судить на своём железе.
- Маленькая модель (3B) может ошибаться или додумывать, несмотря на инструкцию «опирайся только на фрагменты». Источники в ответе нужны, чтобы это проверить.
- Индекс перестраивается целиком; поиск — полный перебор векторов в памяти. Для десятков–сотен документов этого хватает, для больших баз нужна векторная БД.
- Куски режутся по абзацам без перекрытия; таблицы и PDF не поддерживаются — только `.md` и `.txt`.
- Язык распознавания по умолчанию русский (`WHISPER_LANGUAGE` в `.env`), промпт модели тоже на русском.
- Доступ: задавать вопросы может любой, кто нашёл бота; ограничена только команда `/reindex`.
- Распознавание голоса (`stt.py`) тестами не покрыто — нужен файл и модель whisper.

## Лицензия
MIT, см. [LICENSE](LICENSE).
