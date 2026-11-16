import os

from dotenv import load_dotenv

load_dotenv()

# Токен нужен только Telegram-боту (bot.py); demo.py и тесты работают без него.
BOT_TOKEN = os.getenv("BOT_TOKEN", "")
ADMIN_IDS = {int(x) for x in os.getenv("ADMIN_IDS", "").split(",") if x.strip()}

# DEMO_MODE=1: вместо Ollama — заглушки (хеш-эмбеддинги слов и шаблонный ответ).
# Нужен, чтобы посмотреть поток данных без моделей; качество поиска и ответа — не настоящее.
DEMO_MODE = os.getenv("DEMO_MODE", "") == "1"

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_TIMEOUT = float(os.getenv("OLLAMA_TIMEOUT", "600"))  # секунд: на CPU ответ может идти долго
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5:3b")
LLM_NUM_CTX = int(os.getenv("LLM_NUM_CTX", "4096"))
EMBED_MODEL = os.getenv("EMBED_MODEL", "bge-m3")
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")
WHISPER_LANGUAGE = os.getenv("WHISPER_LANGUAGE", "ru")

DOCS_DIR = os.getenv("DOCS_DIR", "docs")
# Демо-индекс лежит отдельно от боевого, чтобы не смешивать векторы разных моделей.
INDEX_PATH = os.getenv("INDEX_PATH", "index.demo.npz" if DEMO_MODE else "index.npz")
DB_PATH = os.getenv("DB_PATH", "journal.db")
TOP_K = int(os.getenv("TOP_K", "3"))
