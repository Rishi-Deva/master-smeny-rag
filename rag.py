"""Поиск по документам: режем тексты на куски, считаем эмбеддинги через Ollama,
храним в index.npz и ищем ближайшие куски по косинусному сходству.

Вместе с векторами в индексе хранится, какой моделью они посчитаны. Если модель
сменилась (например, после DEMO_MODE или правки EMBED_MODEL), старый индекс
не загружается и строится заново: иначе размерности векторов не совпадут."""
import hashlib
import json
import logging
import re
from pathlib import Path

import httpx
import numpy as np

import config

log = logging.getLogger(__name__)

CHUNK_SIZE = 800  # символов в одном куске
BATCH = 16
DEMO_DIM = 512
DEMO_MODEL_ID = f"demo-hash-{DEMO_DIM}"


def current_model_id() -> str:
    """Чем сейчас считаются эмбеддинги: заглушка или модель Ollama."""
    return DEMO_MODEL_ID if config.DEMO_MODE else config.EMBED_MODEL


def demo_embed(texts: list[str]) -> np.ndarray:
    """Заглушка для DEMO_MODE: мешок слов (первые 5 букв слова), захешированный в вектор.
    Ловит только совпадение слов, а не смысл, — это не замена bge-m3."""
    v = np.zeros((len(texts), DEMO_DIM), dtype=np.float32)
    for row, text in enumerate(texts):
        for word in re.findall(r"\w+", text.lower()):
            if len(word) < 3:
                continue
            h = int(hashlib.md5(word[:5].encode()).hexdigest(), 16)
            v[row, h % DEMO_DIM] += 1.0
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return v / norms


def split_text(text: str, size: int = CHUNK_SIZE) -> list[str]:
    """Режем по абзацам, склеивая соседние, пока кусок не превысит size."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks, current = [], ""
    for p in paragraphs:
        if current and len(current) + len(p) > size:
            chunks.append(current)
            current = p
        else:
            current = f"{current}\n\n{p}" if current else p
    if current:
        chunks.append(current)
    return chunks


async def embed(texts: list[str]) -> np.ndarray:
    if config.DEMO_MODE:
        return demo_embed(texts)
    vectors = []
    async with httpx.AsyncClient(timeout=config.OLLAMA_TIMEOUT) as client:
        for i in range(0, len(texts), BATCH):
            r = await client.post(
                f"{config.OLLAMA_URL}/api/embed",
                json={"model": config.EMBED_MODEL, "input": texts[i:i + BATCH]},
            )
            r.raise_for_status()
            vectors.extend(r.json()["embeddings"])
    v = np.array(vectors, dtype=np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return v


class Index:
    def __init__(self, docs_dir: str | None = None, index_path: str | None = None):
        self.docs_dir = docs_dir or config.DOCS_DIR
        self.index_path = index_path or config.INDEX_PATH
        self.vectors: np.ndarray | None = None
        self.chunks: list[dict] = []

    async def build(self) -> int:
        chunks = []
        for path in sorted(Path(self.docs_dir).glob("**/*")):
            if path.suffix.lower() in {".md", ".txt"}:
                for part in split_text(path.read_text(encoding="utf-8")):
                    chunks.append({"source": path.name, "text": part})
        if not chunks:
            raise RuntimeError(f"В папке {self.docs_dir} нет .md/.txt документов")
        self.vectors = await embed([c["text"] for c in chunks])
        self.chunks = chunks
        np.savez(self.index_path, vectors=self.vectors,
                 chunks=json.dumps(chunks, ensure_ascii=False),
                 model=current_model_id())
        log.info("Индекс построен: %d кусков", len(chunks))
        return len(chunks)

    def load(self) -> bool:
        """Загружает индекс, только если он посчитан той же моделью, что и сейчас."""
        if not Path(self.index_path).exists():
            return False
        data = np.load(self.index_path)
        saved_model = str(data["model"]) if "model" in data.files else None
        if saved_model != current_model_id():
            log.warning("Индекс %s посчитан моделью %s, сейчас %s — будет перестроен",
                        self.index_path, saved_model, current_model_id())
            return False
        self.vectors = data["vectors"]
        self.chunks = json.loads(str(data["chunks"]))
        log.info("Индекс загружен: %d кусков", len(self.chunks))
        return True

    async def search(self, query: str, k: int | None = None) -> list[dict]:
        k = k or config.TOP_K
        q = (await embed([query]))[0]
        scores = self.vectors @ q
        best = np.argsort(scores)[::-1][:k]
        return [{**self.chunks[i], "score": float(scores[i])} for i in best]
