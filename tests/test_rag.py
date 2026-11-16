import json

import httpx
import numpy as np
import pytest

import config
import rag
from rag import Index, demo_embed, split_text


def test_split_text_glues_paragraphs_until_size():
    text = "a" * 300 + "\n\n" + "b" * 300 + "\n\n" + "c" * 300
    chunks = split_text(text, size=700)
    assert chunks == ["a" * 300 + "\n\n" + "b" * 300, "c" * 300]


def test_split_text_keeps_long_paragraph_whole_and_skips_empty():
    long = "x" * 2000
    assert split_text(f"\n\n{long}\n\n\n\n", size=800) == [long]
    assert split_text("   \n\n  ") == []


def test_demo_embed_is_normalized_and_handles_empty_text():
    v = demo_embed(["брак на линии", ""])
    assert v.shape == (2, rag.DEMO_DIM)
    assert np.isclose(np.linalg.norm(v[0]), 1.0)
    assert not v[1].any()


async def test_ollama_embed_batches_and_normalizes(monkeypatch):
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        calls.append(len(body["input"]))
        assert request.url.path == "/api/embed"
        assert body["model"] == config.EMBED_MODEL
        return httpx.Response(200, json={"embeddings": [[3.0, 4.0]] * len(body["input"])})

    real_client = httpx.AsyncClient
    monkeypatch.setattr(rag.httpx, "AsyncClient",
                        lambda **kw: real_client(transport=httpx.MockTransport(handler), **kw))

    v = await rag.embed([f"t{i}" for i in range(rag.BATCH + 4)])
    assert calls == [rag.BATCH, 4]
    assert v.shape == (rag.BATCH + 4, 2)
    assert np.allclose(v[0], [0.6, 0.8])


async def test_index_build_search_and_reload(demo_mode, tmp_index):
    index = Index()
    n = await index.build()
    assert n >= 4
    assert {c["source"] for c in index.chunks} >= {"reglament_brak.md", "peredacha_smeny.md"}

    hits = await index.search("брак повторяется третий раз за смену", k=2)
    assert len(hits) == 2
    assert hits[0]["source"] == "reglament_brak.md"
    assert hits[0]["score"] >= hits[1]["score"]

    reloaded = Index()
    assert reloaded.load()
    assert reloaded.chunks == index.chunks
    assert np.allclose(reloaded.vectors, index.vectors)


async def test_index_build_fails_on_empty_folder(demo_mode, tmp_path):
    (tmp_path / "notes.pdf").write_text("не тот формат")
    with pytest.raises(RuntimeError, match="нет .md/.txt"):
        await Index(docs_dir=str(tmp_path), index_path=str(tmp_path / "i.npz")).build()


def test_load_without_index_returns_false(tmp_path):
    assert Index(index_path=str(tmp_path / "missing.npz")).load() is False


async def test_index_from_other_model_is_not_loaded(tmp_index, monkeypatch):
    """Регрессия: демо-индекс (512 измерений) не должен подхватываться боевым режимом (bge-m3)."""
    monkeypatch.setattr(config, "DEMO_MODE", True)
    await Index().build()
    assert Index().load() is True

    monkeypatch.setattr(config, "DEMO_MODE", False)
    assert Index().load() is False            # другая модель — индекс нужно перестроить

    monkeypatch.setattr(config, "DEMO_MODE", True)
    monkeypatch.setattr(config, "EMBED_MODEL", "other-model")
    assert Index().load() is True             # в демо модель Ollama не важна


def test_index_without_model_tag_is_rebuilt(tmp_index):
    np.savez(tmp_index, vectors=np.zeros((1, 4), dtype=np.float32), chunks="[]")
    assert Index().load() is False


def test_demo_index_has_separate_default_path(monkeypatch):
    import importlib
    monkeypatch.setenv("DEMO_MODE", "1")
    monkeypatch.delenv("INDEX_PATH", raising=False)
    try:
        assert importlib.reload(config).INDEX_PATH == "index.demo.npz"
        monkeypatch.setenv("DEMO_MODE", "")
        assert importlib.reload(config).INDEX_PATH == "index.npz"
    finally:
        monkeypatch.undo()
        importlib.reload(config)
