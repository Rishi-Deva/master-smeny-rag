"""Расшифровка голосовых локально через faster-whisper (на CPU, int8)."""
import asyncio

import config

_model = None


def load_model():
    global _model
    if _model is None:
        # импорт здесь, чтобы demo.py и тесты работали без faster-whisper
        # (он ставится отдельно: pip install -r requirements-voice.txt)
        from faster_whisper import WhisperModel
        # при первом запуске модель скачается с Hugging Face (~0.5 ГБ для small)
        _model = WhisperModel(config.WHISPER_MODEL, device="cpu", compute_type="int8")
    return _model


def _transcribe(path: str) -> str:
    segments, _ = load_model().transcribe(path, language=config.WHISPER_LANGUAGE, vad_filter=True)
    return " ".join(s.text.strip() for s in segments).strip()


async def transcribe(path: str) -> str:
    # whisper тяжёлый и синхронный — уводим в отдельный поток, чтобы бот не зависал
    return await asyncio.to_thread(_transcribe, path)
