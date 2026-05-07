"""
Streaming Recitation WebSocket

Receives raw PCM audio chunks from the browser, runs VAD to detect
utterance boundaries, runs CTC inference, and tracks position in the surah.

Protocol:
  Client → Server: binary PCM frames (Int16, 16kHz mono)
  Client → Server: text "END" to signal session end
  Server → Client: JSON position updates
  Server → Client: { "type": "error", "message": str }
  Server → Client: { "type": "done" }
"""
import asyncio
import json
import logging
from concurrent.futures import ThreadPoolExecutor

from fastapi import APIRouter, WebSocket

from ..services.vad import VADAccumulator
from ..services.position_tracker import PositionTracker

logger = logging.getLogger(__name__)

router = APIRouter()
_executor = ThreadPoolExecutor(max_workers=2)


def get_services():
    from .. import main
    return main.reference_service, main.speech_service, main.alignment_service


def _is_disconnect_error(e: Exception) -> bool:
    return any(x in type(e).__name__ for x in ("Disconnect", "WebSocket", "Connection"))


@router.websocket("/ws/recite/{surah_number}")
async def recite_stream(websocket: WebSocket, surah_number: int):
    await websocket.accept()

    try:
        ref_service, speech_service, align_service = get_services()
    except Exception as e:
        await _safe_send(websocket, {"type": "error", "message": f"Service unavailable: {e}"})
        return

    if not ref_service.get_surah_info(surah_number):
        await _safe_send(websocket, {"type": "error", "message": f"Surah {surah_number} not found"})
        return

    # Build str->id vocab for PositionTracker
    vocab_str = {v: k for k, v in speech_service.vocab.items()}

    vad = VADAccumulator()
    tracker = PositionTracker(surah_number, ref_service, align_service, vocab_str)
    loop = asyncio.get_event_loop()

    logger.info(f"Streaming session started: surah {surah_number}")

    try:
        while True:
            message = await websocket.receive()

            if message.get("text") == "END":
                audio = vad.flush_remaining()
                if audio is not None:
                    await _infer_and_send(loop, websocket, audio, speech_service, tracker)
                await _safe_send(websocket, {"type": "done"})
                break

            elif "bytes" in message:
                pcm_bytes = message["bytes"]
                audio, should_reset = vad.feed(pcm_bytes)
                if should_reset:
                    tracker.reset()
                if audio is not None:
                    await _infer_and_send(loop, websocket, audio, speech_service, tracker)
                if tracker.is_complete():
                    await _safe_send(websocket, {"type": "done"})
                    break

    except Exception as e:
        if _is_disconnect_error(e):
            logger.info(f"Client disconnected: surah {surah_number}")
        else:
            logger.exception(f"Streaming error: {e}")


async def _infer_and_send(loop, websocket: WebSocket, audio, speech_service, tracker: PositionTracker):
    """Run blocking inference in thread pool, then update trellis and send result."""
    try:
        logits = await loop.run_in_executor(_executor, lambda: speech_service.get_logits(audio))
        if logits is None or logits.shape[0] == 0:
            return
        update = tracker.update_with_logits(logits)
        if update is None:
            return
        await _safe_send(websocket, {
            "type": "position",
            "ayah": update.ayah_number,
            "word": update.word_index,
            "confidence": round(update.confidence, 3),
            "letter_results": update.letter_results,
            "completed_ayah": update.completed_ayah,
        })
    except Exception as e:
        if not _is_disconnect_error(e):
            logger.warning(f"Inference error: {e}")


async def _safe_send(websocket: WebSocket, data: dict):
    """Send JSON, ignoring disconnect errors."""
    try:
        await websocket.send_text(json.dumps(data, ensure_ascii=False))
    except Exception as e:
        if not _is_disconnect_error(e):
            logger.warning(f"Send error: {e}")
