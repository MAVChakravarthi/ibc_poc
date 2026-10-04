# ----------------------------------------------- V5: SSE-triggered start, per-seg concurrent, optional translations -----------------------------------

import asyncio, subprocess, json, time, os, tempfile, concurrent.futures
from typing import Dict, Set, Optional, List
from urllib.parse import urljoin

import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from faster_whisper import WhisperModel
import aiohttp

# Local multilingual translation (NLLB-200).
# This replaces the previous deep-translator/Google web endpoint so live
# transcription does not depend on Google's rate-limited public endpoint.
_TRANSLATE_AVAILABLE = True
try:
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
except Exception as _e:
    _TRANSLATE_AVAILABLE = False
    print(f"[translate] transformers/torch not available ({_e}); translations disabled.")

# ---------------------- CONFIG ----------------------
M3U8_URL         = os.environ.get("M3U8_URL", "https://multiview.apisaranyu.in/srt_hls/streammain/stream.m3u8")
MODEL_SIZE       = os.environ.get("MODEL_SIZE", "small")   # tiny/base/small/medium/large-v3
DEVICE           = os.environ.get("DEVICE", "cpu")         # "cuda" or "cpu"
COMPUTE_TYPE     = os.environ.get("COMPUTE_TYPE", "int8")  # float16/int8/int8_float32
LANGUAGE         = "en"
SAMPLE_RATE      = 16000
SEG_POLL_SEC     = float(os.environ.get("SEG_POLL_SEC", "2"))
TRANSCRIPT_TXT   = os.environ.get("TRANSCRIPT_TXT", "transcript.txt")
EMIT_MIN_CHARS   = int(os.environ.get("EMIT_MIN_CHARS", "6"))

# Concurrency
MAX_SEG_WORKERS  = int(os.environ.get("MAX_SEG_WORKERS", "4"))
FFMPEG_TIMEOUT   = float(os.environ.get("FFMPEG_TIMEOUT", "25"))

# Translations
TRANSLATE_ENABLED = os.environ.get("TRANSLATE_ENABLED", "1") not in ("0", "false", "False", "")
TARGET_LANGS = [
    lang.strip().lower()
    for lang in os.environ.get("TARGET_LANGS", "de,es,fr").split(",")
    if lang.strip()
]

# NLLB language codes. Add more entries here when a new target language is needed.
# TARGET_LANGS continues to use simple ISO-style codes such as "de", "hi", "te".
NLLB_LANG_CODES = {
    "de": "deu_Latn",
    "es": "spa_Latn",
    "fr": "fra_Latn",
    "hi": "hin_Deva",
    "te": "tel_Telu",
    "ta": "tam_Taml",
    "kn": "kan_Knda",
    "ml": "mal_Mlym",
    "bn": "ben_Beng",
    "mr": "mar_Deva",
    "gu": "guj_Gujr",
    "pa": "pan_Guru",
    "ur": "urd_Arab",
    "or": "ory_Orya",
    "as": "asm_Beng",
    "ne": "npi_Deva",
}

TRANSLATION_MODEL_NAME = os.environ.get(
    "TRANSLATION_MODEL",
    "facebook/nllb-200-distilled-600M",
)
TRANSLATION_DEVICE = os.environ.get(
    "TRANSLATION_DEVICE",
    "cuda" if os.environ.get("DEVICE", "cpu").lower() == "cuda" else "cpu",
)
TRANSLATION_MAX_INPUT_CHARS = int(os.environ.get("TRANSLATION_MAX_INPUT_CHARS", "1000"))

# ----------------------------------------------------

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

# ---------- Fan-out broadcast ----------
class Fanout:
    def __init__(self, max_queue: int = 200):
        self._subs: Dict[int, asyncio.Queue] = {}
        self._next_id = 0
        self._lock = asyncio.Lock()
        self._max_queue = max_queue

    async def subscribe(self) -> asyncio.Queue:
        q = asyncio.Queue(maxsize=self._max_queue)
        async with self._lock:
            sid = self._next_id
            self._next_id += 1
            self._subs[sid] = q
        q._sid = sid  # type: ignore[attr-defined]
        return q

    async def unsubscribe(self, q: asyncio.Queue):
        sid = getattr(q, "_sid", None)
        async with self._lock:
            if sid in self._subs:
                del self._subs[sid]

    async def publish(self, msg: dict):
        async with self._lock:
            subs = list(self._subs.values())
        for q in subs:
            try:
                q.put_nowait(msg)
            except asyncio.QueueFull:
                try:
                    _ = q.get_nowait()
                except asyncio.QueueEmpty:
                    pass
                try:
                    q.put_nowait(msg)
                except asyncio.QueueFull:
                    pass

    async def subscriber_count(self) -> int:
        async with self._lock:
            return len(self._subs)

bcast = Fanout()

# ---------- ASR model ----------
model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)

# ---------- HTTP helpers ----------
async def fetch_text(session: aiohttp.ClientSession, url: str, timeout_sec: float = 5.0) -> str:
    async with session.get(url, timeout=aiohttp.ClientTimeout(total=timeout_sec)) as r:
        r.raise_for_status()
        return await r.text()

async def fetch_bytes(session: aiohttp.ClientSession, url: str, timeout_sec: float = 10.0) -> bytes:
    async with session.get(url, timeout=aiohttp.ClientTimeout(total=timeout_sec)) as r:
        r.raise_for_status()
        return await r.read()

# ---------- media/ASR helpers (sync; run in threads) ----------
def run_ffmpeg_to_pcm_blocking(input_path_or_url: str) -> np.ndarray:
    cmd = [
        "ffmpeg", "-hide_banner",
        "-loglevel", "error",
        "-i", input_path_or_url,
        "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE),
        "-f", "s16le", "pipe:1"
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        out, err = proc.communicate(timeout=FFMPEG_TIMEOUT)
    except subprocess.TimeoutExpired:
        proc.kill()
        raise RuntimeError(f"ffmpeg decode timeout ({FFMPEG_TIMEOUT}s) for {input_path_or_url}")
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg decode failed: rc={proc.returncode}, err={err.decode('utf-8','ignore')}")
    if not out:
        return np.zeros(0, dtype=np.int16)
    return np.frombuffer(out, dtype=np.int16)

def pcm16_to_float32(pcm: np.ndarray) -> np.ndarray:
    if pcm.dtype != np.int16:
        raise ValueError("expected int16 pcm")
    return pcm.astype(np.float32) / 32768.0

def transcribe_audio_block_blocking(audio_f32: np.ndarray) -> str:
    segments, _ = model.transcribe(
        audio_f32,
        language=LANGUAGE,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=250),
        condition_on_previous_text=False,
        beam_size=5,
        word_timestamps=False,
    )
    text = " ".join(s.text.strip() for s in segments).strip()
    return text

# ---------- Translation helper ----------
# Keep the same async interface used by process_segment(), but translate locally
# with NLLB-200 instead of deep-translator/Google web requests.
_translation_tokenizer = None
_translation_model = None
_translation_lock = asyncio.Lock()


def _load_translation_model_sync():
    global _translation_tokenizer, _translation_model

    if not _TRANSLATE_AVAILABLE:
        return False

    if _translation_model is not None and _translation_tokenizer is not None:
        return True

    print(
        f"[translate] loading local model {TRANSLATION_MODEL_NAME} "
        f"on {TRANSLATION_DEVICE}..."
    )

    _translation_tokenizer = AutoTokenizer.from_pretrained(TRANSLATION_MODEL_NAME)
    _translation_model = AutoModelForSeq2SeqLM.from_pretrained(
        TRANSLATION_MODEL_NAME
    )

    if TRANSLATION_DEVICE.lower() == "cuda" and torch.cuda.is_available():
        _translation_model = _translation_model.to("cuda")
    else:
        _translation_model = _translation_model.to("cpu")

    _translation_model.eval()
    print(
        f"[translate] local model ready: {TRANSLATION_MODEL_NAME} "
        f"device={next(_translation_model.parameters()).device}"
    )
    return True


def _translate_one_sync(text: str, target_lang: str) -> Optional[str]:
    if not text or not _load_translation_model_sync():
        return None

    target_code = NLLB_LANG_CODES.get(target_lang)
    if not target_code:
        print(f"[translate] unsupported target language: {target_lang}")
        return None

    # NLLB-200 expects the source language to be set on the tokenizer.
    _translation_tokenizer.src_lang = "eng_Latn"

    # Avoid unexpectedly huge generation input from a malformed/very long segment.
    source_text = text[:TRANSLATION_MAX_INPUT_CHARS]

    inputs = _translation_tokenizer(
        source_text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )

    model_device = next(_translation_model.parameters()).device
    inputs = {k: v.to(model_device) for k, v in inputs.items()}

    forced_bos_token_id = _translation_tokenizer.convert_tokens_to_ids(target_code)

    with torch.inference_mode():
        generated_tokens = _translation_model.generate(
            **inputs,
            forced_bos_token_id=forced_bos_token_id,
            max_length=512,
            num_beams=4,
        )

    return _translation_tokenizer.batch_decode(
        generated_tokens,
        skip_special_tokens=True,
    )[0].strip()


async def translate_text_multi(
    source_lang: str,
    text: str,
    targets: List[str],
) -> Dict[str, str]:
    """
    Returns dict of {lang: translated_text}, skipping failures.

    The public interface is unchanged. Translation is performed locally with
    NLLB-200, so there are no Google Translate requests/rate limits.
    """
    results: Dict[str, str] = {}

    if not text or not targets or not _TRANSLATE_AVAILABLE:
        return results

    # This NLLB model is shared by all segment-processing tasks. Serialize
    # generation so concurrent live segments do not simultaneously mutate/use
    # the same model object.
    async with _translation_lock:
        for tgt in targets:
            tgt = tgt.strip().lower()
            if tgt == source_lang.lower():
                continue

            if tgt not in NLLB_LANG_CODES:
                print(f"[translate] unsupported target language: {tgt}")
                continue

            try:
                translated = await asyncio.to_thread(
                    _translate_one_sync,
                    text,
                    tgt,
                )
                if translated:
                    results[tgt] = translated
            except Exception as e:
                # Keep the original flow alive: a translation failure must not
                # prevent the English transcript from being emitted over SSE.
                print(f"[translate] en->{tgt} error: {e}")

    return results


# ---------- Concurrency primitives ----------
executor = concurrent.futures.ThreadPoolExecutor(max_workers=MAX_SEG_WORKERS)
SEQ_LOCK = asyncio.Lock()
FILE_LOCK = asyncio.Lock()
last_emitted_seq = 0  # guarded by SEQ_LOCK

def _append_to_file_sync(path: str, text: str):
    ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"[{ts}] {text}\n")

async def append_to_file(path: str, text: str):
    async with FILE_LOCK:
        await asyncio.to_thread(_append_to_file_sync, path, text)

# ---------- Per-segment processing (runs concurrently) ----------
async def process_segment(seg_url: str, init_bytes: Optional[bytes]):
    t0 = time.time()
    try:
        # Decode to PCM (blocking → thread)
        if seg_url.endswith(".m4s") and init_bytes is not None:
            def _rebuild_and_decode():
                with tempfile.NamedTemporaryFile(suffix=".mp4", delete=True) as tf:
                    tf.write(init_bytes)
                    import urllib.request
                    with urllib.request.urlopen(seg_url, timeout=10) as r:
                        seg_bytes = r.read()
                    tf.write(seg_bytes)
                    tf.flush()
                    return run_ffmpeg_to_pcm_blocking(tf.name)
            pcm16 = await asyncio.to_thread(_rebuild_and_decode)
        else:
            pcm16 = await asyncio.to_thread(run_ffmpeg_to_pcm_blocking, seg_url)

        if pcm16.size == 0:
            print(f"[segment] empty audio decoded, skipping: {seg_url}")
            return

        audio_f32 = pcm16_to_float32(pcm16)
        text_out = await asyncio.to_thread(transcribe_audio_block_blocking, audio_f32)
        if len(text_out) < EMIT_MIN_CHARS:
            return

        # Allocate sequence
        async with SEQ_LOCK:
            global last_emitted_seq
            last_emitted_seq += 1
            seq = last_emitted_seq

        # Build transcript payload (EN + optional translations)
        transcripts: Dict[str, str] = {"en": text_out}

        if TRANSLATE_ENABLED and TARGET_LANGS:
            try:
                tr_map = await translate_text_multi("en", text_out, [t for t in TARGET_LANGS if t != "en"])
                if tr_map:
                    transcripts.update(tr_map)
            except Exception as e:
                print(f"[translate] batch error: {e}")

        msg = {"seq": seq, "transcript": transcripts}

        # Publish SSE + append English to file (keep file clean/simple)
        await bcast.publish(msg)
        await append_to_file(TRANSCRIPT_TXT, text_out)

        print(f"[segment] emitted seq={seq} ({time.time()-t0:.2f}s) -> {seg_url}")

    except Exception as e:
        print(f"[segment] error processing {seg_url}: {e}")

# ---------- Playlist watcher (lazy; tied to subscribers) ----------
_playlist_task: Optional[asyncio.Task] = None
_seen: Set[str] = set()
_init_bytes: Optional[bytes] = None
_watch_stop_evt = asyncio.Event()

async def playlist_watcher():
    global _init_bytes
    _watch_stop_evt.clear()
    sem = asyncio.Semaphore(MAX_SEG_WORKERS)

    async def submit(seg_url: str):
        async with sem:
            await process_segment(seg_url, _init_bytes)

    async with aiohttp.ClientSession() as session:
        while not _watch_stop_evt.is_set():
            try:
                text = await fetch_text(session, M3U8_URL, timeout_sec=5.0)
            except Exception as e:
                print(f"[playlist] fetch error: {e}")
                await asyncio.sleep(SEG_POLL_SEC)
                continue

            lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

            # Capture EXT-X-MAP (for fMP4)
            for ln in lines:
                if ln.startswith("#EXT-X-MAP:"):
                    try:
                        part = next(p for p in ln.split(":")[1].split(",") if p.strip().startswith("URI="))
                        uri_val = part.split("=", 1)[1].strip().strip('"')
                        init_url = urljoin(M3U8_URL, uri_val)
                        if _init_bytes is None:
                            _init_bytes = await fetch_bytes(session, init_url, timeout_sec=10.0)
                            print(f"[playlist] cached init bytes from {init_url} ({len(_init_bytes)} bytes)")
                    except Exception as e:
                        print(f"[playlist] failed to parse EXT-X-MAP: {e}")

            seg_urls: List[str] = []
            for ln in lines:
                if ln.startswith("#"):
                    continue
                seg_urls.append(urljoin(M3U8_URL, ln))

            # On first pass after /sse subscription, _seen is likely empty:
            # -> this processes ALL current segments (typically 3 for a 30s live window).
            new_segments = [u for u in seg_urls if u not in _seen]
            if new_segments:
                for u in new_segments:
                    _seen.add(u)
                # prune old (safety)
                if len(_seen) > 2000:
                    for u in list(_seen)[:len(_seen)-1500]:
                        _seen.remove(u)

            # Kick off processing tasks concurrently
            for seg_url in new_segments:
                asyncio.create_task(submit(seg_url))

            await asyncio.sleep(SEG_POLL_SEC)

    print("[watcher] exited")

async def ensure_watcher_running():
    global _playlist_task
    if _playlist_task is None or _playlist_task.done():
        print("[watcher] starting (lazy on first subscriber)")
        _playlist_task = asyncio.create_task(playlist_watcher())

async def maybe_stop_watcher():
    # Stop the watcher when no subscribers remain
    if await bcast.subscriber_count() == 0:
        print("[watcher] stopping (no subscribers)")
        _watch_stop_evt.set()
        if _playlist_task:
            try:
                await asyncio.wait_for(_playlist_task, timeout=2.0)
            except Exception:
                pass

# ---------- SSE ----------
@app.get("/sse")
async def sse(request: Request):
    # First subscriber triggers watcher start
    await ensure_watcher_running()
    q = await bcast.subscribe()

    async def gen():
        try:
            yield ": ok\n\n"
            while True:
                if await request.is_disconnected():
                    break
                try:
                    msg = await asyncio.wait_for(q.get(), timeout=5.0)
                except asyncio.TimeoutError:
                    yield ": keep-alive\n\n"
                    continue
                yield f"data: {json.dumps(msg, ensure_ascii=False)}\n\n"
        finally:
            await bcast.unsubscribe(q)
            await maybe_stop_watcher()

    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }
    return StreamingResponse(gen(), media_type="text/event-stream", headers=headers)

# ---------- Health ----------
@app.get("/healthz")
def healthz():
    return PlainTextResponse("ok")

