# Start with --> nohup uvicorn live_transcript:app --host 0.0.0.0 --port 8080 > server.log 2>&1 &

import asyncio, subprocess, json, time, os, tempfile
from typing import Dict, Set, Optional, List
from urllib.parse import urljoin

import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, PlainTextResponse
from faster_whisper import WhisperModel
from fastapi.middleware.cors import CORSMiddleware

import aiohttp

# ---------------------- CONFIG ----------------------
M3U8_URL       = os.environ.get("M3U8_URL", "https://multiview.apisaranyu.in/srt_hls/streammain/stream.m3u8")
MODEL_SIZE     = os.environ.get("MODEL_SIZE", "small")   # tiny/base/small/medium/large-v3
DEVICE         = os.environ.get("DEVICE", "cpu")         # "cuda" or "cpu"  (ASR device)
COMPUTE_TYPE   = os.environ.get("COMPUTE_TYPE", "int8")  # float16/int8/int8_float32
LANGUAGE       = "en"                                    # fixed per spec
SAMPLE_RATE    = 16000
SEG_POLL_SEC   = float(os.environ.get("SEG_POLL_SEC", "2"))
TRANSCRIPT_TXT = os.environ.get("TRANSCRIPT_TXT", "transcript.txt")
EMIT_MIN_CHARS = int(os.environ.get("EMIT_MIN_CHARS", "6"))

# Translation (optional)
TARGET_LANGS            = ["fr"]
TARGET_LANGS = []
TRANSLATION_MODEL       = os.environ.get("TRANSLATION_MODEL", "facebook/m2m100_418M")
TRANSLATION_DEVICE      = os.environ.get("TRANSLATION_DEVICE", "cpu")  # "cpu" or "cuda"
TRANSLATION_MAX_CHARS   = int(os.environ.get("TRANSLATION_MAX_CHARS", "800"))
# ----------------------------------------------------

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)


# ---------- Fan-out broadcast (each client gets every msg) ----------
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

bcast = Fanout()

# ---------- ASR model ----------
model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)

# ---------- Optional translator (M2M100) ----------
translator_ready = False
if TARGET_LANGS:
    # Lazy-import heavy deps only if needed
    import torch
    from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

    class M2MTranslator:
        def __init__(self, model_name: str, device: str = "cpu"):
            self.tokenizer = M2M100Tokenizer.from_pretrained(model_name)
            self.model = M2M100ForConditionalGeneration.from_pretrained(model_name)
            if device == "cuda" and torch.cuda.is_available():
                self.model = self.model.half().to("cuda")
                self.device = "cuda"
            else:
                self.device = "cpu"
            # m2m uses simple ISO codes for many langs, e.g., en, hi, ta, te, kn, mr, bn, ml, pa, gu, ur
            self.tokenizer.src_lang = "en"

        def translate(self, text: str, tgt_lang: str) -> str:
            # guard: short-circuit empty/very small chunks
            t = text if len(text) <= TRANSLATION_MAX_CHARS else text[:TRANSLATION_MAX_CHARS]
            encoded = self.tokenizer(t, return_tensors="pt")
            if self.device == "cuda":
                encoded = {k: v.to("cuda") for k, v in encoded.items()}
            generated_tokens = self.model.generate(
                **encoded,
                forced_bos_token_id=self.tokenizer.get_lang_id(tgt_lang),
                max_new_tokens=512,
            )
            out = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            return out[0] if out else ""

        def translate_many(self, text: str, langs: List[str]) -> Dict[str, str]:
            return {lg: self.translate(text, lg) for lg in langs}

    _translator = M2MTranslator(TRANSLATION_MODEL, TRANSLATION_DEVICE)
    translator_ready = True

# ---------- Helpers ----------
async def fetch_text(session: aiohttp.ClientSession, url: str, timeout_sec: float = 5.0) -> str:
    async with session.get(url, timeout=aiohttp.ClientTimeout(total=timeout_sec)) as r:
        r.raise_for_status()
        return await r.text()

async def fetch_bytes(session: aiohttp.ClientSession, url: str, timeout_sec: float = 10.0) -> bytes:
    async with session.get(url, timeout=aiohttp.ClientTimeout(total=timeout_sec)) as r:
        r.raise_for_status()
        return await r.read()

def run_ffmpeg_to_pcm(input_path_or_url: str) -> np.ndarray:
    cmd = [
        "ffmpeg", "-hide_banner",
        "-loglevel", "error",
        "-i", input_path_or_url,
        "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE),
        "-f", "s16le", "pipe:1"
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg decode failed: rc={proc.returncode}, err={err.decode('utf-8','ignore')}")
    if not out:
        return np.zeros(0, dtype=np.int16)
    return np.frombuffer(out, dtype=np.int16)

def pcm16_to_float32(pcm: np.ndarray) -> np.ndarray:
    if pcm.dtype != np.int16:
        raise ValueError("expected int16 pcm")
    return pcm.astype(np.float32) / 32768.0

def transcribe_audio_block(audio_f32: np.ndarray) -> str:
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

def append_to_file(path: str, text: str):
    with open(path, "a", encoding="utf-8") as f:
        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        f.write(f"[{ts}] {text}\n")

# ---------- Manifest watcher + per-segment transcription ----------
async def playlist_watcher():
    seen: Set[str] = set()
    init_bytes: Optional[bytes] = None
    last_emitted_seq = 0

    async with aiohttp.ClientSession() as session:
        while True:
            try:
                text = await fetch_text(session, M3U8_URL, timeout_sec=5.0)
            except Exception as e:
                print(f"[playlist] fetch error: {e}")
                await asyncio.sleep(SEG_POLL_SEC)
                continue

            lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

            # capture EXT-X-MAP for fMP4
            for ln in lines:
                if ln.startswith("#EXT-X-MAP:"):
                    try:
                        part = next(p for p in ln.split(":")[1].split(",") if p.strip().startswith("URI="))
                        uri_val = part.split("=", 1)[1].strip().strip('"')
                        init_url = urljoin(M3U8_URL, uri_val)
                        if init_bytes is None:
                            init_bytes = await fetch_bytes(session, init_url, timeout_sec=10.0)
                            print(f"[playlist] cached init bytes from {init_url} ({len(init_bytes)} bytes)")
                    except Exception as e:
                        print(f"[playlist] failed to parse EXT-X-MAP: {e}")

            seg_urls: List[str] = []
            for ln in lines:
                if ln.startswith("#"):
                    continue
                seg_urls.append(urljoin(M3U8_URL, ln))

            new_segments = [u for u in seg_urls if u not in seen]
            if new_segments:
                for u in new_segments:
                    seen.add(u)
                if len(seen) > 1200:
                    for u in list(seen)[:len(seen)-1000]:
                        seen.remove(u)

            for seg_url in new_segments:
                t0 = time.time()
                print(f"[segment] new: {seg_url}")
                try:
                    if seg_url.endswith(".m4s") and init_bytes is not None:
                        seg_bytes = await fetch_bytes(session, seg_url, timeout_sec=10.0)
                        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=True) as tf:
                            tf.write(init_bytes)
                            tf.write(seg_bytes)
                            tf.flush()
                            pcm16 = run_ffmpeg_to_pcm(tf.name)
                    else:
                        pcm16 = run_ffmpeg_to_pcm(seg_url)

                    if pcm16.size == 0:
                        print(f"[segment] empty audio decoded, skipping")
                        continue

                    audio_f32 = pcm16_to_float32(pcm16)
                    text_out = transcribe_audio_block(audio_f32)
                    if len(text_out) < EMIT_MIN_CHARS:
                        continue

                    translations = None
                    if translator_ready and TARGET_LANGS:
                        # run translation in thread to avoid blocking the loop
                        translations = await asyncio.to_thread(
                            _translator.translate_many, text_out, TARGET_LANGS
                        )

                    last_emitted_seq += 1
                    msg = {
                        "seq": last_emitted_seq,
                    }
                    if translations:
                        msg["transcript"] = translations
                        msg["transcript"]["en"] = text_out
                    else:
                        msg["transcript"] = {"en": text_out}

                    await bcast.publish(msg)

                    # persist
                    append_to_file(TRANSCRIPT_TXT, text_out)
                    if translations:
                        for lg, txt in translations.items():
                            append_to_file(TRANSCRIPT_TXT, f"[{lg}] {txt}")

                    print(f"[segment] emitted seq={last_emitted_seq} ({time.time()-t0:.2f}s)")

                except Exception as e:
                    print(f"[segment] error processing {seg_url}: {e}")

            await asyncio.sleep(SEG_POLL_SEC)

# ---------- startup / shutdown ----------
_playlist_task: Optional[asyncio.Task] = None

@app.on_event("startup")
async def _startup():
    global _playlist_task
    _playlist_task = asyncio.create_task(playlist_watcher())
    print("[startup] playlist watcher started")

@app.on_event("shutdown")
async def _shutdown():
    global _playlist_task
    if _playlist_task:
        _playlist_task.cancel()
        try:
            await _playlist_task
        except Exception:
            pass
    print("[shutdown] stopped")

# ---------- SSE ----------
@app.get("/sse")
async def sse(request: Request):
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

    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }
    return StreamingResponse(gen(), media_type="text/event-stream", headers=headers)

@app.get("/health")
def healthz():
    return PlainTextResponse("ok")


