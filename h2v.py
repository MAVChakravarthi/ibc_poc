# main.py
import os
import signal
import threading
import subprocess
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

SOURCE_HLS_URL = "https://multiview.apisaranyu.in/srt_hls/streammain/stream.m3u8"
OUTPUT_DIR = "output_stream"
PLAYLIST_NAME = "playlist.m3u8"
MOUNT_PATH = "/hls"

ffmpeg_process: subprocess.Popen | None = None
_ffmpeg_lock = threading.Lock()


def _build_ffmpeg_cmd():
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    return [
        "ffmpeg", "-i", SOURCE_HLS_URL,
        "-vf", "crop=ih*9/16:ih*0.8:(iw-ih*9/16)/2:0",
        "-c:v", "libx264", "-preset", "veryfast",
        "-g", "250", "-keyint_min", "250", "-sc_threshold", "0",
        "-c:a", "aac",
        "-f", "hls",
        "-hls_time", "10",
        "-hls_list_size", "8",
        "-hls_flags", "delete_segments",
        "-hls_segment_filename", str(Path(OUTPUT_DIR) / "segment_%05d.ts"),
        str(Path(OUTPUT_DIR) / PLAYLIST_NAME),
    ]


def run_ffmpeg():
    print("Running ffmpeg...")
    global ffmpeg_process
    cmd = _build_ffmpeg_cmd()
    with _ffmpeg_lock:
        if ffmpeg_process and ffmpeg_process.poll() is None:
            return
        ffmpeg_process = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            preexec_fn=os.setsid if hasattr(os, "setsid") else None,
        )


def stop_ffmpeg():
    global ffmpeg_process
    with _ffmpeg_lock:
        if ffmpeg_process and ffmpeg_process.poll() is None:
            try:
                if hasattr(os, "getpgid") and hasattr(os, "killpg"):
                    os.killpg(os.getpgid(ffmpeg_process.pid), signal.SIGTERM)
                else:
                    ffmpeg_process.terminate()
            except Exception:
                pass
        ffmpeg_process = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup
    threading.Thread(target=run_ffmpeg, daemon=True).start()
    yield
    # shutdown
    stop_ffmpeg()


app = FastAPI(lifespan=lifespan)
app.mount(MOUNT_PATH, StaticFiles(directory=OUTPUT_DIR), name="hls")


@app.get("/status")
def status():
    with _ffmpeg_lock:
        running = ffmpeg_process is not None and ffmpeg_process.poll() is None
    return JSONResponse({"running": running, "playlist": f"{MOUNT_PATH}/{PLAYLIST_NAME}"})


@app.post("/start")
def start():
    threading.Thread(target=run_ffmpeg, daemon=True).start()
    return JSONResponse({"message": "FFmpeg starting", "playlist": f"{MOUNT_PATH}/{PLAYLIST_NAME}"})


@app.post("/stop")
def stop():
    stop_ffmpeg()
    return JSONResponse({"message": "FFmpeg stopped"})

