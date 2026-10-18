"""One running log of what Home is doing. Bug reports copy from it."""

import time
import threading
from pathlib import Path

PATH = Path(__file__).resolve().parent / "home.log"
KEEP_SECONDS = 2 * 24 * 3600
LOCK = threading.Lock()

def stamped(line):
    try:
        return time.mktime(time.strptime(line[:19], "%Y-%m-%d %H:%M:%S"))
    except (ValueError, OverflowError):
        return None

def record(kind, text):
    text = " ".join(str(text or "").split())
    if not text:
        return
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} | {kind} | {text}\n"
    with LOCK:
        PATH.parent.mkdir(exist_ok=True)
        with PATH.open("a") as handle:
            handle.write(line)
        prune()

def prune():
    if not PATH.is_file():
        return
    cutoff = time.time() - KEEP_SECONDS
    kept = []
    for line in PATH.read_text().splitlines():
        when = stamped(line)
        if when is None or when >= cutoff:
            kept.append(line)
    PATH.write_text(("\n".join(kept) + "\n") if kept else "")

def since(minutes, path=None):
    path = PATH if path is None else path
    cutoff = time.time() - minutes * 60
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return ""
    return "\n".join(line for line in lines if (stamped(line) or 0) >= cutoff)
