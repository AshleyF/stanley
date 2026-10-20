"""One log file per session, named by the time the session started."""

import time
import threading
from .config import DATA

DIR = DATA / "logs"
KEEP_SECONDS = 2 * 24 * 3600
LOCK = threading.Lock()
_current = None

def stamped(line):
    try:
        return time.mktime(time.strptime(line[:19], "%Y-%m-%d %H:%M:%S"))
    except (ValueError, OverflowError):
        return None

def _open():
    global _current
    DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    path = DIR / f"{stamp}.log"
    if path.exists():
        path = DIR / f"{stamp}-{time.time_ns() % 10000:04d}.log"
    _current = path
    path.write_text(f"{time.strftime('%Y-%m-%d %H:%M:%S')} | log | {path.name}\n")
    _prune()
    return path

def begin():
    with LOCK:
        return _open()

def record(kind, text):
    text = " ".join(str(text or "").split())
    if not text:
        return
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} | {kind} | {text}\n"
    with LOCK:
        if _current is None:
            _open()
        with _current.open("a") as handle:
            handle.write(line)

def _prune():
    if not DIR.is_dir():
        return
    cutoff = time.time() - KEEP_SECONDS
    for path in DIR.glob("*.log"):
        try:
            if path.stat().st_mtime < cutoff:
                path.unlink()
        except OSError:
            continue

def _lines(path, cutoff):
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return []
    return [line for line in lines if (stamped(line) or 0) >= cutoff]

def _key(path):
    parts = path.stem.split("-")
    extra = int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else -1
    return (parts[0], parts[1] if len(parts) > 1 else "", extra)

def since(minutes, path=None):
    cutoff = time.time() - minutes * 60
    if path is not None:
        return "\n".join(_lines(path, cutoff))
    if not DIR.is_dir():
        return ""
    files = []
    for item in DIR.glob("*.log"):
        try:
            if item.stat().st_mtime < cutoff:
                continue
        except OSError:
            continue
        files.append(item)
    found = []
    for item in sorted(files, key=_key):
        found.extend(_lines(item, cutoff))
    return "\n".join(found)
