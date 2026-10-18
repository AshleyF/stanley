"""Load home.json and the local .env."""

import json, os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def load_env():
    paths = [
        ROOT / ".env",
        ROOT.parent / "prototypes" / "gpt-live-wake" / ".env",
        ROOT.parent / "prototypes" / "gpt-live-lights" / ".env",
    ]
    for path in paths:
        if not path.is_file():
            continue
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip("'\""))
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit(f"Set OPENAI_API_KEY in {paths[0]}")

def load_home():
    path = ROOT / "home.json"
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        sys.exit(f"Could not read {path}: {exc}")
    agent = data.get("agent") or {}
    name = str(agent.get("name") or "").strip()
    wake = str(agent.get("wake") or name).strip()
    voice = str(agent.get("voice") or "").strip()
    aliases = [str(item).strip() for item in (agent.get("wake_aliases") or []) if str(item).strip()]
    try:
        buffer_seconds = float(agent.get("buffer_seconds") or 180)
    except (TypeError, ValueError):
        buffer_seconds = 180
    lights = [light for light in (data.get("lights") or []) if light.get("name") and light.get("host")]
    doors = [door for door in (data.get("doors") or []) if door.get("name")]
    if not name or not wake or not voice:
        sys.exit(f"{path} needs agent.name, agent.wake, and agent.voice")
    if not lights and not doors:
        sys.exit(f"No lights or doors in {path}")
    return {
        "name": name, "wake": wake, "voice": voice, "aliases": aliases,
        "buffer_seconds": max(5.0, buffer_seconds), "lights": lights, "doors": doors,
    }
