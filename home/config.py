"""Load home.json and the local .env."""

import json, os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"

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
    if not os.environ.get("XAI_API_KEY"):
        sys.exit(f"Set XAI_API_KEY in {paths[0]}")

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
    try:
        raw_pause = agent.get("pause_seconds")
        pause_seconds = float(8 if raw_pause is None else raw_pause)
    except (TypeError, ValueError):
        pause_seconds = 8
    try:
        raw_minutes = agent.get("transcript_minutes")
        transcript_minutes = float(1 if raw_minutes is None else raw_minutes)
    except (TypeError, ValueError):
        transcript_minutes = 1
    lights = [light for light in (data.get("lights") or []) if light.get("name") and light.get("host")]
    doors = [door for door in (data.get("doors") or []) if door.get("name")]
    showers = showers_from(data)
    people, voice_match, voice_prior = people_from(data)
    if not name or not wake or not voice:
        sys.exit(f"{path} needs agent.name, agent.wake, and agent.voice")
    if not lights and not doors and not showers:
        sys.exit(f"No lights, doors, or showers in {path}")
    return {
        "name": name, "wake": wake, "voice": voice, "aliases": aliases,
        "buffer_seconds": max(5.0, buffer_seconds),
        "pause_seconds": min(max(5.0, buffer_seconds), max(1.0, pause_seconds)),
        "transcript_minutes": min(240.0, max(0.0, transcript_minutes)),
        "lights": lights, "doors": doors,
        "showers": showers,
        "memory": memory_from(data),
        "people": people,
        "voice_match": voice_match,
        "voice_prior": voice_prior,
    }

def people_from(data):
    names = []
    for item in data.get("people") or []:
        name = item if isinstance(item, str) else (item or {}).get("name")
        name = str(name or "").strip()
        if name and "/" not in name and name not in names:
            names.append(name)
    try:
        match = data.get("voice_match")
        match = float(0.50 if match is None else match)
    except (TypeError, ValueError):
        match = 0.50
    try:
        prior = data.get("voice_prior")
        prior = float(0.95 if prior is None else prior)
    except (TypeError, ValueError):
        prior = 0.95
    return names, min(0.99, max(0.0, match)), min(0.99, max(0.0, prior))


def memory_from(data):
    raw = data.get("memory") or {}

    def number(key, default):
        try:
            value = raw.get(key)
            return float(default if value is None else value)
        except (TypeError, ValueError):
            return default

    return {
        "embed_model": str(raw.get("embed_model") or "intfloat/e5-small-v2").strip() or "intfloat/e5-small-v2",
        "think_model": str(raw.get("think_model") or "grok-4.7").strip() or "grok-4.7",
        "recall": min(0.99, max(0.0, number("recall", 0.50))),
        "merge": min(0.99, max(0.0, number("merge", 0.62))),
    }

def showers_from(data):
    showers = []
    for shower in data.get("showers") or []:
        if not shower.get("name"):
            continue
        presets = []
        for preset in shower.get("presets") or []:
            if not preset.get("name") or preset.get("index") is None:
                continue
            try:
                index = int(preset["index"])
            except (TypeError, ValueError):
                continue
            presets.append({"name": str(preset["name"]).strip(), "index": index})
        try:
            maximum = int(shower.get("max_temperature") or 110)
        except (TypeError, ValueError):
            maximum = 110
        showers.append({
            "name": str(shower["name"]).strip(),
            "description": str(shower.get("description") or "").strip(),
            "serial": str(shower.get("serial") or "").strip(),
            "max_temperature": maximum,
            "presets": presets,
        })
    return showers
