"""Per-person preference notes. One text file each, rewritten after a session.

This is not the memory index. A memory keeps an event. A preference note is a
short description that is allowed to change: the shower temperature was 101,
and now it is 100. A separate model call sees the session and the current
notes, and writes each note again.
"""

from pathlib import Path

from .log import record
from .memory import _json

PREFER = """You update preference notes. Return JSON only, no markdown: {"preferences": {"Name": "the full new note"}}.
Each value replaces that person's note. It is a short description of what they prefer, not a transcript and not a log of events.
Add a preference they stated. Change one they corrected. Keep anything this session does not contradict. If the session says nothing about that person's preferences, return their current note unchanged.
Only lines spoken by that person, or a preference they clearly claim as their own, may change their note. Do not copy one person's preference onto another.
"""


def note(text):
    print(f"\n[{text}]", flush=True)
    record("preferences", text)


def path_for(directory, name):
    return Path(directory) / f"{name}.txt"


def read_notes(directory, names):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    found = {}
    for name in names:
        path = path_for(directory, name)
        found[name] = path.read_text() if path.is_file() else ""
    return found


def prompt_block(names, notes):
    if not names:
        return ""
    lines = [
        "People in the house: " + ", ".join(names) + ". Someone is anybody else.",
        "When you know who is speaking, a shower request with no temperature and no preset uses that person's preference below.",
    ]
    for name in names:
        body = (notes.get(name) or "").strip() or "(none yet)"
        lines.append(f"{name}'s preferences:\n{body}")
    return "\n".join(lines)


def apply_updates(directory, names, raw):
    data = _json(raw) or {}
    prefs = data.get("preferences") if isinstance(data, dict) else None
    if not isinstance(prefs, dict):
        return []
    known = {name.casefold(): name for name in names}
    current = read_notes(directory, names)
    changed = []
    for key, value in prefs.items():
        name = known.get(str(key).strip().casefold())
        if name is None or not isinstance(value, str):
            continue
        text = value.strip()
        if not text or text == current[name].strip():
            continue
        path_for(directory, name).write_text(text + "\n")
        changed.append(name)
    return changed


async def revise(client, model, trace, directory, names):
    dossier = trace.dossier() if trace is not None else ""
    if not dossier or not names:
        return
    current = read_notes(directory, names)
    blocks = []
    for name in names:
        body = current[name].strip() or "(none yet)"
        blocks.append(f"{name}'s current preferences:\n{body}")
    user = "\n\n".join(blocks) + "\n\n" + dossier
    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": PREFER}, {"role": "user", "content": user}],
            reasoning_effort="low",
        )
        raw = response.choices[0].message.content or ""
    except Exception as exc:
        record("preferences", f"update failed: {exc}")
        return
    changed = apply_updates(directory, names, raw)
    if changed:
        note("preferences: updated " + ", ".join(changed))
