"""The last stretch of talk, so the next session can continue it."""

import json, time

from .config import DATA

PATH = DATA / "transcript.jsonl"
KEEP_MINUTES = 60

class Log:
    def __init__(self, minutes, path=PATH):
        self.minutes, self.path = float(minutes or 0), path
        self.keep = max(self.minutes, KEEP_MINUTES)

    def add(self, who, text, name=None):
        text = " ".join((text or "").split())
        if not text:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        row = {"t": time.time(), "who": who, "text": text}
        if name:
            row["name"] = name
        with self.path.open("a") as handle:
            handle.write(json.dumps(row) + "\n")
        self.prune()

    def rows(self, minutes=None):
        window = self.keep if minutes is None else float(minutes)
        if window <= 0:
            return []
        cutoff = time.time() - window * 60
        found = []
        try:
            lines = self.path.read_text().splitlines()
        except OSError:
            return found
        for line in lines:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("who") and row.get("text") and float(row.get("t") or 0) >= cutoff:
                found.append(row)
        return found

    def prune(self):
        if not self.path.is_file():
            return
        body = "".join(json.dumps(row) + "\n" for row in self.rows(self.keep))
        self.path.write_text(body)

    def prompt(self, name):
        rows = self.rows(self.minutes)
        if not rows:
            return ""
        lines = []
        for row in rows:
            lines.append(f"{speaker_of(row, name)}: {row['text']}")
        window = f"{self.minutes:g}"
        return (
            f"Conversation from the last {window} minutes, oldest first. "
            "Continue it when they are still on that subject. Do not mention this record.\n"
            + "\n".join(lines)
        )

def speaker_of(row, agent):
    if row.get("who") == "assistant":
        return agent
    return row.get("name") or "Someone"


def recent(minutes, path=None):
    return Log(minutes, PATH if path is None else path).rows(minutes)
