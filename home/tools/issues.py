"""A problem with Stanley, saved as a file once he understands it."""

import re
import time
from pathlib import Path

from ..log import record, since
from ..transcript import recent

MINUTES = 60
PATH = Path(__file__).resolve().parents[1] / "issues"

def slug(title):
    cleaned = re.sub(r"[^a-z0-9]+", "-", title.casefold()).strip("-")
    return (cleaned or "issue")[:48]

def body(title, problem, transcript, logged, filed):
    talk = transcript or "(none)"
    notes = logged or "(none)"
    return (
        f"# {title}\n\n"
        f"Filed: {filed}\n\n"
        f"## Problem\n\n{problem}\n\n"
        f"## Transcript\n\n{talk}\n\n"
        f"## Log\n\n{notes}\n"
    )

def transcript_text(name, minutes=MINUTES):
    lines = []
    for row in recent(minutes):
        when = time.strftime("%H:%M:%S", time.localtime(float(row["t"])))
        speaker = name if row["who"] == "assistant" else "Person"
        lines.append(f"{when} {speaker}: {row['text']}")
    return "\n".join(lines)

class Issues:
    def handles(self, name):
        return name == "report_issue"

    def instructions(self, home):
        name = home["name"]
        return (
            f"Call report_issue only when they explicitly tell {name} to file an issue or a bug. "
            "Examples of that request, not a complete list: file an issue, file a bug, bug report. "
            "Other words with that same meaning count. A problem, a missing ability, or something too hard is not that request. "
            "If they did not ask to file one, do not file one and do not offer to. A one-off is run_script. A lasting ability is save_tool. "
            "When they did ask, and you do not yet understand the problem, ask. Then call report_issue with a short title and a description in their terms, including what they expected. "
            "If they also sound finished, call report_issue first and end_session after."
        )

    def schemas(self, _home):
        return [
            {
                "type": "function",
                "name": "report_issue",
                "description": "File an issue only after they explicitly say to file an issue or a bug. Not for a one-off and not for a new ability.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string", "description": "A few words naming the problem."},
                        "problem": {"type": "string", "description": "What went wrong, in their terms, and what they expected."},
                    },
                    "required": ["title", "problem"],
                },
            },
        ]

    async def call(self, name, args, home):
        title = " ".join(str(args.get("title") or "").split())
        problem = " ".join(str(args.get("problem") or "").split())
        if not title or not problem:
            return {"error": "Need a title and what went wrong. Ask them if you do not know."}
        filed = time.strftime("%Y-%m-%d %H:%M:%S")
        stamp = time.strftime("%Y%m%d-%H%M%S")
        path = PATH / f"{stamp}-{slug(title)}.md"
        if path.exists():
            path = PATH / f"{stamp}-{slug(title)}-{time.time_ns() % 1000:03d}.md"
        text = body(title, problem, transcript_text(home["name"]), since(MINUTES), filed)
        PATH.mkdir(exist_ok=True)
        path.write_text(text)
        record("issue", f"filed {path.name}")
        print(f"\nissue {path.name}", flush=True)
        return {"filed": True, "title": title}
