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
            f"Someone can report a problem with {name}, or with something {name} did. "
            "Examples of that sound, not a complete list: bug report, there's a bug, something's wrong, can you try to fix that. "
            "Other words with that same meaning count. "
            "A normal request, such as turning on a light, is not a report. "
            "If you do not yet understand what went wrong, ask. Do not call report_issue before that. "
            "A missing ability is not automatically an issue. If you can add it with a tool and you are likely to get it right, do that instead. "
            "If the ability looks complicated, slow, or unlikely to be right the first time, offer the choice out loud: file an issue and have it taken care of, or try it now. "
            "Call report_issue only once you understand, and only when they reported a problem you are not fixing yourself, or they chose the issue. "
            "The description is in their terms, including what they expected. "
            "If they also sound finished, call report_issue first and end_session after."
        )

    def schemas(self, _home):
        return [
            {
                "type": "function",
                "name": "report_issue",
                "description": "Save a problem with Stanley once you understand it. Not for a request you can still do, and not for a small ability you are about to add. Use it when they want an issue filed, including after they choose that over having you try.",
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
