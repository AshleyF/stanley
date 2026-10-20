"""A problem with Stanley, saved as a file once he understands it."""

import re
import time

from ..config import DATA
from ..log import record, since
from ..transcript import recent, speaker_of

MINUTES = 60
PATH = DATA / "issues"

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
        speaker = speaker_of(row, name)
        lines.append(f"{when} {speaker}: {row['text']}")
    return "\n".join(lines)

class Issues:
    def handles(self, name):
        return name == "report_issue"

    def instructions(self, home):
        name = home["name"]
        return (
            f"A request {name} can do by writing Python and running it is run_script, not an issue. "
            "A deeper feature of Home, something that should be built in rather than run as a script, is an issue. "
            "Ask until you understand it, then call report_issue and tell them you are filing it. "
            "Also call report_issue when they explicitly say to file an issue or a bug. "
            "Do not file because a script looks hard or failed. Fix the script. "
            "Do not offer a choice, and do not talk about saving a tool. There is no tool to save. "
            "Use a short title and a description in their terms, including what they expected. "
            "If they also sound finished, call report_issue first and end_session after."
        )

    def schemas(self, _home):
        return [
            {
                "type": "function",
                "name": "report_issue",
                "description": "File an issue for a deeper feature of Home, or when they explicitly say to file an issue or a bug. Not for something a script can do now.",
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
