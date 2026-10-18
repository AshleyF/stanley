"""Local reminders. The clock runs here. A session opens only to speak."""

import json
import re
import time
import uuid
from pathlib import Path

MAX_ATTEMPTS = 6
PATH = Path(__file__).resolve().parents[1] / "timers.json"

def minutes_of(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number:
        return None
    return number

def length_phrase(seconds):
    if seconds < 90:
        return f"{max(1, int(round(seconds)))} seconds"
    minutes = seconds / 60
    if abs(minutes - round(minutes)) < 0.05:
        return f"{int(round(minutes))} minutes"
    return f"{minutes:.1f} minutes"

def nag_seconds(original_seconds, attempts):
    if original_seconds <= 10 * 60:
        base = 30
    elif original_seconds <= 30 * 60:
        base = 90
    else:
        base = 150
    return min(900, int(base * (1 + 0.5 * max(0, attempts - 1))))

def clock(when):
    return time.strftime("%I:%M %p", time.localtime(when)).lstrip("0")

TIME_TAIL = re.compile(
    r"\s+in\s+\d+(?:\.\d+)?\s*(?:seconds?|secs?|minutes?|mins?|hours?)\s*$",
    re.I,
)

def fallback_line(what):
    what = TIME_TAIL.sub("", (what or "").strip()).strip(" .")
    if not what:
        return "Hey, that's your timer."
    if len(what) > 1 and what[0].isupper() and what[1].islower():
        what = what[0].lower() + what[1:]
    return f"Hey, it's time to {what}."

def spoken_line(timers):
    lines = []
    for timer in timers:
        line = (timer.get("say") or "").strip() or fallback_line(timer.get("what") or "")
        if line and line[-1] not in ".!?":
            line += "."
        if line:
            lines.append(line)
    return " ".join(lines) or "Hey, that's your timer."

def announce_text(timers):
    line = spoken_line(timers)
    wait = nag_seconds(
        min(timer["original_seconds"] for timer in timers),
        max(timer["attempts"] for timer in timers),
    )
    return f"Say exactly this sentence and no other words: {line} Then call retry_in once with {wait}."

class Book:
    def __init__(self, path=PATH):
        self.path, self.timers, self.session_ids = path, [], []
        self.load()

    def load(self):
        try:
            data = json.loads(self.path.read_text())
        except (OSError, json.JSONDecodeError):
            data = {}
        now = time.time()
        timers = []
        for timer in data.get("timers") or []:
            if not timer.get("id") or not timer.get("what"):
                continue
            if timer.get("status") == "announcing":
                timer["status"] = "pending"
                timer["due_at"] = min(float(timer.get("due_at") or now), now)
            timers.append(timer)
        self.timers = timers

    def save(self):
        self.path.parent.mkdir(exist_ok=True)
        blob = json.dumps({"timers": self.timers}, indent=2)
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(blob)
        tmp.replace(self.path)

    def active(self):
        return [timer for timer in self.timers if timer["status"] in ("pending", "announcing")]

    def due(self):
        now = time.time()
        ready, kept, changed = [], [], False
        for timer in self.timers:
            waiting = timer["status"] == "pending" and float(timer["due_at"]) <= now
            if waiting and timer["attempts"] >= MAX_ATTEMPTS:
                print(f"\nstopped reminding — {timer['what']}", flush=True)
                changed = True
                continue
            kept.append(timer)
            if waiting:
                ready.append(timer)
        if changed:
            self.timers = kept
            self.save()
        return ready

    def claim_due(self):
        due = self.due()
        if not due:
            return []
        for timer in due:
            timer["status"] = "announcing"
            timer["attempts"] += 1
            if timer["id"] not in self.session_ids:
                self.session_ids.append(timer["id"])
        self.save()
        return due

    def add(self, minutes, what, said, say):
        seconds = int(round(minutes * 60))
        timer = {
            "id": uuid.uuid4().hex[:8],
            "what": what,
            "say": say,
            "said": said,
            "set_at": time.time(),
            "due_at": time.time() + seconds,
            "original_seconds": seconds,
            "attempts": 0,
            "status": "pending",
        }
        self.timers.append(timer)
        self.save()
        print(f"\ntimer {length_phrase(seconds)} — {say}", flush=True)
        return timer

    def match(self, what):
        active = self.active()
        wanted = (what or "").casefold().strip()
        if wanted in ("", "timer", "the timer", "it", "that"):
            return active if len(active) == 1 else []
        skip = {"the", "a", "to", "timer", "reminder", "my", "in"}
        words = [word for word in wanted.split() if word not in skip]
        hits = []
        for timer in active:
            hay = f"{timer['what']} {timer.get('say') or ''} {timer.get('said') or ''}".casefold()
            if wanted in hay or timer["what"].casefold() in wanted:
                hits.append(timer)
            elif words and all(word in hay for word in words):
                hits.append(timer)
        return hits if len(hits) == 1 else []

    def cancel(self, what):
        hits = self.match(what)
        if len(hits) != 1:
            return None
        timer = hits[0]
        self.finish([timer["id"]])
        print(f"\ncancelled — {timer['what']}", flush=True)
        return timer

    def retry(self, seconds):
        seconds = max(20, min(900, int(seconds)))
        hits = [timer for timer in self.timers if timer["status"] == "announcing"]
        if not hits:
            hits = [timer for timer in self.timers if timer["id"] in self.session_ids]
        if not hits:
            return None
        when = time.time() + seconds
        for timer in hits:
            timer["status"] = "pending"
            timer["due_at"] = when
        self.save()
        names = ", ".join(timer["what"] for timer in hits)
        print(f"\nagain in {seconds}s — {names}", flush=True)
        return {"seconds": seconds, "what": [timer["what"] for timer in hits]}

    def acknowledge(self, what=None):
        if what:
            hits = self.match(what)
        else:
            hits = [timer for timer in self.timers if timer["id"] in self.session_ids or timer["status"] == "announcing"]
        if not hits:
            return None
        self.finish([timer["id"] for timer in hits])
        names = ", ".join(timer["what"] for timer in hits)
        print(f"\nheard — {names}", flush=True)
        return {"what": [timer["what"] for timer in hits]}

    def finish(self, ids):
        gone = set(ids)
        if not gone:
            return
        self.timers = [timer for timer in self.timers if timer["id"] not in gone]
        self.session_ids = [item for item in self.session_ids if item not in gone]
        self.save()

    def fallback(self, ids):
        wanted = set(ids)
        changed = False
        for timer in self.timers:
            if timer["id"] not in wanted or timer["status"] != "announcing":
                continue
            gap = nag_seconds(timer["original_seconds"], timer["attempts"])
            timer["status"] = "pending"
            timer["due_at"] = time.time() + gap
            changed = True
            print(f"\nagain in {gap}s — {timer['what']}", flush=True)
        if changed:
            self.save()

    def unspoken(self, ids):
        wanted = set(ids)
        changed = False
        for timer in self.timers:
            if timer["id"] in wanted and timer["status"] == "announcing" and timer["attempts"]:
                timer["attempts"] -= 1
                changed = True
        if changed:
            self.save()

    def next_phrase(self):
        pending = [timer for timer in self.timers if timer["status"] == "pending"]
        if not pending:
            return ""
        soon = min(pending, key=lambda timer: float(timer["due_at"]))
        left = max(0, int(float(soon["due_at"]) - time.time()))
        return f", {soon['what']} in {left // 60}m{left % 60:02d}s"

    def rows(self):
        items = []
        for timer in self.active():
            left = max(0, int(float(timer["due_at"]) - time.time()))
            items.append({
                "what": timer["what"],
                "say": timer.get("say") or fallback_line(timer.get("what") or ""),
                "due_in": length_phrase(left),
                "set_at": clock(timer["set_at"]),
                "said": timer.get("said") or "",
            })
        return items

    def summary(self):
        return "\n".join(
            f"- {item['what']}: due in {item['due_in']}, set at {item['set_at']}."
            + (f" They said: {item['said']}." if item["said"] else "")
            for item in self.rows()
        )

class Timers:
    def __init__(self):
        self.book = Book()

    def handles(self, name):
        return name in ("set_timer", "cancel_timer", "list_reminders", "retry_in", "acknowledge")

    def instructions(self, _home):
        lines = [
            "You can set a short local timer: remind them in a few minutes, or a cooking timer.",
            "This is not a date on a calendar and not an errand list.",
            "When you call set_timer, you write the sentence that will be spoken when it goes off. The wait is minutes. It is not part of the task.",
            "\"Remind me to solve a Rubik's cube in 20 seconds\" waits about a third of a minute, then the task is solve a Rubik's cube. The sentence is about the cube, not about having 20 seconds to solve it.",
            "A cooking timer is different. \"Set a timer for 10 minutes\" has no other task, so the sentence can say the 10 minutes are up.",
            "Call cancel_timer when they cancel one.",
            "If they ask what reminders or timers are set, call list_reminders and answer from it. Reminders from earlier sessions are still there.",
            "When a reminder is due, say the sentence stored for it, in those words, then call retry_in once if they do not answer.",
            "A timer of about ten minutes or less is urgent: about 30 seconds, then a little longer.",
            "A longer one, such as a shower in forty five minutes, waits a couple of minutes, then a couple more.",
            "If they say they heard you — okay, thanks, got it — call acknowledge and do not call retry_in.",
        ]
        active = self.book.summary()
        lines.append("Reminders already waiting:\n" + active if active else "No reminders are waiting.")
        return "\n".join(lines)

    def schemas(self, _home):
        return [
            {
                "type": "function",
                "name": "set_timer",
                "description": "Start a local timer. You write, now, the exact sentence to speak when it goes off. The wait goes in minutes, not in that sentence, unless the timer itself is the whole request.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "minutes": {"type": "number", "description": "How long to wait. Twenty seconds is about 0.33. Twenty minutes is 20. This is the delay, not part of the task."},
                        "what": {"type": "string", "description": "The task only, with the wait left out. 'Remind me to solve a Rubik's cube in 20 seconds' is 'solve a Rubik's cube'."},
                        "say": {"type": "string", "description": "The exact sentence to speak when it goes off. For that cube: 'Hey, time to solve a Rubik's cube.' A bare 10 minute timer can be 'Your 10 minute timer is up.'"},
                        "said": {"type": "string", "description": "The sentence they just said."},
                    },
                    "required": ["minutes", "what", "say", "said"],
                },
            },
            {
                "type": "function",
                "name": "cancel_timer",
                "description": "Cancel one local timer.",
                "parameters": {
                    "type": "object",
                    "properties": {"what": {"type": "string"}},
                    "required": ["what"],
                },
            },
            {
                "type": "function",
                "name": "list_reminders",
                "description": "List the reminders and timers still waiting, including ones set in an earlier session.",
                "parameters": {"type": "object", "properties": {}},
            },
            {
                "type": "function",
                "name": "retry_in",
                "description": "Call once, after the reminder sentence has been spoken, if they have not answered. Do not call it again while they are talking. Seconds, not minutes.",
                "parameters": {
                    "type": "object",
                    "properties": {"seconds": {"type": "number"}},
                    "required": ["seconds"],
                },
            },
            {
                "type": "function",
                "name": "acknowledge",
                "description": "They heard the reminder. Okay, thanks, or got it. Do not schedule another reminder.",
                "parameters": {"type": "object", "properties": {"what": {"type": "string"}}},
            },
        ]

    async def call(self, name, args, _home):
        if name == "set_timer":
            minutes = minutes_of(args.get("minutes"))
            what = str(args.get("what") or "").strip()
            say = str(args.get("say") or "").strip()
            said = str(args.get("said") or "").strip()
            if minutes is None or not what or not say:
                return {"error": "need minutes, what, and say. say is the sentence to speak when it goes off. Put the wait in minutes, not in say."}
            if minutes < 0.25:
                return {"error": "too short. Use at least a quarter of a minute."}
            if minutes > 12 * 60:
                return {"error": "too long. Twelve hours is the longest."}
            timer = self.book.add(minutes, what, said, say)
            return {"set": timer["what"], "will_say": timer["say"], "minutes": round(minutes, 2), "due": clock(timer["due_at"])}
        if name == "cancel_timer":
            timer = self.book.cancel(args.get("what"))
            if timer is None:
                return {"error": "which timer?", "timers": [item["what"] for item in self.book.active()]}
            return {"cancelled": timer["what"]}
        if name == "list_reminders":
            return {"reminders": self.book.rows()}
        if name == "retry_in":
            try:
                seconds = float(args.get("seconds"))
            except (TypeError, ValueError):
                return {"error": "need seconds"}
            result = self.book.retry(seconds)
            if result is None:
                return {"error": "no reminder is waiting"}
            return result
        if name == "acknowledge":
            result = self.book.acknowledge(args.get("what"))
            if result is None:
                return {"error": "no reminder is waiting"}
            return result
        return {"error": f"unknown tool {name}"}
